use axum::{
    extract::State,
    http::{StatusCode, HeaderMap, HeaderName, header::ACCESS_CONTROL_EXPOSE_HEADERS},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};

mod model;
use model::{BertModel, Config};
use std::collections::HashMap;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use candle::{DType, Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use safetensors::tensor::SafeTensors;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tower_http::trace::{self, TraceLayer};
use tracing::{instrument, Level};
use hf_hub::api::tokio::ApiError;
use safetensors::tensor::SafeTensorError;

type OutMsg = Vec<Vec<Output>>;

struct InMsg {
    payload: String,
    tx: tokio::sync::oneshot::Sender<OutMsg>,
}

#[derive(Clone)]
struct AppState {
    tx: Arc<Mutex<SyncSender<InMsg>>>,
}

pub fn get_label(id2label: Option<&HashMap<String, String>>, i: usize) -> Option<String> {
    let id2label: &HashMap<String, String> = id2label?;
    let label: String = id2label.get(&format!("{}", i))?.to_string();
    Some(label)
}

#[derive(Debug, thiserror::Error)]
pub enum BertError {
    #[error("i/o error")]
    IOError(#[from] std::io::Error),
    #[error("safetensor error")]
    SafeTensorError(#[from] SafeTensorError),
    #[error("slice error")]
    Slice(#[from] std::array::TryFromSliceError),
    #[error("parsing int error")]
    ParseIntError(#[from] core::num::ParseIntError),
    #[error("Hub api error")]
    RequestError(#[from] ApiError),
    #[error("JSON parsing error")]
    JSONError(#[from] serde_json::Error),
}


#[tokio::main]
async fn main() -> Result<(), BertError> {
    start().await
}

fn server_loop(config: Config, tokenizer: Tokenizer, model: BertModel, rx: Receiver<InMsg>) -> Result<(), BertError> {
    tracing::debug!("Starting server loop");
    loop {
        if let Ok(InMsg { payload, tx }) = rx.recv() {
            tracing::debug!("Recived {payload:?}");
            let payload: Inputs = if let Ok(payload) = serde_json::from_str(&payload) {
                payload
            } else {
                Inputs {
                    inputs: payload,
                    ..Default::default()
                }
            };
            let encoded = tokenizer.encode(payload.inputs, false).unwrap();
            let encoded = tokenizer.post_process(encoded, None, true).unwrap();

            let device = Device::Cpu;
            let input_ids = Tensor::new(encoded.get_ids(), &device).unwrap().unsqueeze(0).unwrap();
            let type_ids = Tensor::new(encoded.get_type_ids(), &device).unwrap().unsqueeze(0).unwrap();
            let probs = model.forward(&input_ids, &type_ids).unwrap();
            let id2label = config.id2label();
            let mut outputs: Vec<_> = probs.i(0).unwrap()
                .to_vec1::<f32>().unwrap()
                .iter()
                .enumerate()
                .map(|(i, &p)| Output {
                    label: get_label(id2label, i).unwrap_or(format!("LABEL_{}", i)),
                    score: p,
                })
                .collect();
            outputs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            if let Some(top_k) = payload.parameters.top_k {
                outputs = outputs.into_iter().take(top_k).collect()
            }
            if let Err(_err) = tx.send(vec![outputs]) {
                println!("The receiver dropped");
            }
        }
    }
}
async fn start() -> Result<(), BertError> {
    // initialize tracing
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "fast_bert=debug,tower_http=debug")
    }
    tracing_subscriber::fmt::init();
    let queue_size = 2;

    let model_id: String = std::env::var("MODEL_ID").expect("MODEL_ID is not defined");
    let api = Api::new().unwrap();
    let repo = Repo::new(model_id, RepoType::Model);
    let tokenizer = Tokenizer::from_file(api.get(&repo, "tokenizer.json").await?).unwrap();

    let config = api.get(&repo, "config.json").await?;
    let config: String = std::fs::read_to_string(config).expect("Could not read config");
    let config: Config = serde_json::from_str(&config)?;

    let model = api.get(&repo, "model.safetensors").await?;
    let buffer = std::fs::read(model)?;
    let tensors: SafeTensors<'_> = SafeTensors::deserialize(&buffer)?;
    let device = Device::Cpu;
    let vb = VarBuilder::from_safetensors(vec![tensors], DType::F32, &device);
    let model = BertModel::load(vb, &config).unwrap();

    let (tx, rx) = std::sync::mpsc::sync_channel(queue_size);

    tokio::task::spawn_blocking(move || {
        server_loop(config, tokenizer, model, rx).unwrap();
    });

    let state = AppState {
        tx: Arc::new(Mutex::new(tx)),
    };

    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        // .route("/", get(root))
        // `POST /users` goes to `create_user`
        .route("/", post(inference))
        .route("/", get(health))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(trace::DefaultMakeSpan::new().level(Level::INFO))
                .on_response(trace::DefaultOnResponse::new().level(Level::INFO)),
        )
        .with_state(state);

    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()?;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
    Ok(())
}

#[derive(Deserialize, Default)]
struct Parameters {
    top_k: Option<usize>,
}

// the input to our `create_user` handler
#[derive(Deserialize, Default)]
struct Inputs {
    inputs: String,
    #[serde(default)]
    parameters: Parameters,
}

// the output to our `create_user` handler
#[derive(Debug, Serialize)]
struct Output {
    label: String,
    score: f32,
}

async fn health() -> impl IntoResponse {
    "Ok"
}

#[axum_macros::debug_handler]
async fn inference(State(state): State<AppState>, payload: String) -> impl IntoResponse {
    let start = std::time::Instant::now();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let msg = InMsg { payload, tx };
    {
        let stx = state.tx.lock().unwrap();
        if let Err(_) = (*stx).try_send(msg) {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "Queue is full"})),
            )
                .into_response();
        }
    }
    let receive = rx.await.unwrap();
    let time = start.elapsed().as_secs_f32().to_string();
    let mut headers = HeaderMap::new();
    let header_time = HeaderName::from_static("x-compute-time");
    let header_type = HeaderName::from_static("x-compute-type");
    headers.insert(header_time.clone(), time.parse().unwrap());
    headers.insert(header_type.clone(), "cpu".parse().unwrap());
    headers.insert(ACCESS_CONTROL_EXPOSE_HEADERS, format!("{header_time}, {header_type}").parse().unwrap());
    (headers, Json(receive)).into_response()
}
