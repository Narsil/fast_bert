use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use fast_bert::{
    download::download,
    get_label,
    model::{BertClassifier, FromSafetensors},
    BertError, Config,
};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::SafeTensors;
use serde::{Deserialize, Serialize};
use serde_json::json;
#[cfg(feature = "cpu")]
use smelte_rs::cpu::f32::{Device, Tensor};
#[cfg(feature = "gpu")]
use smelte_rs::gpu::f32::{Device, Tensor};
use std::fs::File;
use std::net::SocketAddr;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tower_http::trace::{self, TraceLayer};
use tracing::{instrument, Level};

type OutMsg = Vec<Vec<Output>>;

struct InMsg {
    payload: String,
    tx: tokio::sync::oneshot::Sender<OutMsg>,
}

#[derive(Clone)]
struct AppState {
    tx: Arc<Mutex<SyncSender<InMsg>>>,
}

fn leak_buffer(buffer: Mmap) -> &'static [u8] {
    let buffer: &'static mut Mmap = Box::leak(Box::new(buffer));
    buffer
}

#[instrument]
async fn get_model(
    model_id: &str,
    filename: &str,
    num_heads: usize,
) -> Result<BertClassifier<Tensor>, BertError> {
    let max_files = 100;
    let chunk_size = 10_000_000;
    if !std::path::Path::new(filename).exists() {
        let url = format!("https://huggingface.co/{model_id}/resolve/main/model.safetensors");
        println!("Downloading {url:?} into {filename:?}");
        download(&url, filename, max_files, chunk_size).await?;
    }
    let file = File::open(filename)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let buffer: &'static [u8] = leak_buffer(buffer);
    let tensors: SafeTensors<'static> = SafeTensors::deserialize(buffer)?;
    let tensors: &'static SafeTensors<'static> = Box::leak(Box::new(tensors));

    #[cfg(feature = "gpu")]
    let device = Device::new(0).unwrap();
    #[cfg(feature = "cpu")]
    let device = Device {};

    let mut bert = BertClassifier::from_tensors(tensors, &device);
    bert.set_num_heads(num_heads);
    Ok(bert)
}

#[instrument]
async fn get_config(model_id: &str, filename: &str) -> Result<Config, BertError> {
    let max_files = 100;
    let chunk_size = 10_000_000;
    if !std::path::Path::new(filename).exists() {
        let url = format!("https://huggingface.co/{model_id}/resolve/main/config.json");
        println!("Downloading {url:?} into {filename:?}");
        download(&url, filename, max_files, chunk_size).await?;
    }
    let config_str: String = std::fs::read_to_string(filename).expect("Could not read config");
    let config: Config = serde_json::from_str(&config_str)?;
    Ok(config)
}

#[instrument]
async fn get_tokenizer(model_id: &str, filename: &str) -> Result<Tokenizer, BertError> {
    let max_files = 100;
    let chunk_size = 10_000_000;
    if !std::path::Path::new(filename).exists() {
        let url = format!("https://huggingface.co/{model_id}/resolve/main/tokenizer.json");
        println!("Downloading {url:?} into {filename:?}");
        download(&url, filename, max_files, chunk_size).await?;
    }
    Ok(Tokenizer::from_file(filename).unwrap())
}

#[cfg(feature = "gpu")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), BertError> {
    start().await
}

#[cfg(feature = "cpu")]
#[tokio::main]
async fn main() -> Result<(), BertError> {
    start().await
}

fn server_loop(rx: Receiver<InMsg>) -> Result<(), BertError> {
    tracing::debug!("Starting server loop");
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").unwrap();

    tracing::debug!("Working directory {:?}", std::env::current_dir());
    tracing::debug!("{:?}", std::env::current_dir());
    if !std::path::Path::new("models/config.json").exists() {
        panic!("Bad mount");
    }

    let config_str = std::fs::read_to_string("models/config.json").unwrap();
    let config: Config = serde_json::from_str(&config_str).unwrap();
    let file = File::open("models/model.safetensors").unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let buffer: &'static [u8] = leak_buffer(buffer);
    let tensors: SafeTensors<'static> = SafeTensors::deserialize(buffer)?;
    let tensors: &'static SafeTensors<'static> = Box::leak(Box::new(tensors));

    #[cfg(feature = "gpu")]
    let device = Device::new(0).unwrap();
    #[cfg(feature = "cpu")]
    let device = Device {};

    let mut model = BertClassifier::from_tensors(tensors, &device);
    model.set_num_heads(config.num_attention_heads());

    tracing::debug!("Loaded server loop");
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
            let input_ids: Vec<_> = encoded.get_ids().iter().map(|i| *i as usize).collect();
            let position_ids: Vec<_> = (0..input_ids.len()).collect();
            let type_ids: Vec<_> = encoded.get_type_ids().iter().map(|i| *i as usize).collect();
            let probs = model.run(input_ids, position_ids, type_ids).unwrap();
            let id2label = config.id2label();
            #[cfg(feature = "gpu")]
            let mut outputs: Vec<_> = probs
                .cpu_data()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, &p)| Output {
                    label: get_label(id2label, i).unwrap_or(format!("LABEL_{}", i)),
                    score: p,
                })
                .collect();
            #[cfg(feature = "cpu")]
            let mut outputs: Vec<_> = probs
                .data()
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
    get_tokenizer(&model_id, "models/tokenizer.json").await?;
    let config = get_config(&model_id, "models/config.json").await?;
    get_model(
        &model_id,
        "models/model.safetensors",
        config.num_attention_heads(),
    )
    .await?;

    let (tx, rx) = std::sync::mpsc::sync_channel(queue_size);

    tokio::task::spawn_blocking(move || {
        server_loop(rx).unwrap();
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
    Json(rx.await.unwrap()).into_response()
}
