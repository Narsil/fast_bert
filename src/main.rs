use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use fast_bert::{
    download::download,
    get_label,
    model::smelt::{BertClassifier, FromSafetensors},
    BertError, Config,
};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::SafeTensors;
use serde::{Deserialize, Serialize};
use smelte_rs::cpu::f32::Tensor;
use std::fs::File;
use std::net::SocketAddr;
use tokenizers::Tokenizer;
use tower_http::trace::{self, TraceLayer};
use tracing::{instrument, Level};

#[derive(Clone)]
struct AppState {
    model: BertClassifier<Tensor<'static>>,
    config: Config,
    tokenizer: Tokenizer,
}

fn leak_buffer(buffer: Mmap) -> &'static [u8] {
    let buffer: &'static mut Mmap = Box::leak(Box::new(buffer));
    buffer
}

#[instrument]
async fn get_model(
    model_id: &str,
    filename: &str,
) -> Result<BertClassifier<Tensor<'static>>, BertError> {
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
    let num_heads = 12;
    let mut bert = BertClassifier::from_tensors(tensors);
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

#[tokio::main]
async fn main() -> Result<(), BertError> {
    // initialize tracing
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "fast_bert=debug,tower_http=debug")
    }
    tracing_subscriber::fmt::init();
    let model_id: String = std::env::var("MODEL_ID").expect("MODEL_ID is not defined");
    let model = get_model(&model_id, "model.safetensors").await?;
    let tokenizer = get_tokenizer(&model_id, "tokenizer.json").await?;
    let config = get_config(&model_id, "config.json").await?;

    let state = AppState {
        model,
        tokenizer,
        config,
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
#[derive(Serialize)]
struct Output {
    label: String,
    score: f32,
}

async fn health() -> impl IntoResponse {
    "Ok"
}

async fn inference((State(state), payload): (State<AppState>, String)) -> impl IntoResponse {
    let payload: Inputs = if let Ok(payload) = serde_json::from_str(&payload) {
        payload
    } else {
        Inputs {
            inputs: payload,
            ..Default::default()
        }
    };
    let tokenizer = state.tokenizer;
    let encoded = tokenizer.encode(payload.inputs, false).unwrap();
    let encoded = tokenizer.post_process(encoded, None, true).unwrap();
    let input_ids: Vec<_> = encoded.get_ids().iter().map(|i| *i as usize).collect();
    let position_ids: Vec<_> = (0..input_ids.len()).collect();
    let type_ids: Vec<_> = encoded.get_type_ids().iter().map(|i| *i as usize).collect();
    let probs = state.model.run(input_ids, position_ids, type_ids).unwrap();
    let id2label = state.config.id2label();
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
    Json(vec![outputs])
}
