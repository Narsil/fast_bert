use axum::{
    extract::State,
    http::{
        header::{ACCESS_CONTROL_EXPOSE_HEADERS, CONTENT_TYPE},
        HeaderMap, HeaderName, Request, StatusCode,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{
    api::tokio::{ApiBuilder, ApiError, ApiRepo},
    Repo, RepoType,
};
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::sdk::{trace, Resource};
use opentelemetry::{global, sdk::propagation::TraceContextPropagator, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use safetensors::tensor::{SafeTensorError, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tower_http::trace::{DefaultOnRequest, DefaultOnResponse, MakeSpan, TraceLayer};
use tracing::{info_span, instrument, Instrument};
use tracing::{Level, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

mod model;
use model::{BertModel, Config, Pool, PoolConfig};

type OutMsg = Vec<f32>;

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
    #[error("Cande Error: {0}")]
    CandleError(#[from] candle::Error),
}

#[tokio::main]
async fn main() -> Result<(), BertError> {
    start().await
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor, BertError> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

fn run(
    payload: String,
    tx: tokio::sync::oneshot::Sender<OutMsg>,
    tokenizer: &Tokenizer,
    model: &BertModel,
    pool: Pool,
) -> Result<(), BertError> {
    tracing::debug!("Received {payload:?}");
    let payload: Inputs = serde_json::from_str(&payload)?;
    let mut batch = vec![payload.inputs.source_sentence];
    batch.extend(payload.inputs.sentences);

    let device = Device::Cpu;
    let embeddings: Result<Vec<Tensor>, BertError> = batch
        .into_iter()
        .map(|s| -> Result<Tensor, BertError> {
            let encoded = tokenizer.encode(s, true).unwrap();

            let input_ids = Tensor::new(encoded.get_ids(), &device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();

            let type_ids = Tensor::new(encoded.get_type_ids(), &device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let embeddings = model.forward(&input_ids, &type_ids)?;
            let embeddings = match pool {
                Pool::Cls => embeddings.i((.., 0))?,
                Pool::Mean => {
                    let n = embeddings.dims()[1];
                    (embeddings.sum(1)? / (n as f64))?
                }
                _ => todo!(),
            };
            // tracing::info!("After mean: {embeddings}");
            // Normalize
            let embeddings = normalize_l2(&embeddings)?;
            // tracing::info!("After normalize: {embeddings}");
            // Remove batch
            let embeddings = embeddings.i(0)?;

            Ok(embeddings)
        })
        .collect();
    let embeddings = embeddings?;
    let embeddings = Tensor::stack(&embeddings, 0)?;
    let source = embeddings.i(0..1)?;
    let queries = embeddings.i(1..)?;

    let norm_source = &source.sqr()?.sum(1)?.sqrt()?.unsqueeze(0)?;
    let norm_queries = &queries.sqr()?.sum(1)?.sqrt()?.unsqueeze(0)?;

    // tracing::info!("Norms {norm_source} - queries {norm_queries}");

    let similarities = source
        .matmul(&queries.t()?)?
        .broadcast_div(norm_queries)?
        .broadcast_div(norm_source)?;
    let similarities = similarities.i(0)?.to_vec1::<f32>()?;

    if let Err(_err) = tx.send(similarities) {
        tracing::warn!("The receiver dropped");
    }
    Ok(())
}

fn server_loop(
    tokenizer: Tokenizer,
    model: BertModel,
    pool: Pool,
    rx: Receiver<InMsg>,
) -> Result<(), BertError> {
    tracing::info!("Starting server loop");
    loop {
        if let Ok(InMsg { payload, tx }) = rx.recv() {
            if let Err(err) = run(payload, tx, &tokenizer, &model, pool) {
                tracing::error!("{}", err);
            }
        }
    }
}

/// Init logging using env variables OTLP_ENDPOINT, LOG_LEVEL and LOG_FORMAT
/// OTLP_ENDPOINT is an optional URL to an Open Telemetry collector
/// LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
/// LOG_FORMAT may be TEXT or JSON (default to TEXT)
fn init_logging() {
    let mut layers = Vec::new();

    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true);

    let json = std::env::var("LOG_FORMAT")
        .map(|value| value.to_lowercase() == "json")
        .unwrap_or(false);

    let fmt_layer = match json {
        true => fmt_layer
            .json()
            .flatten_event(true)
            .with_span_list(false)
            .boxed(),
        false => fmt_layer.boxed(),
    };
    layers.push(fmt_layer);

    // OpenTelemetry tracing layer
    if let Ok(otlp_endpoint) = std::env::var("OTLP_ENDPOINT") {
        global::set_text_map_propagator(TraceContextPropagator::new());
        init_tracing_opentelemetry::init_propagator().unwrap();
        let model_id: String = std::env::var("MODEL_ID").expect("MODEL_ID is not defined");

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint),
            )
            .with_trace_config(trace::config().with_resource(Resource::new(vec![
                KeyValue::new("service.name", "api-inference.fast_bert"),
                KeyValue::new("model_id", model_id),
            ])))
            .install_batch(opentelemetry::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
        };
    }

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();
}

#[instrument(skip_all)]
async fn download(api: ApiRepo) -> Result<(Pool, Config, Tokenizer, PathBuf), BertError> {
    let start = std::time::Instant::now();
    let mut tokenizer = Tokenizer::from_file(api.get("tokenizer.json").await?).unwrap();
    tokenizer.with_padding(None);
    let config = api.get("config.json").await?;
    let config: String = std::fs::read_to_string(config).expect("Could not read config");
    let config: Config = serde_json::from_str(&config)?;
    let pool_config = api.get("1_Pooling/config.json").await?;
    let pool_config = std::fs::read_to_string(pool_config).expect("Could not read config");
    let pool_config: PoolConfig = serde_json::from_str(&pool_config)?;
    let pool: Pool = pool_config.into();
    let model = api.get("model.safetensors").await?;
    tracing::info!("Download took {:?}", start.elapsed());
    Ok((pool, config, tokenizer, model))
}

fn reinterpret_or_copy(v: &[u8]) -> Vec<f32> {
    let num_bytes = std::mem::size_of::<f32>();
    if (v.as_ptr() as usize) % num_bytes == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let size = v.len() / num_bytes;
        unsafe { Vec::from_raw_parts(v.as_ptr() as *mut f32, size, size) }
    } else {
        let mut c = Vec::with_capacity(v.len() / num_bytes);
        let mut i = 0;
        while i < v.len() {
            c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
            i += num_bytes;
        }
        c
    }
}

fn load_tensor(view: TensorView) -> Tensor {
    let data = view.data();
    let data = match view.dtype() {
        safetensors::Dtype::F32 => reinterpret_or_copy(data),
        _ => {
            return Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        }
    };
    Tensor::from_vec(data, view.shape(), &Device::Cpu).unwrap()
}

#[instrument(skip_all)]
async fn load(path: PathBuf, config: &Config) -> Result<BertModel, BertError> {
    let start = std::time::Instant::now();
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let mmap: &'static [u8] = Box::leak(Box::new(mmap));
    let tensors: SafeTensors<'_> = SafeTensors::deserialize(&mmap)?;
    let tensors: HashMap<String, Tensor> = tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, load_tensor(view)))
        .collect();
    let device = Device::Cpu;
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let bert = BertModel::load(vb, &config).unwrap();
    tracing::info!("Load took {:?}", start.elapsed());
    Ok(bert)
}

#[instrument(skip_all)]
async fn init() -> Result<(Tokenizer, BertModel, Pool), BertError> {
    let model_id: String = std::env::var("MODEL_ID").expect("MODEL_ID is not defined");
    let revision: String = std::env::var("REVISION").unwrap_or("main".to_string());
    let token: Option<String> = std::env::var("HF_API_TOKEN").ok();
    let mut builder = ApiBuilder::new().with_progress(false).with_token(token);
    if let Ok(cache_dir) = std::env::var("TRANSFORMERS_CACHE") {
        builder = builder.with_cache_dir(cache_dir.into());
    }
    let api = builder.build().unwrap();
    let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let (pool, config, tokenizer, model) = download(api).await?;

    let model = load(model, &config).await?;
    Ok((tokenizer, model, pool))
}

/// The default way [`Span`]s will be created for [`Trace`].
///
/// [`Span`]: tracing::Span
/// [`Trace`]: super::Trace
#[derive(Debug, Clone)]
pub struct DefaultMakeSpan {
    level: Level,
    include_headers: bool,
}

impl DefaultMakeSpan {
    /// Create a new `DefaultMakeSpan`.
    pub fn new() -> Self {
        Self {
            level: Level::INFO,
            include_headers: false,
        }
    }

    /// Set the [`Level`] used for the [tracing span].
    ///
    /// Defaults to [`Level::DEBUG`].
    ///
    /// [tracing span]: https://docs.rs/tracing/latest/tracing/#spans
    pub fn level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// Include request headers on the [`Span`].
    ///
    /// By default headers are not included.
    ///
    /// [`Span`]: tracing::Span
    pub fn include_headers(mut self, include_headers: bool) -> Self {
        self.include_headers = include_headers;
        self
    }
}

impl Default for DefaultMakeSpan {
    fn default() -> Self {
        Self::new()
    }
}

impl<B> MakeSpan<B> for DefaultMakeSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // This ugly macro is needed, unfortunately, because `tracing::span!`
        // required the level argument to be static. Meaning we can't just pass
        // `self.level`.
        let traceparent: HeaderName = "traceparent".try_into().unwrap();
        let tracestate: HeaderName = "tracestate".try_into().unwrap();
        let traceparent = request
            .headers()
            .get(traceparent)
            .map(|v| v.to_str().unwrap());
        let tracestate = request
            .headers()
            .get(tracestate)
            .map(|v| v.to_str().unwrap());
        let mut fields: HashMap<String, String> = HashMap::new();
        if let Some(traceparent) = traceparent {
            fields.insert("traceparent".to_string(), traceparent.to_string());
        }
        if let Some(tracestate) = tracestate {
            fields.insert("tracestate".to_string(), tracestate.to_string());
        }

        let propagator = TraceContextPropagator::new();
        let context = propagator.extract(&fields);
        macro_rules! make_span {
            ($level:expr) => {
                if self.include_headers {
                    tracing::span!(
                        $level,
                        "request",
                        method = %request.method(),
                        uri = %request.uri(),
                        version = ?request.version(),
                        headers = ?request.headers(),
                    )
                } else {
                    tracing::span!(
                        $level,
                        "request",
                        method = %request.method(),
                        uri = %request.uri(),
                        version = ?request.version(),
                    )
                }
            }
        }

        let span = match self.level {
            Level::ERROR => make_span!(Level::ERROR),
            Level::WARN => make_span!(Level::WARN),
            Level::INFO => make_span!(Level::INFO),
            Level::DEBUG => make_span!(Level::DEBUG),
            Level::TRACE => make_span!(Level::TRACE),
        };
        span.set_parent(context);
        span
    }
}

async fn start() -> Result<(), BertError> {
    // initialize tracing
    init_logging();

    let mut fields: HashMap<String, String> = HashMap::new();
    let traceparent = std::env::var("TRACEPARENT");
    let tracestate = std::env::var("TRACESTATE");
    if let Ok(traceparent) = traceparent {
        fields.insert("traceparent".to_string(), traceparent.to_string());
    }
    if let Ok(tracestate) = tracestate {
        fields.insert("tracestate".to_string(), tracestate.to_string());
    }

    let propagator = TraceContextPropagator::new();
    let context = propagator.extract(&fields);
    let span = info_span!("init");
    span.set_parent(context);

    let (tokenizer, model, pool) = init().instrument(span).await?;

    let queue_size = 2;

    let (tx, rx) = std::sync::mpsc::sync_channel(queue_size);

    tokio::task::spawn_blocking(move || {
        server_loop(tokenizer, model, pool, rx).unwrap();
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
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(
                    DefaultMakeSpan::new()
                        .level(Level::INFO)
                        .include_headers(true),
                )
                .on_request(DefaultOnRequest::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .route("/", get(health))
        // .layer(OtelInResponseLayer::default())
        // //start OpenTelemetry trace on incoming request
        // .layer(OtelAxumLayer::default())
        .with_state(state);

    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()?;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
    Ok(())
}

// #[derive(Deserialize, Default)]
// struct Parameters {
//     top_k: Option<usize>,
// }
#[derive(Deserialize, Default)]
struct Similarity {
    sentences: Vec<String>,
    source_sentence: String,
}

// the input to our `create_user` handler
#[derive(Deserialize, Default)]
struct Inputs {
    inputs: Similarity,
    // #[serde(default)]
    // parameters: Parameters,
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

#[instrument(skip_all)]
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
    headers.insert(CONTENT_TYPE, "application/safetensors".parse().unwrap());
    headers.insert(header_time.clone(), time.parse().unwrap());
    headers.insert(header_type.clone(), "cpu".parse().unwrap());
    headers.insert(
        ACCESS_CONTROL_EXPOSE_HEADERS,
        format!("{header_time}, {header_type}").parse().unwrap(),
    );
    (headers, Json(receive)).into_response()
}
