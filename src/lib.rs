pub mod download;
pub mod model;
use hf_hub::api::tokio::ApiError;
use safetensors::tensor::SafeTensorError;
use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
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

#[derive(Clone, Deserialize)]
pub struct Config {
    num_attention_heads: usize,
    id2label: Option<HashMap<String, String>>,
}

impl Config {
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    pub fn id2label(&self) -> Option<&HashMap<String, String>> {
        self.id2label.as_ref()
    }
}

pub fn get_label(id2label: Option<&HashMap<String, String>>, i: usize) -> Option<String> {
    let id2label: &HashMap<String, String> = id2label?;
    let label: String = id2label.get(&format!("{}", i))?.to_string();
    Some(label)
}
