pub mod download;
pub mod model;
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
    #[error("reqwest int error")]
    RequestError(#[from] reqwest::Error),
    #[error("reqwest header cannot be parsed error")]
    HeaderToStrError(#[from] reqwest::header::ToStrError),
    #[error("Cannot acquire semaphore")]
    AcquireError(#[from] tokio::sync::AcquireError),
    #[error("No content length")]
    NoContentLength,
    #[error("Could not parse content length")]
    ParseContentLength,
    #[error("JSON parsing error")]
    JSONError(#[from] serde_json::Error),
}

#[derive(Clone, Deserialize)]
pub struct Config {
    num_attention_heads: usize,
    id2label: Option<HashMap<String, String>>,
}

impl Config {
    pub fn id2label(&self) -> Option<&HashMap<String, String>> {
        self.id2label.as_ref()
    }
}

pub fn get_label(id2label: Option<&HashMap<String, String>>, i: usize) -> Option<String> {
    let id2label: &HashMap<String, String> = id2label?;
    let label: String = id2label.get(&format!("{}", i))?.to_string();
    Some(label)
}
