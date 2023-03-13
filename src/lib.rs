pub mod download;
pub mod model;
use crate::download::download;
use crate::model::Bert;
use memmap2::MmapOptions;
use safetensors::tensor::{SafeTensorError, SafeTensors};
use serde::Deserialize;
use smelt::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use thiserror::Error;
use tokenizers::Tokenizer;

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

pub async fn run() -> Result<(), BertError> {
    let start = std::time::Instant::now();

    let model_id = std::env::var("MODEL_ID")
        .expect("Please set MODEL_ID in your environment to load the correct model");

    let model_id_slug = model_id.replace('/', "-");

    let filename = format!("model-{model_id_slug}.safetensors");
    let max_files = 100;
    let chunk_size = 10_000_000;
    if !std::path::Path::new(&filename).exists() {
        let revision = "main";
        let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors");
        println!("Downloading {url:?} into {filename:?}");
        download(&url, &filename, max_files, chunk_size).await?;
    }

    let file = File::open(filename)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;
    println!("Safetensors {:?}", start.elapsed());

    let filename = format!("tokenizer-{model_id_slug}.json");
    if !std::path::Path::new(&filename).exists() {
        let url = format!("https://huggingface.co/{model_id}/resolve/main/tokenizer.json");
        println!("Downloading {url:?} into {filename:?}");
        download(&url, &filename, max_files, chunk_size).await?;
    }
    let tokenizer = Tokenizer::from_file(filename).unwrap();
    println!("Tokenizer {:?}", start.elapsed());

    let filename = format!("config-{model_id_slug}.json");
    if !std::path::Path::new(&filename).exists() {
        let revision = "main";
        let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/config.json");
        println!("Downloading {url:?} into {filename:?}");
        download(&url, &filename, max_files, chunk_size).await?;
    }
    let config_str: String = std::fs::read_to_string(filename).expect("Could not read config");
    let config: Config = serde_json::from_str(&config_str).expect("Could not parse Config");

    let bert = Bert::from_tensors(&tensors, config.num_attention_heads);
    println!("Loaded {:?}", start.elapsed());

    let default_string = "test eqwlewqk ewqlke qwlkeqwl ewlqke qwlke eklqwekwqlek qwlkeqwl ekqwlk eqwlke qwlke qwlke qwlkelqw elqwkelwk elkw elkqwel qwel qwle kqwejqwkehjqwjkeh qwjkhe qwjkhekqweh qwjkeh qwjkeh qwkje";
    let string = std::env::var("STRING").unwrap_or(default_string.to_string());

    let encoded = tokenizer.encode(string, false).unwrap();
    let encoded = tokenizer.post_process(encoded, None, true).unwrap();
    println!("Loaded & encoded {:?}", start.elapsed());
    for _ in 0..5 {
        let inference_start = std::time::Instant::now();
        let probs = bert.forward(&encoded);

        let id2label = config.id2label();
        let outputs: Vec<_> = probs
            .data()
            .iter()
            .enumerate()
            .map(|(i, &p)| (get_label(id2label, i).unwrap_or(format!("LABEL_{}", i)), p))
            .collect();
        println!("Probs {:?}", outputs);
        println!("Inference in {:?}", inference_start.elapsed());
    }
    println!("Total Inference {:?}", start.elapsed());
    Ok(())
}

pub fn get_label(id2label: Option<&HashMap<String, String>>, i: usize) -> Option<String> {
    let id2label: &HashMap<String, String> = id2label?;
    let label: String = id2label.get(&format!("{}", i))?.to_string();
    Some(label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_logits() {
        let num_heads = 12;
        let filename = "model.safetensors";
        let file = File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();

        let filename = "tokenizer.json";
        let tokenizer = Tokenizer::from_file(filename).unwrap();
        let bert = Bert::from_tensors(&tensors, num_heads);
        let string = "My name is";
        let encoded = tokenizer.encode(string, false).unwrap();
        let _logits = bert.forward(&encoded);
    }
}
