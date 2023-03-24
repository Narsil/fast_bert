use memmap2::MmapOptions;
use safetensors::{
    tensor::{Dtype, SafeTensorError, TensorView},
    SafeTensors,
};
use serde::Deserialize;
use smelte_rs::gpu::f32::{Device, Tensor};
use smelte_rs::nn::layers::{Embedding, LayerNorm, Linear};
pub use smelte_rs::nn::models::bert::BertClassifier;
use smelte_rs::nn::models::bert::{
    Bert, BertAttention, BertEmbeddings, BertEncoder, BertLayer, BertPooler, Mlp,
};
use smelte_rs::SmeltError;
use std::borrow::Cow;
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

pub trait FromSafetensors<'a> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized;
}

fn to_tensor<'data>(view: TensorView<'data>, device: &Device) -> Result<Tensor, SmeltError> {
    let shape = view.shape().to_vec();
    let data = to_f32(view);
    Tensor::from_cpu(&data, shape, device)
}

pub fn to_f32<'data>(view: TensorView<'data>) -> Cow<'data, [f32]> {
    assert_eq!(view.dtype(), Dtype::F32);
    let v = view.data();
    if (v.as_ptr() as usize) % 4 == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[f32] =
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) };
        Cow::Borrowed(data)
    } else {
        let mut c = Vec::with_capacity(v.len() / 4);
        let mut i = 0;
        while i < v.len() {
            c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
            i += 4;
        }
        Cow::Owned(c)
    }
}

fn linear_from<'a>(
    weights: TensorView<'a>,
    bias: TensorView<'a>,
    device: &Device,
) -> Linear<Tensor> {
    Linear::new(
        to_tensor(weights, device).unwrap(),
        to_tensor(bias, device).unwrap(),
    )
}

fn linear_from_prefix<'a>(
    prefix: &str,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> Linear<Tensor> {
    linear_from(
        tensors.tensor(&format!("{}.weight", prefix)).unwrap(),
        tensors.tensor(&format!("{}.bias", prefix)).unwrap(),
        device,
    )
}

fn embedding_from<'a>(weights: TensorView<'a>, device: &Device) -> Embedding<Tensor> {
    Embedding::new(to_tensor(weights, device).unwrap())
}

impl<'a> FromSafetensors<'a> for BertClassifier<Tensor> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized,
    {
        let pooler = BertPooler::from_tensors(tensors, device);
        let bert = Bert::from_tensors(tensors, device);
        let (weight, bias) = if let (Ok(weight), Ok(bias)) = (
            tensors.tensor("classifier.weight"),
            tensors.tensor("classifier.bias"),
        ) {
            (weight, bias)
        } else {
            (
                tensors.tensor("cls.seq_relationship.weight").unwrap(),
                tensors.tensor("cls.seq_relationship.bias").unwrap(),
            )
        };
        let classifier = linear_from(weight, bias, device);
        Self::new(bert, pooler, classifier)
    }
}
impl<'a> FromSafetensors<'a> for BertPooler<Tensor> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized,
    {
        let pooler = linear_from(
            tensors.tensor("bert.pooler.dense.weight").unwrap(),
            tensors.tensor("bert.pooler.dense.bias").unwrap(),
            device,
        );
        Self::new(pooler)
    }
}

impl<'a> FromSafetensors<'a> for Bert<Tensor> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized,
    {
        let embeddings = BertEmbeddings::from_tensors(tensors, device);
        let encoder = BertEncoder::from_tensors(tensors, device);
        Bert::new(embeddings, encoder)
    }
}

impl<'a> FromSafetensors<'a> for BertEmbeddings<Tensor> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized,
    {
        let input_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.word_embeddings.weight")
                .unwrap(),
            device,
        );
        let position_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.position_embeddings.weight")
                .unwrap(),
            device,
        );
        let type_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.token_type_embeddings.weight")
                .unwrap(),
            device,
        );

        let layer_norm = layer_norm_from_prefix("bert.embeddings.LayerNorm", &tensors, device);
        BertEmbeddings::new(
            input_embeddings,
            position_embeddings,
            type_embeddings,
            layer_norm,
        )
    }
}

fn bert_layer_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> BertLayer<Tensor> {
    let attention = bert_attention_from_tensors(index, tensors, device);
    let mlp = bert_mlp_from_tensors(index, tensors, device);
    BertLayer::new(attention, mlp)
}
fn bert_attention_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> BertAttention<Tensor> {
    let query = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.query"),
        tensors,
        device,
    );
    let key = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.key"),
        tensors,
        device,
    );
    let value = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.value"),
        tensors,
        device,
    );
    let output = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.output.dense"),
        tensors,
        device,
    );
    let output_ln = layer_norm_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.output.LayerNorm"),
        &tensors,
        device,
    );
    BertAttention::new(query, key, value, output, output_ln)
}

fn bert_mlp_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> Mlp<Tensor> {
    let intermediate = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.intermediate.dense"),
        tensors,
        device,
    );
    let output = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.output.dense"),
        tensors,
        device,
    );
    let output_ln = layer_norm_from_prefix(
        &format!("bert.encoder.layer.{index}.output.LayerNorm"),
        &tensors,
        device,
    );
    Mlp::new(intermediate, output, output_ln)
}

fn layer_norm_from_prefix<'a>(
    prefix: &str,
    tensors: &'a SafeTensors<'a>,
    device: &Device,
) -> LayerNorm<Tensor> {
    let epsilon = 1e-5;
    if let (Ok(weight), Ok(bias)) = (
        tensors.tensor(&format!("{}.weight", prefix)),
        tensors.tensor(&format!("{}.bias", prefix)),
    ) {
        LayerNorm::new(
            to_tensor(weight, device).unwrap(),
            to_tensor(bias, device).unwrap(),
            epsilon,
        )
    } else {
        LayerNorm::new(
            to_tensor(
                tensors.tensor(&format!("{}.gamma", prefix)).unwrap(),
                device,
            )
            .unwrap(),
            to_tensor(tensors.tensor(&format!("{}.beta", prefix)).unwrap(), device).unwrap(),
            epsilon,
        )
    }
}

impl<'a> FromSafetensors<'a> for BertEncoder<Tensor> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, device: &Device) -> Self
    where
        Self: Sized,
    {
        // TODO ! Count heads from tensors present
        let layers: Vec<_> = (0..12)
            .map(|i| bert_layer_from_tensors(i, tensors, device))
            .collect();
        Self::new(layers)
    }
}
