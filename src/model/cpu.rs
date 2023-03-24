use safetensors::{
    tensor::{Dtype, TensorView},
    SafeTensors,
};
use smelte_rs::nn::layers::{Embedding, LayerNorm, Linear};
pub use smelte_rs::nn::models::bert::BertClassifier;
use smelte_rs::nn::models::bert::{
    Bert, BertAttention, BertEmbeddings, BertEncoder, BertLayer, BertPooler, Mlp,
};
use smelte_rs::SmeltError;
use std::borrow::Cow;

use smelte_rs::cpu::f32::{Device, Tensor};

pub trait FromSafetensors {
    fn from_tensors(tensors: &SafeTensors, _device: &Device) -> Self
    where
        Self: Sized;
}

fn to_tensor(view: TensorView) -> Result<Tensor, SmeltError> {
    let shape = view.shape().to_vec();
    let data = to_f32(view);
    Tensor::new(data, shape)
}

pub fn to_f32(view: TensorView) -> Cow<'static, [f32]> {
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

fn linear_from(weights: TensorView, bias: TensorView) -> Linear<Tensor> {
    Linear::new(to_tensor(weights).unwrap(), to_tensor(bias).unwrap())
}

fn linear_from_prefix(prefix: &str, tensors: &SafeTensors) -> Linear<Tensor> {
    linear_from(
        tensors.tensor(&format!("{}.weight", prefix)).unwrap(),
        tensors.tensor(&format!("{}.bias", prefix)).unwrap(),
    )
}

fn embedding_from(weights: TensorView) -> Embedding<Tensor> {
    Embedding::new(to_tensor(weights).unwrap())
}

impl FromSafetensors for BertClassifier<Tensor> {
    fn from_tensors(tensors: &SafeTensors, device: &Device) -> Self
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
        let classifier = linear_from(weight, bias);
        Self::new(bert, pooler, classifier)
    }
}

impl FromSafetensors for BertPooler<Tensor> {
    fn from_tensors(tensors: &SafeTensors, _device: &Device) -> Self
    where
        Self: Sized,
    {
        let pooler = linear_from(
            tensors.tensor("bert.pooler.dense.weight").unwrap(),
            tensors.tensor("bert.pooler.dense.bias").unwrap(),
        );
        Self::new(pooler)
    }
}

impl FromSafetensors for Bert<Tensor> {
    fn from_tensors(tensors: &SafeTensors, device: &Device) -> Self
    where
        Self: Sized,
    {
        let embeddings = BertEmbeddings::from_tensors(tensors, device);
        let encoder = BertEncoder::from_tensors(tensors, device);
        Bert::new(embeddings, encoder)
    }
}

impl FromSafetensors for BertEmbeddings<Tensor> {
    fn from_tensors(tensors: &SafeTensors, _device: &Device) -> Self
    where
        Self: Sized,
    {
        let input_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.word_embeddings.weight")
                .unwrap(),
        );
        let position_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.position_embeddings.weight")
                .unwrap(),
        );
        let type_embeddings = embedding_from(
            tensors
                .tensor("bert.embeddings.token_type_embeddings.weight")
                .unwrap(),
        );

        let layer_norm = layer_norm_from_prefix("bert.embeddings.LayerNorm", &tensors);
        BertEmbeddings::new(
            input_embeddings,
            position_embeddings,
            type_embeddings,
            layer_norm,
        )
    }
}

fn bert_layer_from_tensors(index: usize, tensors: &SafeTensors) -> BertLayer<Tensor> {
    let attention = bert_attention_from_tensors(index, tensors);
    let mlp = bert_mlp_from_tensors(index, tensors);
    BertLayer::new(attention, mlp)
}
fn bert_attention_from_tensors(index: usize, tensors: &SafeTensors) -> BertAttention<Tensor> {
    let query = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.query"),
        tensors,
    );
    let key = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.key"),
        tensors,
    );
    let value = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.self.value"),
        tensors,
    );
    let output = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.output.dense"),
        tensors,
    );
    let output_ln = layer_norm_from_prefix(
        &format!("bert.encoder.layer.{index}.attention.output.LayerNorm"),
        &tensors,
    );
    BertAttention::new(query, key, value, output, output_ln)
}

fn bert_mlp_from_tensors(index: usize, tensors: &SafeTensors) -> Mlp<Tensor> {
    let intermediate = linear_from_prefix(
        &format!("bert.encoder.layer.{index}.intermediate.dense"),
        tensors,
    );
    let output = linear_from_prefix(&format!("bert.encoder.layer.{index}.output.dense"), tensors);
    let output_ln = layer_norm_from_prefix(
        &format!("bert.encoder.layer.{index}.output.LayerNorm"),
        &tensors,
    );
    Mlp::new(intermediate, output, output_ln)
}

fn layer_norm_from_prefix(prefix: &str, tensors: &SafeTensors) -> LayerNorm<Tensor> {
    let epsilon = 1e-5;
    if let (Ok(weight), Ok(bias)) = (
        tensors.tensor(&format!("{}.weight", prefix)),
        tensors.tensor(&format!("{}.bias", prefix)),
    ) {
        LayerNorm::new(
            to_tensor(weight).unwrap(),
            to_tensor(bias).unwrap(),
            epsilon,
        )
    } else {
        LayerNorm::new(
            to_tensor(tensors.tensor(&format!("{}.gamma", prefix)).unwrap()).unwrap(),
            to_tensor(tensors.tensor(&format!("{}.beta", prefix)).unwrap()).unwrap(),
            epsilon,
        )
    }
}
impl FromSafetensors for BertEncoder<Tensor> {
    fn from_tensors(tensors: &SafeTensors, _device: &Device) -> Self
    where
        Self: Sized,
    {
        // TODO ! Count heads from tensors present
        let layers: Vec<_> = (0..12)
            .map(|i| bert_layer_from_tensors(i, tensors))
            .collect();
        Self::new(layers)
    }
}