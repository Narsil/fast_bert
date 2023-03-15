use safetensors::{
    tensor::{Dtype, TensorView},
    SafeTensors,
};
use smelt::cpu::f32::Tensor;
use smelt::nn::layers::{Embedding, LayerNorm, Linear};
use smelt::nn::models::bert::{
    Bert as BertModel, BertAttention, BertEmbeddings, BertEncoder, BertLayer, BertPooler, Mlp,
};
use smelt::TensorError;
use std::borrow::Cow;

// Renamed
pub use smelt::nn::models::bert::BertClassifier as Bert;

pub trait FromSafetensors<'a> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self
    where
        Self: Sized;
}

fn to_tensor<'data>(view: TensorView<'data>) -> Result<Tensor<'data>, TensorError> {
    let shape = view.shape().to_vec();
    let data = to_f32(view);
    Tensor::new(data, shape)
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

fn linear_from<'a>(weights: TensorView<'a>, bias: TensorView<'a>) -> Linear<Tensor<'a>> {
    Linear::new(to_tensor(weights).unwrap(), to_tensor(bias).unwrap())
}

fn linear_from_prefix<'a>(prefix: &str, tensors: &'a SafeTensors<'a>) -> Linear<Tensor<'a>> {
    linear_from(
        tensors.tensor(&format!("{}.weight", prefix)).unwrap(),
        tensors.tensor(&format!("{}.bias", prefix)).unwrap(),
    )
}

fn embedding_from<'a>(weights: TensorView<'a>) -> Embedding<Tensor<'a>> {
    Embedding::new(to_tensor(weights).unwrap())
}

fn layer_norm_from_prefix<'a>(prefix: &str, tensors: &'a SafeTensors<'a>) -> LayerNorm<Tensor<'a>> {
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

impl<'a> FromSafetensors<'a> for Bert<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self
    where
        Self: Sized,
    {
        let pooler = BertPooler::from_tensors(tensors, num_heads);
        let bert = BertModel::from_tensors(tensors, num_heads);
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

impl<'a> FromSafetensors<'a> for BertPooler<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, _num_heads: usize) -> Self
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

impl<'a> FromSafetensors<'a> for BertModel<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self
    where
        Self: Sized,
    {
        let embeddings = BertEmbeddings::from_tensors(tensors, num_heads);
        let encoder = BertEncoder::from_tensors(tensors, num_heads);
        BertModel::new(embeddings, encoder)
    }
}

impl<'a> FromSafetensors<'a> for BertEmbeddings<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, _num_heads: usize) -> Self
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

impl<'a> FromSafetensors<'a> for BertEncoder<Tensor<'a>> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self
    where
        Self: Sized,
    {
        // TODO ! Count heads from tensors present
        let layers: Vec<_> = (0..12)
            .map(|i| bert_layer_from_tensors(i, tensors, num_heads))
            .collect();
        Self::new(layers)
    }
}

fn bert_layer_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
    num_heads: usize,
) -> BertLayer<Tensor<'a>> {
    let attention = bert_attention_from_tensors(index, tensors, num_heads);
    let mlp = bert_mlp_from_tensors(index, tensors);
    BertLayer::new(attention, mlp)
}
fn bert_attention_from_tensors<'a>(
    index: usize,
    tensors: &'a SafeTensors<'a>,
    num_heads: usize,
) -> BertAttention<Tensor<'a>> {
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
    BertAttention::new(query, key, value, output, output_ln, num_heads)
}

fn bert_mlp_from_tensors<'a>(index: usize, tensors: &'a SafeTensors<'a>) -> Mlp<Tensor<'a>> {
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
