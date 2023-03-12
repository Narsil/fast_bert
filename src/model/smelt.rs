use safetensors::tensor::{SafeTensors, TensorView};
use smelt::ops::{
    add, apply, faster_gelu, gelu, matmul, matmul_t, mul, normalize, par_apply, select, softmax,
};
use smelt::tensor::{OwnedTensor, Tensor, TensorMut, ViewTensor};
use tokenizers::Encoding;

fn split_heads<T: Tensor>(q: &T, num_heads: usize) -> OwnedTensor {
    let sequence_length = q.shape()[0];
    let hidden_dim = q.shape()[1];
    assert_eq!(hidden_dim % num_heads, 0);
    let head_dim = hidden_dim / num_heads;
    let mut query_data = vec![0.0; num_heads * sequence_length * head_dim];
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let index = j * hidden_dim + i * head_dim + k;
                let out_index = i * sequence_length * head_dim + j * head_dim + k;
                let value = q.data()[index];
                query_data[out_index] = value;
            });
        });
    });
    OwnedTensor::new(query_data, vec![num_heads, sequence_length, head_dim]).unwrap()
}

fn attention<T: Tensor, TM: TensorMut>(
    query: &T,
    key: &T,
    value: &T,
    qk: &mut TM,
    max: &mut [f32],
    out: &mut OwnedTensor,
) {
    let sequence_length = query.shape()[0];
    let hidden_dim = query.shape()[1];
    let num_heads = qk.shape()[0];
    assert_eq!(hidden_dim % num_heads, 0);

    assert_eq!(
        qk.shape(),
        vec![num_heads, sequence_length, sequence_length]
    );

    let query = split_heads(query, num_heads);
    let key = split_heads(key, num_heads);
    let value = split_heads(value, num_heads);

    matmul_t(&query, &key, qk).unwrap();
    let head_dim = hidden_dim / num_heads;
    let scale = (head_dim as f32).sqrt();
    qk.data_mut().iter_mut().for_each(|v| *v /= scale);

    softmax(qk, max).unwrap();
    matmul(qk, &value, out).unwrap();

    let mut new_out = vec![0.0; sequence_length * hidden_dim];
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let in_index = i * sequence_length * head_dim + j * head_dim + k;
                let out_index = j * hidden_dim + i * head_dim + k;
                new_out[out_index] = out.data()[in_index];
            });
        });
    });
    *out = OwnedTensor::new(new_out, vec![sequence_length, hidden_dim]).unwrap();
}

#[derive(Clone)]
pub struct Mlp<'a> {
    intermediate: Linear<'a>,
    output: Linear<'a>,
    output_ln: LayerNorm<'a>,
}

impl<'a> Mlp<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>) -> Self {
        let intermediate = Linear::from(
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.intermediate.dense.weight"
                ))
                .unwrap(),
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.intermediate.dense.bias"
                ))
                .unwrap(),
        );
        let output = Linear::from(
            tensors
                .tensor(&format!("bert.encoder.layer.{index}.output.dense.weight"))
                .unwrap(),
            tensors
                .tensor(&format!("bert.encoder.layer.{index}.output.dense.bias"))
                .unwrap(),
        );
        let output_ln = LayerNorm::from_prefix(
            &format!("bert.encoder.layer.{index}.output.LayerNorm"),
            &tensors,
        );
        Self {
            intermediate,
            output,
            output_ln,
        }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        let input_tensor = tensor.clone();
        // println!("Intermediate {:?}", tensor.shape());
        // println!("Intermediate {:?}", self.intermediate.weight.shape());
        // println!("Intermediate {:?}", self.intermediate.bias.shape());
        self.intermediate.forward(tensor);
        // println!("Intermediate after {:?}", tensor.shape());
        par_apply(tensor, faster_gelu);
        // let tmp = tensor.data();
        // println!("After gelu {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
        // println!("Output {:?}", tensor.shape());
        self.output.forward(tensor);
        // println!("Output after {:?}", tensor.shape());
        // TODO SKIP connection
        add(&input_tensor, tensor).unwrap();
        self.output_ln.forward(tensor);
        // let tmp = tensor.data();
        // println!("After MLP {:?} {:?}", &tmp[..5], &tmp[tmp.len() - 5..]);
    }
}

#[derive(Clone)]
pub struct BertAttention<'a> {
    query: Linear<'a>,
    key: Linear<'a>,
    value: Linear<'a>,
    output: Linear<'a>,
    output_ln: LayerNorm<'a>,
    num_heads: usize,
}

impl<'a> BertAttention<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self {
        let query = Linear::from(
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.attention.self.query.weight"
                ))
                .unwrap(),
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.attention.self.query.bias"
                ))
                .unwrap(),
        );
        let key = Linear::from(
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.attention.self.key.weight"
                ))
                .unwrap(),
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.attention.self.key.bias"
                ))
                .unwrap(),
        );
        let value = Linear::from(
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.attention.self.value.weight"
                ))
                .unwrap(),
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.attention.self.value.bias"
                ))
                .unwrap(),
        );
        let output = Linear::from(
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.attention.output.dense.weight"
                ))
                .unwrap(),
            tensors
                .tensor(&format!(
                    "bert.encoder.layer.{index}.attention.output.dense.bias"
                ))
                .unwrap(),
        );
        let output_ln = LayerNorm::from_prefix(
            &format!("bert.encoder.layer.{index}.attention.output.LayerNorm"),
            &tensors,
        );
        Self {
            query,
            key,
            value,
            output,
            output_ln,
            num_heads,
        }
    }

    pub fn forward(&self, hidden_states: &mut OwnedTensor) {
        // println!("---");
        //debug!("Attention", hidden_states);
        assert_eq!(hidden_states.shape().len(), 2);
        let input_tensor = hidden_states.clone();
        let sequence_length = hidden_states.shape()[0];
        let hidden_dim = hidden_states.shape()[1];

        let mut q = hidden_states.clone();
        let mut k = hidden_states.clone();
        let mut v = hidden_states.clone();
        // println!("Q {:?}", q.shape());
        self.query.forward(&mut q);
        // println!("Q after {:?}", q.shape());
        // println!("k {:?}", q.shape());
        self.key.forward(&mut k);
        // println!("v {:?}", q.shape());
        self.value.forward(&mut v);

        let num_heads = self.num_heads;
        assert_eq!(hidden_dim % num_heads, 0);
        let head_dim = hidden_dim / num_heads;
        let mut qk = OwnedTensor::zeros(vec![num_heads, sequence_length, sequence_length]);
        let mut qv = OwnedTensor::zeros(vec![num_heads, sequence_length, head_dim]);
        let mut max = vec![0.0; (sequence_length) * num_heads];
        attention(&q, &k, &v, &mut qk, &mut max, &mut qv);

        // debug!("After self attention", qv);
        // println!("qv {:?}", qv.shape());
        self.output.forward(&mut qv);
        add(&input_tensor, &mut qv).unwrap();
        // println!("ln {:?}", qv.shape());
        self.output_ln.forward(&mut qv);
        *hidden_states = qv;
    }
}

#[derive(Clone)]
pub struct BertLayer<'a> {
    mlp: Mlp<'a>,
    attention: BertAttention<'a>,
}

impl<'a> BertLayer<'a> {
    fn from_tensors(index: usize, tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self {
        let mlp = Mlp::from_tensors(index, tensors);
        let attention = BertAttention::from_tensors(index, tensors, num_heads);
        Self {
            // ln_1,
            // ln_2,
            mlp,
            attention,
        }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        // println!("==============");
        // debug!("Incoming", tensor);

        // let residual = tensor.clone();
        // self.ln_1.forward(tensor);
        self.attention.forward(tensor);
        // debug!("Attention", tensor);

        // add(&residual, tensor);
        // let residual = tensor.clone();
        // self.ln_2.forward(tensor);

        self.mlp.forward(tensor);
        // add(&residual, tensor);
        // debug!("After layer", tensor);
    }
}

#[derive(Clone)]
pub struct BertEncoder<'a> {
    layers: Vec<BertLayer<'a>>,
}

impl<'a> BertEncoder<'a> {
    fn from_tensors(tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self {
        let layers: Vec<_> = (0..12)
            .map(|i| BertLayer::from_tensors(i, tensors, num_heads))
            .collect();
        Self { layers }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        self.layers.iter().for_each(|layer| {
            layer.forward(tensor);
        });
    }
}

#[derive(Clone)]
pub struct Linear<'a> {
    weight: ViewTensor<'a>,
    bias: ViewTensor<'a>,
}

impl<'a> std::fmt::Debug for Linear<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("shape", &self.weight.shape())
            .finish()
    }
}

impl<'a> Linear<'a> {
    pub fn new(weight: ViewTensor<'a>, bias: ViewTensor<'a>) -> Self {
        Self { weight, bias }
    }

    fn from(weight: TensorView<'a>, bias: TensorView<'a>) -> Self {
        let weight: ViewTensor = weight.into();
        let bias: ViewTensor = bias.into();
        Self::new(weight, bias)
    }

    pub fn forward(&self, tensor: &mut OwnedTensor) {
        assert_eq!(tensor.shape().len(), 2);
        let m = tensor.shape()[0];
        let n = self.weight.shape()[0];
        let mut c = OwnedTensor::zeros(vec![m, n]);

        matmul_t(tensor, &self.weight, &mut c).unwrap();
        add(&self.bias, &mut c).unwrap();
        //addmm(tensor, &self.weight, &self.bias, &mut c);
        *tensor = c;
    }
}

#[derive(Clone)]
pub struct Embedding<'a> {
    weight: ViewTensor<'a>,
}

impl<'a> Embedding<'a> {
    fn from(weight: TensorView<'a>) -> Self {
        let weight: ViewTensor = weight.into();
        Self { weight }
    }

    fn forward(&self, ids: &[u32]) -> OwnedTensor {
        let _vocab_size = self.weight.shape()[0];
        let hidden_dim = self.weight.shape()[1];
        let shape = vec![ids.len(), hidden_dim];
        let mut tensor = OwnedTensor::zeros(shape);
        select(ids, &self.weight, &mut tensor).unwrap();
        tensor
    }
}

#[derive(Clone)]
pub struct LayerNorm<'a> {
    weight: ViewTensor<'a>,
    bias: ViewTensor<'a>,
    epsilon: f32,
}

impl<'a> LayerNorm<'a> {
    fn from_prefix(prefix: &str, tensors: &'a SafeTensors<'a>) -> Self {
        let layer_norm = if let (Ok(weight), Ok(bias)) = (
            tensors.tensor(&format!("{}.weight", prefix)),
            tensors.tensor(&format!("{}.bias", prefix)),
        ) {
            LayerNorm::from(weight, bias)
        } else {
            LayerNorm::from(
                tensors.tensor(&format!("{}.gamma", prefix)).unwrap(),
                tensors.tensor(&format!("{}.beta", prefix)).unwrap(),
            )
        };
        layer_norm
    }

    fn from(weight: TensorView<'a>, bias: TensorView<'a>) -> Self {
        let weight: ViewTensor = weight.into();
        let bias: ViewTensor = bias.into();
        let epsilon = 1e-5;
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        normalize(tensor, self.epsilon).unwrap();
        mul(&self.weight, tensor).unwrap();
        add(&self.bias, tensor).unwrap();
    }
}

#[derive(Clone)]
pub struct BertPooler<'a> {
    pooler: Linear<'a>,
}

impl<'a> BertPooler<'a> {
    fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self {
        let pooler = Linear::from(
            tensors.tensor("bert.pooler.dense.weight").unwrap(),
            tensors.tensor("bert.pooler.dense.bias").unwrap(),
        );
        Self { pooler }
    }

    fn forward(&self, tensor: &mut OwnedTensor) {
        // debug!("Before pooler", tensor);
        let mut first = OwnedTensor::zeros(vec![1, tensor.shape()[1]]);
        select(&[0], tensor, &mut first).unwrap();
        // debug!("select", first);
        self.pooler.forward(&mut first);
        // debug!("pooler", first);
        first.data_mut().iter_mut().for_each(|v| *v = f32::tanh(*v));
        // debug!("tanh", first);
        *tensor = first;
        // debug!("After", tensor);
    }
}

#[derive(Clone)]
pub struct BertEmbeddings<'a> {
    wte: Embedding<'a>,
    wpe: Embedding<'a>,
    type_embeddings: Embedding<'a>,
    layer_norm: LayerNorm<'a>,
}

impl<'a> BertEmbeddings<'a> {
    pub fn from_tensors(tensors: &'a SafeTensors<'a>) -> Self {
        let wte = Embedding::from(
            tensors
                .tensor("bert.embeddings.word_embeddings.weight")
                .unwrap(),
        );
        let wpe = Embedding::from(
            tensors
                .tensor("bert.embeddings.position_embeddings.weight")
                .unwrap(),
        );
        let type_embeddings = Embedding::from(
            tensors
                .tensor("bert.embeddings.token_type_embeddings.weight")
                .unwrap(),
        );

        let layer_norm = LayerNorm::from_prefix("bert.embeddings.LayerNorm", &tensors);
        Self {
            wte,
            wpe,
            type_embeddings,
            layer_norm,
        }
    }
}
impl<'a> BertEmbeddings<'a> {
    pub fn forward(&self, encoded: &Encoding) -> OwnedTensor {
        let ids = encoded.get_ids();
        let mut tensor = self.wte.forward(ids);
        let type_embeds = self.type_embeddings.forward(encoded.get_type_ids());
        let positions: Vec<u32> = (0..ids.len()).map(|i| i as u32).collect();
        let position_embeddings = self.wpe.forward(&positions[..]);

        add(&type_embeds, &mut tensor).unwrap();
        add(&position_embeddings, &mut tensor).unwrap();
        self.layer_norm.forward(&mut tensor);
        // debug!("After bert embeddings", tensor);
        tensor
    }
}

#[derive(Clone)]
pub struct Bert<'a> {
    embeddings: BertEmbeddings<'a>,
    encoder: BertEncoder<'a>,
    pooler: BertPooler<'a>,
    classifier: Linear<'a>,
    num_classes: usize,
}

impl<'a> Bert<'a> {
    pub fn from_tensors(tensors: &'a SafeTensors<'a>, num_heads: usize) -> Self {
        let embeddings = BertEmbeddings::from_tensors(tensors);
        let encoder = BertEncoder::from_tensors(tensors, num_heads);
        let pooler = BertPooler::from_tensors(tensors);

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
        let classifier = Linear::from(weight, bias);
        let num_classes = classifier.weight.shape()[0];
        Self {
            encoder,
            pooler,
            embeddings,
            classifier,
            num_classes,
        }
    }
}

impl<'a> Bert<'a> {
    pub fn forward(&self, encoded: &Encoding) -> OwnedTensor {
        let mut tensor = self.embeddings.forward(encoded);
        // println!("tensor {:?}", tensor.shape());
        self.encoder.forward(&mut tensor);
        // debug!("After encoder ", tensor);
        // println!("tensor {:?}", tensor.shape());
        self.pooler.forward(&mut tensor);
        // println!("tensor {:?}", tensor.shape());
        self.classifier.forward(&mut tensor);
        // debug!("outputs ", &tensor);
        let mut logits = tensor;
        let mut max = vec![0.0; self.num_classes];
        // println!("logits {:?}", logits.shape());
        softmax(&mut logits, &mut max).unwrap();
        // debug!("logits ", logits);
        logits
    }
}
