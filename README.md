# fast_bert

Experiment to run from load to finish ML almost 5x faster, works mostly by optimizing load.

[![Fast bert on a real cluster is 3x faster to run](https://img.youtube.com/vi/yqHLIIgOze8/0.jpg)](https://www.youtube.com/watch?v=yqHLIIgOze8)

This is an experimental test to remove the need for PyTorch and have a highly specific
runtime that enables to load much faster than using regular PyTorch + transformers using
`safetensors` and direct memory mapping.

## Overview

- Written in Rust
- Almost no dependency (intel-mkl/blas/cuda)
- Has a webserver (used to demonstrate differences on real clusters)
- Implements Bert text-classification.
- Docker build (optimized for intel-mkl,or cuda).
- Docker image:
    - CPU: **42Mb** (excluding model + tokenizer which get downloaded at runtime, since it's faster than pulling from registry).
    - GPU: **200Mb** (excluding model + tokenizer which get downloaded at runtime, since it's faster than pulling from registry).

## Use
```
cargo run --release --features cpu  # requires intel-mkl, or `--features gpu`
```

Caveat: The first run will actually download the models so will definitely be much slower than this.
Speed to load and run 20 forward passes of bert.
### CPU (22ms load)
```
2023-03-27T12:13:44.836290Z DEBUG fast_bert: listening on 0.0.0.0:8000
2023-03-27T12:13:44.836403Z DEBUG fast_bert: Starting server loop
2023-03-27T12:13:44.854727Z DEBUG fast_bert: Loaded server loop
```

### GPU (204ms load)
```
2023-03-27T12:10:47.488241Z DEBUG fast_bert: listening on 0.0.0.0:8000
2023-03-27T12:10:47.488290Z DEBUG fast_bert: Starting server loop
2023-03-27T12:10:47.692254Z DEBUG fast_bert: Loaded server loop
```

## Performance

Runtime performance is always faster than naive torch, even faster than torch 2.0 compiled
mode (albeit not by much).
However, this code has NOT been optimized (yet). For instance the attention is still using
a naive attention, and not flash attention. Also all the code in CPU is single threaded (except matmul whihc is linked against
intel-mkl).
