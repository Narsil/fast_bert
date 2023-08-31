FROM rust:1.72 as build

SHELL ["/bin/bash", "-c"]

WORKDIR /usr/src/fast_bert
COPY docker/dummy.rs .
COPY Cargo.toml .
COPY Cargo.lock .
RUN sed -i 's#src/main.rs#dummy.rs#' Cargo.toml
RUN RUSTFLAGS="-C target-feature=+sse3,+avx2,+fma,+f16c" cargo build --profile=release-fat
RUN sed -i 's#dummy.rs#src/main.rs#' Cargo.toml
COPY src/ src/

RUN RUSTFLAGS="-C target-feature=+sse3,+avx2,+fma,+f16c" cargo build --profile=release-fat

FROM gcr.io/distroless/cc-debian12

COPY --from=build /usr/src/fast_bert/target/release-fat/fast_bert /usr/local/bin/fast_bert
ENV PORT=80
# COPY tokenizer.json .
# COPY model.safetensors .
CMD ["fast_bert"]
