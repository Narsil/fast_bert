FROM rust:1.71 as build

SHELL ["/bin/bash", "-c"]

WORKDIR /usr/src/fast_bert
COPY docker/dummy.rs .
COPY Cargo.toml .
COPY Cargo.lock .
RUN sed -i 's#src/main.rs#dummy.rs#' Cargo.toml
RUN cargo build --release
RUN sed -i 's#dummy.rs#src/main.rs#' Cargo.toml
COPY src/ src/

RUN cargo build --release

FROM gcr.io/distroless/cc-debian11

COPY --from=build /usr/src/fast_bert/target/release/fast_bert /usr/local/bin/fast_bert
ENV PORT=80
# COPY tokenizer.json .
# COPY model.safetensors .
CMD ["fast_bert"]


