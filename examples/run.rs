use fast_bert::{run, BertError};
#[tokio::main]
async fn main() -> Result<(), BertError> {
    run().await
}
