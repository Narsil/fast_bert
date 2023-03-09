use crate::BertError;
use reqwest::header::{CONTENT_RANGE, RANGE};
use std::io::SeekFrom;
use std::sync::Arc;
use tokio::io::AsyncSeekExt;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

pub async fn download(
    url: &str,
    filename: &str,
    max_files: usize,
    chunk_size: usize,
) -> Result<(), BertError> {
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .header(RANGE, "bytes=0-0".to_string())
        .send()
        .await?;
    let content_range = response
        .headers()
        .get(CONTENT_RANGE)
        .ok_or(BertError::NoContentLength)?
        .to_str()?;

    let size: Vec<&str> = content_range.split('/').collect();
    // Content-Range: bytes 0-0/702517648
    // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Range
    let length: usize = size.last().ok_or(BertError::NoContentLength)?.parse()?;
    let file = tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(filename)
        .await?;
    file.set_len(length as u64).await?;

    let mut handles = vec![];
    let semaphore = Arc::new(Semaphore::new(max_files));

    let chunk_size = chunk_size;
    for start in (0..length).step_by(chunk_size) {
        let url = url.to_string();
        let filename = filename.to_string();
        let client = client.clone();

        let stop = std::cmp::min(start + chunk_size - 1, length);
        let permit = semaphore.clone().acquire_owned().await?;
        handles.push(tokio::spawn(async move {
            let chunk = download_chunk(client, url, filename, start, stop).await;
            drop(permit);
            chunk
        }));
    }

    // Output the chained result
    let results: Vec<Result<Result<(), BertError>, tokio::task::JoinError>> =
        futures::future::join_all(handles).await;
    let results: Result<(), BertError> = results.into_iter().flatten().collect();
    results?;
    Ok(())
}

async fn download_chunk(
    client: reqwest::Client,
    url: String,
    filename: String,
    start: usize,
    stop: usize,
) -> Result<(), BertError> {
    // Process each socket concurrently.
    let range = format!("bytes={start}-{stop}");
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(filename)
        .await?;
    file.seek(SeekFrom::Start(start as u64)).await?;
    let response = client.get(url).header(RANGE, range).send().await?;
    let content = response.bytes().await?;
    file.write_all(&content).await?;
    Ok(())
}
