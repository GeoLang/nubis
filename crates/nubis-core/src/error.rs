use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("empty point cloud")]
    EmptyCloud,

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
