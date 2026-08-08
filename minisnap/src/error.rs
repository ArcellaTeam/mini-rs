// mini-rs/minisnap/src/error.rs
//
// Copyright (c) 2025 Arcella Team
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE>
// or the MIT license <LICENSE-MIT>, at your option.
// This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// A specialized [`Result`](std::result::Result) type for `minisnap` operations.
pub type Result<T> = std::result::Result<T, MiniSnapError>;

/// Errors returned by `minisnap` operations.
///
/// This type wraps I/O, codec, and snapshot discovery failures.
#[derive(Error, Debug)]
pub enum MiniSnapError {
    #[error("I/O error on {path}: {source}")]
    Io { source: std::io::Error, path: PathBuf },

    #[error("Codec error: {source}")]
    Codec { source: Box<dyn std::error::Error + Send + Sync + 'static> },

    #[error("Snapshot not found")]
    NotFound,

    #[error("Tokio error")]
    TokioError {
        #[from]
        source: tokio::task::JoinError,
    },
}

impl MiniSnapError {
    /// Convert an I/O error into a `MiniSnapError` with path context.
    pub fn io(err: std::io::Error, path: impl AsRef<Path>) -> Self {
        Self::Io {
            source: err,
            path: path.as_ref().into(),
        }
    }
}

impl From<std::io::Error> for MiniSnapError {
    fn from(err: std::io::Error) -> Self {
        // Try to extract path from error if possible (simplified)
        // In practice, you may want a more robust approach
        Self::io(err, PathBuf::from("<unknown>"))
    }
}

impl From<tempfile::PersistError> for MiniSnapError {
    fn from(err: tempfile::PersistError) -> Self {
        let path = err.file.path().to_path_buf();
        Self::io(err.into(), path)
    }
}
