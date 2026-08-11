// mini-rs/ministore/src/error.rs
//
// Copyright (c) 2025 Arcella Team
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE>
// or the MIT license <LICENSE-MIT>, at your option.
// This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum MiniStoreError {
    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("Serialization error: {source}")]
    Serialize {
        #[from]
        source: serde_json::Error,
    },

    #[error("Deserialization error at line {line}: {source}")]
    Deserialize {
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("Invalid journal file: missing initial state marker")]
    MissingInitialState,

    #[error("Path must be a file, got: {path:?}")]
    PathIsNotFile { path: PathBuf },

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}
