// mini-rs/minisnap/src/error.rs
//
// Copyright (c) 2025 Arcella Team
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE>
// or the MIT license <LICENSE-MIT>, at your option.
// This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::PathBuf;

use serde_json;
use thiserror::Error;

/// Error type for `minisnap` operations.
#[derive(Error, Debug)]
pub enum MiniSnapError {
    #[error("I/O error on {path}: {source}")]
    Io {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("Failed to serde_json state: {source}")]
    Serde {
        #[from]
        source: serde_json::Error,
    },

    #[error("Snapshot sequence file contains invalid number")]
    InvalidSequence,

    #[error("Snapshot not found (missing snapshot.json or snapshot.seq)")]
    NotFound,
}

impl From<std::io::Error> for MiniSnapError {
    fn from(err: std::io::Error) -> Self {
        // Try to extract path from error if possible (simplified)
        // In practice, you may want a more robust approach
        Self::Io {
            source: err,
            path: PathBuf::from("<unknown>"),
        }
    }
}
