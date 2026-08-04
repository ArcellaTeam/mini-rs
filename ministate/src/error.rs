// mini-rs/ministate/src/error.rs
//
// Copyright (c) 2025 Arcella Team
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE>
// or the MIT license <LICENSE-MIT>, at your option.
// This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(feature = "snapshot")]
use minisnap::MiniSnapError;
use ministore::MiniStoreError;

#[derive(Debug, thiserror::Error)]
pub enum MiniStateError {
    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("Store error: {source}")]
    Store {
        #[from]
        source: MiniStoreError,
    },

    #[cfg(feature = "snapshot")]
    #[error("Snapshot error: {source}")]
    Snapshot {
        #[from]
        source: MiniSnapError,
    },

    #[cfg(feature = "snapshot")]
    #[error("Snapshot sequence number too high")]
    SnapshotSeqTooHigh,
}
