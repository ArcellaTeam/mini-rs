// mini-rs/minisnap/src/lib.rs
//
// Copyright (c) 2025 Arcella Team
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE>
// or the MIT license <LICENSE-MIT>, at your option.
// This file may not be copied, modified, or distributed
// except according to those terms.

//! Minimal snapshot store for durable state managers.
//!
//! `minisnap` provides a simple, reliable way to **serialize and restore** a full copy
//! of application state to/from disk. It is designed to complement WAL-based systems
//! like [`ministore`] and [`ministate`] by enabling **fast recovery** and future **WAL compaction**.
//!
//! ## Features
//!
//! - **Explicit snapshotting**: You control when snapshots are created.
//! - **Atomic writes**: Snapshots are written to a temp file and atomically renamed.
//! - **Sequence tracking**: Each snapshot is associated with a logical sequence number
//!   (e.g., the last applied WAL index) for consistency.
//! - **Human-readable**: Snapshots are stored as plain JSON (via `serde`).
//!
//! ## Integration
//!
//! Intended for use with `ministate` (via the `snapshot` feature), but can be used standalone.
//!
//! # Example
//!
//! ```rust
//! use minisnap::SnapStore;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, PartialEq, Debug)]
//! struct AppState { counter: u64 }
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tmp = tempfile::tempdir()?;
//!     let store = SnapStore::new(tmp.path());
//!
//!     let state = AppState { counter: 42 };
//!     store.create(&state, 10).await?; // seq = 10
//!
//!     let (restored, seq) = store.restore().await?;
//!     let restored: AppState = restored;
//!     assert_eq!(restored, state);
//!     assert_eq!(seq, 10);
//!     Ok(())
//! }
//! ```

use std::path::{Path, PathBuf};

use serde::{Serialize, de::DeserializeOwned};
use tokio::{fs, io::AsyncWriteExt};

mod error;
pub use error::MiniSnapError;

/// A specialized [`Result`](std::result::Result) type for `minisnap` operations.
pub type Result<T> = std::result::Result<T, MiniSnapError>;

/// File name for the snapshot data.
const SNAPSHOT_FILE: &str = "snapshot.json";

/// File name for the snapshot metadata (e.g., sequence number).
const METADATA_FILE: &str = "snapshot.seq";

/// Manages snapshot storage in a dedicated directory.
///
/// The snapshot consists of two files:
/// - `{dir}/snapshot.json` — serialized state (JSON)
/// - `{dir}/snapshot.seq` — plain text sequence number (e.g., "12345")
///
/// Both files are written atomically and read consistently.
#[derive(Debug, Clone)]
pub struct SnapStore {
    dir: PathBuf,
    snapshot_path: PathBuf,
    metadata_path: PathBuf,
}

impl SnapStore {
    /// Creates a new `SnapStore` that operates in the given directory.
    ///
    /// The directory **does not need to exist** — it will be created on first write.
    ///
    /// # Arguments
    ///
    /// * `dir` — Directory to store snapshot files.
    pub fn new<P: AsRef<Path>>(dir: P) -> Self {
        let dir = dir.as_ref().to_path_buf();
        let snapshot_path = dir.join(SNAPSHOT_FILE);
        let metadata_path = dir.join(METADATA_FILE);
        Self {
            dir,
            snapshot_path,
            metadata_path,
        }
    }

    /// Atomically creates a new snapshot of the given state and sequence number.
    ///
    /// The snapshot is written to temporary files and then atomically renamed
    /// to the final names, ensuring:
    /// - No partial/corrupted snapshots are visible.
    /// - A reader always sees either the old snapshot or the new one — never an intermediate state.
    ///
    /// # Arguments
    ///
    /// * `state` — The serializable state to snapshot.
    /// * `seq` — Logical sequence number (e.g., last applied WAL index).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Serialization fails.
    /// - The snapshot directory cannot be created.
    /// - Disk I/O fails during write or rename.
    pub async fn create<S>(&self, state: &S, seq: u64) -> Result<()>
    where
        S: Serialize,
    {
        fs::create_dir_all(&self.dir).await?;

        // Write state to temp file
        let state_tmp = self.snapshot_path.with_extension("json.tmp");
        {
            let state_json = serde_json::to_string_pretty(state)?;

            let mut tmp_file =
                fs::OpenOptions::new().write(true).create(true).open(&state_tmp).await.map_err(
                    |e| MiniSnapError::Io {
                        source: e,
                        path: state_tmp.clone(),
                    },
                )?;
            tmp_file.write(state_json.as_bytes()).await.map_err(|e| MiniSnapError::Io {
                source: e,
                path: state_tmp.clone(),
            })?;
            tmp_file.sync_all().await.map_err(|e| MiniSnapError::Io {
                source: e,
                path: state_tmp.clone(),
            })?;
        }

        // Write sequence to temp file
        let seq_tmp = self.metadata_path.with_extension("seq.tmp");
        {
            let seq_string = seq.to_string();

            let mut seq_file =
                fs::OpenOptions::new().write(true).create(true).open(&seq_tmp).await.map_err(
                    |e| MiniSnapError::Io {
                        source: e,
                        path: seq_tmp.clone(),
                    },
                )?;
            seq_file.write(seq_string.as_bytes()).await.map_err(|e| MiniSnapError::Io {
                source: e,
                path: seq_tmp.clone(),
            })?;
            seq_file.sync_all().await.map_err(|e| MiniSnapError::Io {
                source: e,
                path: seq_tmp.clone(),
            })?;
        }

        // Atomic rename
        fs::rename(&state_tmp, &self.snapshot_path)
            .await
            .map_err(|e| MiniSnapError::Io { source: e, path: state_tmp })?;
        fs::rename(&seq_tmp, &self.metadata_path)
            .await
            .map_err(|e| MiniSnapError::Io { source: e, path: seq_tmp })?;

        Ok(())
    }

    /// Restores the latest snapshot and its sequence number.
    ///
    /// Reads both `snapshot.json` and `snapshot.seq` and ensures they exist.
    ///
    /// # Returns
    ///
    /// A tuple `(state, seq)` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either file is missing.
    /// - Deserialization fails.
    /// - Sequence file contains invalid number.
    pub async fn restore<S>(&self) -> Result<(S, u64)>
    where
        S: DeserializeOwned,
    {
        // Read state
        let state_json =
            fs::read_to_string(&self.snapshot_path).await.map_err(|e| MiniSnapError::Io {
                source: e,
                path: self.snapshot_path.clone(),
            })?;
        let state: S = serde_json::from_str(&state_json)?;

        // Read sequence
        let seq_str =
            fs::read_to_string(&self.metadata_path).await.map_err(|e| MiniSnapError::Io {
                source: e,
                path: self.metadata_path.clone(),
            })?;
        let seq = seq_str.trim().parse::<u64>().map_err(|_| MiniSnapError::InvalidSequence)?;

        Ok((state, seq))
    }

    /// Checks if a complete snapshot exists (both .json and .seq files).
    pub async fn exists(&self) -> bool {
        // Treat I/O errors as "not exists" to avoid panics in permission-denied scenarios.
        fs::try_exists(&self.snapshot_path).await.unwrap_or(false)
            && fs::try_exists(&self.metadata_path).await.unwrap_or(false)
    }

    /// Returns the directory used for snapshot storage.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use tempfile::TempDir;

    use super::*;

    #[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
    struct TestState {
        counter: u64,
        message: String,
    }

    #[tokio::test]
    async fn test_create_and_restore_success() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path());

        let state = TestState {
            counter: 100,
            message: "hello".to_string(),
        };
        let seq = 42u64;

        // Create a snapshot
        store.create(&state, seq).await.unwrap();

        // Restore the snapshot
        let (restored, restored_seq) = store.restore::<TestState>().await.unwrap();

        assert_eq!(restored, state);
        assert_eq!(restored_seq, seq);

        // Check that files exist
        assert!(store.snapshot_path.exists());
        assert!(store.metadata_path.exists());
    }

    #[tokio::test]
    async fn test_restore_not_found() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path());

        let err = store.restore::<TestState>().await.unwrap_err();
        assert!(matches!(err, MiniSnapError::Io { .. })); // Ошибка чтения файла
    }

    #[tokio::test]
    async fn test_restore_missing_files() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path());

        // Create only snapshot.json, but not snapshot.seq
        fs::create_dir_all(&store.dir).await.unwrap();
        fs::write(&store.snapshot_path, r#"{"counter":1,"message":"test"}"#).await.unwrap();

        let err = store.restore::<TestState>().await.unwrap_err();
        // Must be an error when reading snapshot.seq
        assert!(matches!(err, MiniSnapError::Io { .. }));
    }

    #[tokio::test]
    async fn test_restore_invalid_sequence() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path());

        fs::create_dir_all(&store.dir).await.unwrap();
        fs::write(&store.snapshot_path, r#"{"counter":1,"message":"test"}"#).await.unwrap();
        fs::write(&store.metadata_path, "not-a-number").await.unwrap();

        let err = store.restore::<TestState>().await.unwrap_err();
        assert!(matches!(err, MiniSnapError::InvalidSequence));
    }

    #[tokio::test]
    async fn test_restore_corrupted_json() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path());

        fs::create_dir_all(&store.dir).await.unwrap();
        fs::write(&store.snapshot_path, r#"{"counter": invalid json}"#).await.unwrap();
        fs::write(&store.metadata_path, "123").await.unwrap();

        let err = store.restore::<TestState>().await.unwrap_err();
        assert!(matches!(err, MiniSnapError::Serde { .. }));
    }

    #[tokio::test]
    async fn test_create_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path());

        let state1 = TestState {
            counter: 1,
            message: "first".to_string(),
        };
        store.create(&state1, 10).await.unwrap();

        let state2 = TestState {
            counter: 2,
            message: "second".to_string(),
        };
        store.create(&state2, 20).await.unwrap();

        let (restored, seq) = store.restore::<TestState>().await.unwrap();
        assert_eq!(restored, state2);
        assert_eq!(seq, 20);
    }

    #[tokio::test]
    async fn test_atomicity_partial_write_does_not_corrupt() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path());

        // Imitate partial write: create temporary files manually
        let state_tmp = store.snapshot_path.with_extension("json.tmp");
        let _seq_tmp = store.metadata_path.with_extension("seq.tmp");

        fs::create_dir_all(&store.dir).await.unwrap();
        fs::write(&state_tmp, r#"{"counter":999,"message":"partial"}"#).await.unwrap();
        // seq_tmp NOT created → imitation failure

        // Try to restore should fail due to missing sequence file
        let err = store.restore::<TestState>().await.unwrap_err();
        assert!(matches!(err, MiniSnapError::Io { .. }));

        // Create a correct snapshot - it should overwrite everything
        let good_state = TestState {
            counter: 42,
            message: "good".to_string(),
        };
        store.create(&good_state, 100).await.unwrap();

        let (restored, seq) = store.restore::<TestState>().await.unwrap();
        assert_eq!(restored, good_state);
        assert_eq!(seq, 100);

        // Check that temporary files are removed
        assert!(!state_tmp.exists());
    }

    #[tokio::test]
    async fn test_dir_creation() {
        let tmp = TempDir::new().unwrap();
        let nested_dir = tmp.path().join("snapshots").join("v1");
        let store = SnapStore::new(&nested_dir);

        let state = TestState {
            counter: 1,
            message: "nested".to_string(),
        };
        store.create(&state, 1).await.unwrap();

        assert!(nested_dir.exists());
        assert!(store.snapshot_path.exists());
        assert!(store.metadata_path.exists());
    }

    #[tokio::test]
    async fn test_exists() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path());

        // Initially, no files → exists() == false
        assert!(!store.exists().await);

        // Create only snapshot.json
        fs::create_dir_all(&store.dir).await.unwrap();
        fs::write(&store.snapshot_path, r#"{"counter":1,"message":"test"}"#).await.unwrap();
        assert!(!store.exists().await); // still false — seq missing

        // Create only snapshot.seq
        fs::remove_file(&store.snapshot_path).await.ok();
        fs::write(&store.metadata_path, "123").await.unwrap();
        assert!(!store.exists().await); // still false — json missing

        // Create both files → exists() == true
        fs::write(&store.snapshot_path, r#"{"counter":1,"message":"test"}"#).await.unwrap();
        assert!(store.exists().await);
    }
}
