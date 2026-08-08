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
//! - **File-based recovery**: Each snapshot is saved as a separate numbered file so
//!   the latest completed snapshot can be restored without requiring a separate
//!   sequence metadata file.
//!
//! ## Integration
//!
//! Intended for use with `ministate` (via the `snapshot` feature), but can be used standalone.
//!
//! # Example
//!
//! ```rust
//! use minisnap::{
//!     SnapStore,
//!     codec::json::JsonCodec,
//! };
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
//! struct AppState { counter: u64 }
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tmp = tempfile::tempdir()?;
//!     let store = SnapStore::new(tmp.path(), JsonCodec);
//!
//!     let state = AppState { counter: 42 };
//!     store.create(state.clone(), 10).await?; // seq = 10
//!
//!     let (restored, seq) = store.restore::<AppState>().await?;
//!     assert_eq!(restored, state);
//!     assert_eq!(seq, 10);
//!     Ok(())
//! }
//! ```

use std::{
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// Codec trait and codec implementations for snapshot serialization.
pub mod codec;
mod error;
#[cfg(feature = "json")]
pub use crate::codec::json::JsonCodec;
#[cfg(feature = "rmp")]
pub use crate::codec::rmp::RmpCodec;
pub use crate::{
    codec::Codec,
    error::{MiniSnapError, Result},
};


/// Manages snapshot storage in a dedicated directory.
///
/// Snapshots are written as numbered files. The latest available snapshot is
/// selected by sequence number during restore.
#[derive(Debug)]
pub struct SnapStore<C> {
    inner: Arc<SnapStoreInner<C>>,
}

impl<C> Clone for SnapStore<C> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<C> SnapStore<C>
where
    C: Codec + Send + Sync + 'static,
{
    /// Creates a new `SnapStore` that operates in the given directory.
    ///
    /// The directory **does not need to exist** — it will be created on first write.
    ///
    /// # Arguments
    ///
    /// * `dir` — Directory to store snapshot files.
    /// * `codec` — Codec to serialize/deserialize snapshots.
    pub fn new(dir: impl AsRef<Path>, codec: C) -> Self {
        Self {
            inner: Arc::new(SnapStoreInner::new(dir, codec)),
        }
    }

    /// Creates a new `SnapStore` using the codec's default value.
    ///
    /// This is convenient when the codec has a sensible `Default` implementation,
    /// such as `JsonCodec`.
    pub fn new_default(dir: impl AsRef<Path>) -> Self
    where
        C: Default,
    {
        Self::new(dir, C::default())
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
    pub async fn create<S>(&self, state: S, seq: u64) -> Result<()>
    where
        S: Serialize + Send + 'static,
    {
        self.inner.create(state, seq).await
    }

    /// Restores the latest snapshot and its sequence number.
    ///
    /// Reads the latest snapshot file in the configured directory.
    ///
    /// # Returns
    ///
    /// A tuple `(state, seq)` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No snapshot file exists.
    /// - Deserialization fails.
    pub async fn restore<S>(&self) -> Result<(S, u64)>
    where
        S: DeserializeOwned + Send + 'static,
    {
        self.inner.restore().await
    }

    /// Returns the latest snapshot file path from a directory.
    ///
    /// This helper is useful for testing and direct inspection of snapshot files.
    pub fn latest_snapshot_sync(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        self.inner.latest_snapshot_sync(dir)
    }

    /// Returns the latest snapshot file path from the store directory.
    pub async fn latest_snapshot(&self) -> Result<PathBuf> {
        self.inner.latest_snapshot().await
    }

    /// Returns the directory used for snapshot storage.
    pub fn dir(&self) -> &Path {
        &self.inner.dir
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope<S> {
    seq: u64,
    state: S,
}

#[derive(Debug, Clone)]
struct SnapStoreInner<C> {
    dir: PathBuf,
    codec: C,
}

impl<C> SnapStoreInner<C>
where
    C: Codec + Send + Sync + 'static,
{
    const SNAPSHOT_PREFIX: &str = "snapshot_";

    fn new(dir: impl AsRef<Path>, codec: C) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            codec,
        }
    }

    async fn create<S>(self: &Arc<Self>, state: S, seq: u64) -> Result<()>
    where
        S: Serialize + Send + 'static,
    {
        let this = Arc::clone(self);

        tokio::task::spawn_blocking(move || {
            std::fs::create_dir_all(&this.dir)?;
            let envelope = Envelope { seq, state };
            let mut tmp = tempfile::NamedTempFile::new_in(&this.dir)?; // unique name, same fs
            {
                let mut writer = BufWriter::new(tmp.as_file_mut());
                this.codec.encode(&mut writer, &envelope)?;
                // flush on drop
            }
            let snapshot_path = this.snapshot_path(seq);
            tmp.as_file().sync_all()?; // data durable before it becomes visible
            tmp.persist(snapshot_path)?;

            #[cfg(unix)]
            std::fs::File::open(&this.dir)?.sync_all()?;

            Ok(())
        })
        .await?
    }

    async fn restore<S>(self: &Arc<Self>) -> Result<(S, u64)>
    where
        S: DeserializeOwned + Send + 'static,
    {
        let this = Arc::clone(self);
        let envelope: Envelope<S> = tokio::task::spawn_blocking(move || -> Result<_> {
            let path = this.latest_snapshot_sync(&this.dir)?;
            let file = std::fs::File::open(&path).map_err(|e| MiniSnapError::io(e, path))?;
            let res = this.codec.decode(BufReader::new(file))?;
            Ok(res)
        })
        .await??;

        Ok((envelope.state, envelope.seq))
    }

    fn snapshot_path(&self, seq: u64) -> PathBuf {
        let mut path = self.dir.join(format!("{}{seq:020}", Self::SNAPSHOT_PREFIX));
        if !self.codec.ext().is_empty() {
            path.set_extension(self.codec.ext());
        }
        path
    }

    fn latest_snapshot_sync(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let dir = dir.as_ref();

        std::fs::read_dir(dir)
            .map_err(|err| MiniSnapError::io(err, dir))?
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let seq = path.file_stem()?.to_str()?.strip_prefix(Self::SNAPSHOT_PREFIX)?;
                let seq = seq.parse::<u64>().ok()?;
                Some((seq, path))
            })
            .max_by_key(|(seq, _)| *seq)
            .map(|(_, path)| path)
            .ok_or_else(|| MiniSnapError::NotFound)
    }

    async fn latest_snapshot(self: &Arc<Self>) -> Result<PathBuf> {
        let this = Arc::clone(self);
        tokio::task::spawn_blocking(move || this.latest_snapshot_sync(&this.dir)).await?
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use tempfile::TempDir;

    use super::*;
    use crate::codec::json::JsonCodec;

    #[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
    struct TestState {
        counter: u64,
        message: String,
    }

    #[tokio::test]
    async fn test_create_and_restore_success() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path(), JsonCodec);

        let state = TestState {
            counter: 100,
            message: "hello".to_string(),
        };

        store.create(state.clone(), 42).await.unwrap();

        let (restored, seq) = store.restore::<TestState>().await.unwrap();
        assert_eq!(restored, state);
        assert_eq!(seq, 42);
    }

    #[tokio::test]
    async fn test_restore_latest_snapshot_use_highest_seq() {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path(), JsonCodec);

        let first = TestState {
            counter: 1,
            message: "first".to_string(),
        };
        store.create(first, 10).await.unwrap();

        let second = TestState {
            counter: 2,
            message: "second".to_string(),
        };
        store.create(second.clone(), 20).await.unwrap();

        let (restored, seq) = store.restore::<TestState>().await.unwrap();
        assert_eq!(restored, second);
        assert_eq!(seq, 20);
    }

    async fn create_many(
        store: &SnapStore<JsonCodec>,
        states: impl IntoIterator<Item = (TestState, u64)>,
    ) -> Result<()> {
        let tasks: Vec<_> = states
            .into_iter()
            .map(|(state, seq)| {
                tokio::spawn({
                    let store = store.clone();
                    async move { store.create(state, seq).await }
                })
            })
            .collect();

        futures::future::try_join_all(tasks).await?;
        Ok(())
    }

    async fn restore_after_concurrent_creates(n: u64) {
        let tmp = TempDir::new().unwrap();
        let store = SnapStore::new(tmp.path(), JsonCodec);

        let states = (0..=n).map(|seq| {
            (
                TestState {
                    counter: seq,
                    message: format!("concurrent-{seq}"),
                },
                seq,
            )
        });

        create_many(&store, states).await.unwrap();

        let latest = store.latest_snapshot().await.unwrap();
        let name = latest.file_name().unwrap().to_string_lossy();
        assert!(name.contains(&format!("{n:020}")));

        let (restored, seq) = store.restore::<TestState>().await.unwrap();
        assert_eq!(seq, n);
        assert_eq!(restored.message, format!("concurrent-{n}"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_restore_after_concurrent_creates_single_thread() {
        restore_after_concurrent_creates(42).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_restore_after_concurrent_creates_multi_thread() {
        restore_after_concurrent_creates(42).await;
    }
}
