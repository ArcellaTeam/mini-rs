// mini-rs/ministate/src/lib.rs
//
// Copyright (c) 2025 Arcella Team
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE>
// or the MIT license <LICENSE-MIT>, at your option.
// This file may not be copied, modified, or distributed
// except according to those terms.

//! A minimal, in-memory state manager with durable WAL logging.
//!
//! `ministate` provides a simple yet robust way to maintain **mutable application state**
//! that survives process restarts, using an **append-only Write-Ahead Log (WAL)**.
//! It builds directly on [`ministore`] for reliable, human-readable journaling.
//!
//! ## Core Features
//!
//! - **In-memory state**: Fast reads via `RwLock`, full `Clone` on demand.
//! - **Durable mutations**: Every change is first written to disk (`fsync`-ed) before being applied.
//! - **Crash recovery**: On startup, the state is reconstructed by replaying the WAL from the beginning.
//! - **Logical ordering**: Each mutation is assigned a monotonically increasing sequence number.
//!
//! ## Optional Snapshot Support (`snapshot` feature)
//!
//! When the `snapshot` Cargo feature is enabled, `ministate` integrates with [`minisnap`] to:
//! - Save full state snapshots to disk for **faster recovery**.
//! - Enable future WAL compaction (truncating the log prefix after a snapshot is taken).
//!
//! > ⚠️ Snapshotting is **explicit** — you must call `create_snapshot()` manually.
//! > WAL compaction is not yet implemented but will be added in a future release.
//!
//! ## Concurrency Model
//!
//! - **Reads**: Concurrent via `snapshot()` (uses `RwLock::read`).
//! - **Writes**: Serialized via `apply()` (uses `RwLock::write`).
//! - **No hidden threads**: All I/O is explicit and `await`-driven.
//!
//! ## Guarantees
//!
//! - **Durability**: If `apply().await` returns `Ok(_)`, the mutation is guaranteed to be on disk.
//! - **Atomicity**: The in-memory state is updated **only if** the WAL write succeeds.
//! - **Ordering**: Mutations are applied in the exact order they appear in the WAL.
//! - **Recoverability**: Full state restoration is possible from the WAL alone (or WAL + snapshot).
//!
//! ## Use Cases
//!
//! - Stateful services that must recover after a crash (e.g., component registries, deployment specs).
//! - Embedded systems with limited resources but strict durability requirements.
//! - Local coordination primitives (e.g., leader election state, queue metadata).
//!
//! # Example
//!
//! ```rust
//! use ministate::{Mutator, StateManager};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Default, Clone, Serialize, Deserialize)]
//! struct Counter { value: u32 }
//!
//! #[derive(Serialize, Deserialize)]
//! struct Inc { by: u32 }
//!
//! impl Mutator<Counter> for Inc {
//!     fn apply(&self, state: &mut Counter) {
//!         state.value += self.by;
//!     }
//! }
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tmp = tempfile::tempdir()?;
//!     let dir = tmp.path();
//!
//!     let mgr = StateManager::open(dir, "counter.wal.jsonl").await?;
//!     mgr.apply(Inc { by: 10 }).await?;
//!     assert_eq!(mgr.snapshot().await.value, 10);
//!     Ok(())
//! }
//! ```

#[cfg(feature = "snapshot")]
use std::convert::TryFrom;
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
};

use ministore::MiniStore;
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::{Mutex, RwLock};

mod error;
pub use error::MiniStateError;

/// A mutation that can be applied to a state `S`.
///
/// User-defined types implement this trait to describe **how** they transform state.
/// This is the core extension point of `ministate`.
///
/// # Safety and Correctness Requirements
///
/// Implementations **must be**:
/// - **Pure**: No side effects (no I/O, no randomness, no time dependence).
/// - **Deterministic**: The same mutation applied to the same state **must always**
///   produce the identical result.
/// - **Idempotent in context**: While the mutation itself need not be idempotent,
///   replaying it **multiple times** during recovery must yield the same final state
///   as applying it once (which is naturally satisfied if the above rules are followed).
///
/// Violating these rules can lead to **state divergence after recovery**.
pub trait Mutator<S> {
    /// Apply this mutation to the mutable state.
    ///
    /// This method is called **after** the mutation has been successfully written
    /// and synchronized to the WAL, so it is **safe to assume durability**.
    ///
    /// The implementation should be fast and free of I/O to avoid blocking the state lock.
    fn apply(&self, state: &mut S);
}

/// Manages in-memory state with durable WAL logging.
///
/// The `StateManager` owns:
/// - An in-memory copy of the current state (`S`), protected by `RwLock`.
/// - A durable WAL journal (`MiniStore`) for crash recovery.
/// - A monotonically increasing sequence number (`seq`) tracking total applied mutations.
///
/// It is **`Send + Sync`** and can be safely shared across tasks.
/// Reads (`snapshot()`) are concurrent; writes (`apply()`) are serialized.
#[derive(Debug)]
pub struct StateManager<S, M> {
    /// The current in-memory state, protected by a reader-writer lock for concurrent reads.
    state: RwLock<S>,

    /// The underlying durable WAL store. Guarded by a `Mutex` to serialize appends.
    store: Mutex<MiniStore>,

    /// Logical sequence number: total number of successfully applied mutations.
    /// Starts at 0 for an empty state, increments by 1 after each successful `apply()`.
    /// Accessed atomically to allow lock-free reads via `sequence()`.
    seq: std::sync::atomic::AtomicU64,

    /// Base directory for state (used for future snapshot storage).
    state_dir: PathBuf,

    /// Full path to the journal file (e.g., `./state/deployments.wal.jsonl`).
    journal_path: PathBuf,

    /// Phantom marker to bind the generic mutation type `M` to this instance.
    /// Ensures type safety without storing an actual `M` value.
    _phantom: PhantomData<M>,
}

impl<S, M> StateManager<S, M>
where
    S: Default + Clone + Serialize + DeserializeOwned,
    M: Mutator<S> + Serialize + DeserializeOwned,
{
    /// Opens a state manager from a state directory and a custom journal filename.
    ///
    /// On first run (no journal exists), it initializes an empty state (`S::default()`).
    /// On subsequent runs, it **replays the entire WAL** to reconstruct the latest state.
    ///
    /// The journal is stored at `{state_dir}/{journal_file}` and is created if missing.
    ///
    /// ## Arguments
    ///
    /// - `state_dir`: Directory to store journal (and later, snapshots). Created if missing.
    /// - `journal_file`: Name of the WAL file (e.g., `"components.wal.jsonl"`).
    ///
    /// ## Returns
    ///
    /// A new `StateManager` with state fully restored from the WAL.
    ///
    /// ## Errors
    ///
    /// Returns an error if:
    /// - The state directory cannot be created or accessed.
    /// - The journal file exists but is corrupted or contains invalid records.
    /// - Disk I/O fails during replay or journal opening.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use ministate::{Mutator, StateManager};
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Default, Clone, Serialize, Deserialize)]
    /// struct Counter { value: u32 }
    ///
    /// #[derive(Serialize, Deserialize)]
    /// struct Inc { by: u32 }
    ///
    /// impl Mutator<Counter> for Inc {
    ///     fn apply(&self, state: &mut Counter) {
    ///         state.value += self.by;
    ///     }
    /// }
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tmp = tempfile::tempdir()?;
    /// let mgr = StateManager::open(tmp.path(), "counter.wal.jsonl").await?;
    /// mgr.apply(Inc { by: 1 }).await?;
    /// assert_eq!(mgr.snapshot().await.value, 1);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(not(feature = "snapshot"))]
    pub async fn open<P1, P2>(state_dir: P1, journal_file: P2) -> Result<Self, MiniStateError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
    {
        let state_dir = state_dir.as_ref();
        let journal_file = journal_file.as_ref();

        tokio::fs::create_dir_all(state_dir).await?;

        let journal_path = state_dir.join(journal_file);
        let records: Vec<M> = MiniStore::replay(&journal_path).await?;

        let mut state = S::default();
        for record in &records {
            record.apply(&mut state);
        }

        let store = MiniStore::open(&journal_path).await?;
        let seq = std::sync::atomic::AtomicU64::new(records.len() as u64);

        Ok(Self {
            state: RwLock::new(state),
            store: Mutex::new(store),
            seq,
            state_dir: state_dir.to_path_buf(),
            journal_path,
            _phantom: PhantomData,
        })
    }

    /// Opens a state manager from a state directory and a custom journal filename,
    /// **with snapshot support enabled**.
    ///
    /// This method:
    /// 1. Checks for a valid snapshot in `{state_dir}` (via [`minisnap::SnapStore`]).
    /// 2. If a snapshot exists, **loads it** and sets the initial state and sequence number.
    /// 3. **Replays only the tail of the WAL** (entries after `snapshot.seq`) to bring state up to date.
    /// 4. If no snapshot exists, falls back to full WAL replay (same as non-snapshot mode).
    ///
    /// This enables **fast startup** for systems with long WALs.
    ///
    /// The journal is stored at `{state_dir}/{journal_file}`.
    ///
    /// Requires the `snapshot` Cargo feature.
    ///
    /// See [`open_with_snapshot`] for full details.
    #[cfg(feature = "snapshot")]
    pub async fn open<P1, P2>(state_dir: P1, journal_file: P2) -> Result<Self, MiniStateError>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
    {
        Self::open_with_snapshot(state_dir, journal_file).await
    }

    /// Applies a mutation to the state **only after** it has been durably logged to the WAL.
    ///
    /// This is the **only way to modify managed state**. The operation is atomic:
    /// - If WAL write fails → mutation is **not applied**.
    /// - If WAL write succeeds → mutation **is applied** to in-memory state.
    ///
    /// The method holds the **write lock** for the entire duration, so concurrent `apply()`
    /// calls are serialized. Reads (`snapshot()`) are blocked only during the in-memory update step.
    ///
    /// # Returns
    ///
    /// The **logical sequence number** of this mutation (1-based index in the WAL).
    /// This number is globally unique per `StateManager` instance and monotonically increasing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The mutation cannot be serialized (e.g., contains non-UTF8 strings).
    /// - The underlying disk write or `fsync` fails.
    pub async fn apply(&self, mutation: M) -> Result<u64, MiniStateError> {
        let mut state_guard = self.state.write().await;
        let mut store_guard = self.store.lock().await;

        // 1. Durable write to WAL (includes fsync)
        store_guard.append(&mutation).await?;

        // 2. Increment sequence number atomically
        let new_seq = self.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

        // 3. Apply mutation to in-memory state (now that durability is guaranteed)
        mutation.apply(&mut state_guard);

        Ok(new_seq)
    }

    /// Returns a **clone** of the current in-memory state.
    ///
    /// This is a **read-only snapshot** — modifications to the returned value
    /// do not affect the managed state.
    ///
    /// This method **acquires a read lock**, so it is non-blocking for other readers
    /// and only blocks during concurrent `apply()` calls (while the write lock is held).
    ///
    /// **Note**: For large states, cloning may be expensive. Use judiciously.
    pub async fn snapshot(&self) -> S {
        self.state.read().await.clone()
    }

    /// Returns the current sequence number (number of successfully applied mutations).
    ///
    /// Starts at `0` for empty state, `1` after first `apply()`, etc.
    ///
    /// This value is **monotonically increasing** and reflects the total number
    /// of mutations that have been durably logged and applied.
    ///
    /// **This is a lock-free read** (uses atomic load), so it is very cheap.
    pub fn sequence(&self) -> u64 {
        self.seq.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the full path to the journal file.
    ///
    /// Useful for:
    /// - Manual inspection (`cat journal.wal.jsonl | jq`)
    /// - Backup scripts
    /// - Debugging recovery issues
    pub fn journal_path(&self) -> &Path {
        &self.journal_path
    }

    /// Returns the directory where state snapshots are stored.
    ///
    /// Useful for:
    /// - Backup scripts
    /// - Debugging recovery issues
    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }
}

// === Future: minisnap integration (opt-in via feature) ===

#[cfg(feature = "snapshot")]
mod snapshot_support {
    use minisnap::SnapStore;

    use super::*;

    impl<S, M> StateManager<S, M>
    where
        S: Default + Clone + Serialize + DeserializeOwned,
        M: Mutator<S> + Serialize + DeserializeOwned,
    {
        /// Creates a snapshot of the current state and stores it in the state directory.
        ///
        /// The snapshot is saved as a JSON file (e.g., `snapshot.json`) and includes
        /// the current sequence number for consistency tracking.
        ///
        /// **This does NOT truncate or compact the WAL.** Future versions will provide
        /// a `compact()` method to safely remove WAL entries preceding the last snapshot.
        ///
        /// Requires the `snapshot` Cargo feature.
        ///
        /// # Returns
        ///
        /// `Ok(())` if the snapshot was successfully serialized and written to disk.
        ///
        /// # Errors
        ///
        /// Returns an error if:
        /// - The current state cannot be serialized.
        /// - The snapshot file cannot be created or written (e.g., permission denied).
        pub async fn create_snapshot(&self) -> Result<(), MiniStateError> {
            let snap_store = SnapStore::new(&self.state_dir);
            let seq = self.sequence();
            snap_store.create(&self.snapshot().await, seq).await?;
            Ok(())
        }

        /// Opens a state manager using **snapshot + WAL tail replay** for fast recovery.
        ///
        /// This is the core implementation used when the `snapshot` feature is enabled.
        /// It is **not available** when the feature is disabled.
        ///
        /// ## Recovery algorithm
        ///
        /// 1. **Attempt to load snapshot**:  
        ///    Reads `{state_dir}/snapshot.json` and `{state_dir}/snapshot.seq`.
        /// 2. **If snapshot exists**:  
        ///    - Initialize state from snapshot.  
        ///    - Set current sequence number to `snapshot.seq`.  
        ///    - Replay only WAL records **after** this sequence.  
        /// 3. **If no snapshot**:  
        ///    - Start from `S::default()`.  
        ///    - Replay the **entire WAL** (same as non-snapshot mode).  
        /// 4. Open WAL for future appends.
        ///
        /// ## Guarantees
        ///
        /// - **Consistency**: The final state is **identical** to full WAL replay.
        /// - **Safety**: If snapshot is corrupt or missing, falls back to safe full replay.
        /// - **Durability**: No data is lost — all WAL entries are processed.
        ///
        /// ## Arguments
        ///
        /// - `state_dir`: Directory containing both WAL and optional snapshot files.
        /// - `journal_file`: Name of the WAL file (e.g., `"deployments.wal.jsonl"`).
        ///
        /// ## Returns
        ///
        /// A `StateManager` with state fully reconstructed.
        ///
        /// ## Errors
        ///
        /// Returns an error if:
        /// - Snapshot files are present but corrupted (invalid JSON or sequence).
        /// - WAL is corrupted at any point (including tail).
        /// - I/O fails during file access.
        ///
        /// ## Note
        ///
        /// **WAL compaction is not performed automatically.**  
        /// You must call [`create_snapshot`] periodically and implement truncation separately
        /// (future versions may add a `compact()` method).
        pub async fn open_with_snapshot<P1, P2>(
            state_dir: P1,
            journal_file: P2,
        ) -> Result<Self, MiniStateError>
        where
            P1: AsRef<Path>,
            P2: AsRef<Path>,
        {
            let state_dir = state_dir.as_ref();
            let journal_path = state_dir.join(journal_file.as_ref());

            // Try to load snapshot first
            let snap_store = SnapStore::new(state_dir);
            let (mut state, mut seq) = if snap_store.exists().await {
                let (s, s_seq) = snap_store.restore().await?;
                (s, s_seq)
            } else {
                (S::default(), 0)
            };

            // Read sequence number from snapshot.seq
            let skip_count =
                usize::try_from(seq).map_err(|_| MiniStateError::SnapshotSeqTooHigh)?;

            // Replay WAL from snapshot.seq onwards
            let all_records: Vec<M> = MiniStore::replay(&journal_path).await?;

            // Skip records before seq
            if skip_count > all_records.len() {
                return Err(MiniStateError::SnapshotSeqTooHigh);
            }

            // Skip records before seq
            let tail_records = all_records.into_iter().skip(seq as usize).collect::<Vec<_>>();

            // Apply all records
            for record in &tail_records {
                record.apply(&mut state);
                seq += 1;
            }

            // Open WAL for appending
            let store = MiniStore::open(&journal_path).await?;
            let seq_atomic = std::sync::atomic::AtomicU64::new(seq);

            Ok(Self {
                state: RwLock::new(state),
                store: Mutex::new(store),
                seq: seq_atomic,
                state_dir: state_dir.to_path_buf(),
                journal_path,
                _phantom: PhantomData,
            })
        }

        // TODO: compact() — truncate WAL up to last snapshot
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde::{Deserialize, Serialize};
    use tempfile::TempDir;

    use super::*;

    #[cfg(test)]
    mod base_tests {
        use super::*;

        #[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
        struct Counter {
            value: u32,
        }

        #[derive(Serialize, Deserialize)]
        struct Inc {
            by: u32,
        }

        impl Mutator<Counter> for Inc {
            fn apply(&self, state: &mut Counter) {
                state.value += self.by;
            }
        }

        async fn new_tmp_state_manager() -> (TempDir, StateManager<Counter, Inc>) {
            let tmp = tempfile::tempdir().unwrap();
            let mgr = StateManager::open(tmp.path(), "counter.wal.jsonl").await.unwrap();
            (tmp, mgr)
        }

        #[tokio::test]
        async fn test_initial_state_is_default() {
            let (_tmp, mgr) = new_tmp_state_manager().await;
            let state = mgr.snapshot().await;
            assert_eq!(state, Counter { value: 0 });
            assert_eq!(mgr.sequence(), 0);
        }

        #[tokio::test]
        async fn test_apply_increments_sequence_and_updates_state() {
            let (_tmp, mgr) = new_tmp_state_manager().await;

            let seq1 = mgr.apply(Inc { by: 5 }).await.unwrap();
            assert_eq!(seq1, 1);
            assert_eq!(mgr.sequence(), 1);

            let seq2 = mgr.apply(Inc { by: 10 }).await.unwrap();
            assert_eq!(seq2, 2);
            assert_eq!(mgr.sequence(), 2);

            let state = mgr.snapshot().await;
            assert_eq!(state.value, 15);
        }

        #[tokio::test]
        async fn test_recovery_after_restart() {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path();

            // First run: apply mutations
            {
                let mgr: StateManager<Counter, Inc> =
                    StateManager::open(path, "counter.wal.jsonl").await.unwrap();
                mgr.apply(Inc { by: 100 }).await.unwrap();
                mgr.apply(Inc { by: 200 }).await.unwrap();
                assert_eq!(mgr.snapshot().await.value, 300);
            }

            // Second run: recover from WAL
            {
                let mgr: StateManager<Counter, Inc> =
                    StateManager::open(path, "counter.wal.jsonl").await.unwrap();
                let state = mgr.snapshot().await;
                assert_eq!(state.value, 300);
                assert_eq!(mgr.sequence(), 2);
            }
        }

        #[tokio::test]
        async fn test_concurrent_reads_are_allowed() {
            let (_tmp, mgr) = new_tmp_state_manager().await;
            let mgr = Arc::new(mgr);

            mgr.apply(Inc { by: 42 }).await.unwrap();

            // Spawn multiple concurrent readers
            let handles: Vec<_> = (0..5)
                .map(|_| {
                    let mgr = Arc::clone(&mgr);
                    tokio::spawn(async move {
                        let s = mgr.snapshot().await;
                        s.value
                    })
                })
                .collect();

            for handle in handles {
                assert_eq!(handle.await.unwrap(), 42);
            }
        }

        #[tokio::test]
        async fn test_writes_are_serialized() {
            let (_tmp, mgr) = new_tmp_state_manager().await;
            let mgr = Arc::new(mgr);

            // Parallel applies should be serialized and summed correctly
            let handles: Vec<_> = (0..10)
                .map(|_i| {
                    let mgr = Arc::clone(&mgr);
                    tokio::spawn(async move {
                        mgr.apply(Inc { by: 1 }).await.unwrap();
                    })
                })
                .collect();

            for handle in handles {
                handle.await.unwrap();
            }

            let state = mgr.snapshot().await;
            assert_eq!(state.value, 10); // 10 increments of 1
            assert_eq!(mgr.sequence(), 10);
        }

        #[tokio::test]
        async fn test_journal_path_and_state_dir() {
            let tmp = tempfile::tempdir().unwrap();
            let mgr: StateManager<Counter, Inc> =
                StateManager::open(tmp.path(), "my.wal.jsonl").await.unwrap();

            assert_eq!(mgr.journal_path(), tmp.path().join("my.wal.jsonl"));
            assert_eq!(mgr.state_dir(), tmp.path());
        }
    }

    #[cfg(all(test, feature = "snapshot"))]
    mod snapshot_tests {
        use serde::{Deserialize, Serialize};
        use tempfile::TempDir;

        use super::*;

        #[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
        struct Config {
            version: u32,
            enabled: bool,
        }

        #[derive(Serialize, Deserialize)]
        struct UpdateVersion {
            to: u32,
        }

        impl Mutator<Config> for UpdateVersion {
            fn apply(&self, state: &mut Config) {
                state.version = self.to;
            }
        }

        async fn new_tmp_state_manager() -> (TempDir, StateManager<Config, UpdateVersion>) {
            let tmp = tempfile::tempdir().unwrap();
            let mgr = StateManager::open(tmp.path(), "config.wal.jsonl").await.unwrap();
            (tmp, mgr)
        }

        #[tokio::test]
        async fn test_create_snapshot() {
            let (_tmp, mgr) = new_tmp_state_manager().await;
            mgr.apply(UpdateVersion { to: 42 }).await.unwrap();
            mgr.apply(UpdateVersion { to: 84 }).await.unwrap();

            // Create snapshot
            mgr.create_snapshot().await.unwrap();

            // Verify snapshot file exists and contains correct data
            let snap_path = mgr.state_dir().join("snapshot.json");
            assert!(snap_path.exists());

            let _content = tokio::fs::read_to_string(&snap_path).await.unwrap();

            let snap_store = minisnap::SnapStore::new(mgr.state_dir());
            let (restored_state, restored_seq) = snap_store.restore::<Config>().await.unwrap();

            assert_eq!(restored_state.version, 84);
            assert_eq!(restored_seq, 2);
        }

        #[tokio::test]
        async fn test_recovery_from_snapshot_and_wal_tail() {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path();

            // 1. Create state and snapshot
            {
                let mgr: StateManager<Config, UpdateVersion> =
                    StateManager::open(path, "test.wal.jsonl").await.unwrap();
                mgr.apply(UpdateVersion { to: 10 }).await.unwrap(); // seq=1
                mgr.apply(UpdateVersion { to: 20 }).await.unwrap(); // seq=2
                mgr.create_snapshot().await.unwrap();
                mgr.apply(UpdateVersion { to: 30 }).await.unwrap(); // seq=3 — после снапшота!
            }

            // 2. Recover from snapshot and WAL tail - should be 30
            {
                let mgr: StateManager<Config, UpdateVersion> =
                    StateManager::open(path, "test.wal.jsonl").await.unwrap();
                assert_eq!(mgr.snapshot().await.version, 30);
                assert_eq!(mgr.sequence(), 3);
            }
        }
    }
}
