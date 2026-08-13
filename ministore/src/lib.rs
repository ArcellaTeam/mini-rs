// mini-rs/ministore/src/lib.rs
//
// Copyright (c) 2025 Arcella Team
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE>
// or the MIT license <LICENSE-MIT>, at your option.
// This file may not be copied, modified, or distributed
// except according to those terms.

//! A minimal, durable, append-only log store for serializable records.
//!
//! `ministore` is **not a state manager**. It is a **Write-Ahead Log (WAL) engine** that provides:
//! 1. **Durability**: every record is written to disk and `fsync`ed before the write returns.
//! 2. **Replay**: the entire log can be read back as a sequence of strongly-typed records.
//!
//! The caller is responsible for:
//! - Defining the record type (e.g., mutations, events, commands).
//! - Applying records to in-memory state.
//! - Managing concurrency (e.g., via `Arc<RwLock<MiniStore>>`).
//!
//! This design makes `ministore` ideal for building:
//! - Event-sourced systems
//! - State machines with durable logs
//! - Metadata stores (like Arcella's component registry)
//!
//! # Guarantees
//!
//! - **Atomicity**: each `append()` call writes exactly one record (as one JSON line).
//! - **Durability**: after `append()` returns `Ok(())`, the record is on stable storage.
//! - **Ordering**: records are replayed in the exact order they were appended.
//! - **Replay Safety**: the journal format includes a magic header to prevent misuse.
//!
//! # Journal Format
//!
//! The on-disk journal is a text file in [JSONL](http://jsonlines.org/) format:
//! ```text
//! // MINISTORE JOURNAL v0.1.4
//! {"Set":{"value":10}}
//! {"Inc":{"by":5}}
//! ```
//! - Line 1: magic header (for versioning and validation).
//! - Line N (N >= 2): one JSON-serialized record per line.
//!
//! The format is human-readable and easy to inspect/debug with standard tools (`cat`, `jq`, etc.).
//!
//! # Segmented Rotation
//!
//! To prevent unbounded growth, `ministore` supports **segmented WAL rotation**:
//! - When a segment reaches `max_bytes_per_segment`, it is renamed to `journal.jsonl.001`, etc.
//! - Only up to `max_segments` files are retained. Oldest are deleted automatically.
//! - `replay()` reads all segments in order: `.001`, `.002`, ..., then active `journal.jsonl`.
//!
//! # Example: Simple Counter
//!
//! ```
//! use ministore::MiniStore;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Serialize, Deserialize)]
//! enum CounterMutation {
//!     Set { value: u32 },
//!     Inc { delta: u32 },
//! }
//!
//! #[derive(Default)]
//! struct Counter {
//!     value: u32,
//! }
//!
//! impl Counter {
//!     fn apply(&mut self, mutation: &CounterMutation) {
//!         match mutation {
//!             CounterMutation::Set { value } => self.value = *value,
//!             CounterMutation::Inc { delta } => self.value += *delta,
//!         }
//!     }
//! }
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tmp = tempfile::tempdir()?;
//!     let path = tmp.path().join("counter.log");
//!
//!     // 1. Open the store
//!     let mut store = MiniStore::open(&path).await?;
//!
//!     // 2. Append mutations
//!     store.append(&CounterMutation::Set { value: 100 }).await?;
//!     store.append(&CounterMutation::Inc { delta: 25 }).await?;
//!
//!     // 3. Rebuild state from log
//!     let mut counter = Counter::default();
//!     let records: Vec<CounterMutation> = MiniStore::replay(&path).await?;
//!     for record in records {
//!         counter.apply(&record);
//!     }
//!
//!     assert_eq!(counter.value, 125);
//!     Ok(())
//! }
//! ```

use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
};

mod error;
pub use error::MiniStoreError;

/// A specialized [`Result`](std::result::Result) type for `ministore` operations.
pub type Result<T> = std::result::Result<T, MiniStoreError>;

/// Magic header written at the beginning of every new journal file.
/// Full header format: "// MINISTORE JOURNAL v<semver>\n"
///
/// Used to:
/// - Identify the file as a `ministore` journal.
/// - Validate the journal version during replay.
/// - Prevent accidental corruption by external tools.
const JOURNAL_MAGIC_CURRENT: &str = "// MINISTORE JOURNAL v0.1.4\n";

/// Prefix of the magic header (without version).
const JOURNAL_MAGIC_PREFIX: &str = "// MINISTORE JOURNAL v";

/// Configuration for `MiniStore` with support for segmented WAL rotation.
///
/// By default:
/// - `max_bytes_per_segment = 64 MiB`
/// - `max_segments = 3`
///
/// You can customize these via the builder pattern.
#[derive(Debug, Clone)]
pub struct MiniStoreOptions {
    /// Maximum size (in bytes) of a single journal segment before rotation.
    pub max_bytes_per_segment: u64,
    /// Maximum number of segments to retain (including the active file).
    /// Must be >= 1.
    pub max_segments: usize,
}

impl MiniStoreOptions {
    /// Creates a new set of options with default values:
    /// - 64 MiB per segment
    /// - 3 total segments
    pub fn new() -> Self {
        Self {
            max_bytes_per_segment: 64 * 1024 * 1024, // 64 MiB by default
            max_segments: 3,
        }
    }

    /// Sets the maximum size (in bytes) for a single journal segment.
    pub fn max_bytes_per_segment(mut self, bytes: u64) -> Self {
        self.max_bytes_per_segment = bytes;
        self
    }

    /// Sets the maximum number of journal segments to retain.
    ///
    /// The oldest segments are deleted when this limit is exceeded.
    /// Must be at least 1.
    pub fn max_segments(mut self, n: usize) -> Self {
        self.max_segments = n;
        self
    }

    /// Opens a `MiniStore` with these configuration options.
    pub async fn open<P: AsRef<Path>>(self, path: P) -> Result<MiniStore> {
        MiniStore::open_with_options(path.as_ref(), self).await
    }
}

/// A durable, append-only log store for serializable records.
///
/// `MiniStore` manages a journal file (or set of rotated segments) on disk.
/// It provides two core operations:
/// - [`append`](Self::append): write a record and guarantee durability.
/// - [`replay`](Self::replay): read all records in order (static method).
///
/// # Concurrency
///
/// `MiniStore` is **not thread-safe** by itself. To share it across tasks, wrap it in a
/// synchronization primitive like `Arc<RwLock<MiniStore>>` (for write-heavy workloads)
/// or `Arc<Mutex<MiniStore>>`.
///
/// # Durability & Rotation
///
/// - Every `append()` performs an `fsync` before returning.
/// - When the active segment reaches `max_bytes_per_segment`, it is rotated:
///   - Closed, renamed to `journal.jsonl.001`, etc.
///   - A new active file is created.
/// - Only `max_segments` files are kept; older ones are deleted.
/// - `replay()` automatically reads all segments in correct order.
#[derive(Debug)]
pub struct MiniStore {
    /// Base path of the active journal (e.g., `journal.jsonl`).
    base_path: PathBuf,
    /// Rotation and size configuration.
    opts: MiniStoreOptions,
    /// Direct writer to the active journal file (unbuffered for strict fsync guarantees).
    journal_file: Option<File>,
    /// Current size of the active journal in bytes (including header).
    current_size: u64,
}

impl MiniStore {
    /// Opens a `ministore` journal at the given path with default options.
    ///
    /// Equivalent to `MiniStoreOptions::new().open(path)`.
    ///
    /// # Behavior
    ///
    /// - If the file **does not exist**, it is created and initialized with the magic header.
    /// - If the file **exists and is empty**, the magic header is written.
    /// - If the file **exists and is non-empty**, it is assumed to be a valid journal
    ///   (the magic header must already be present; validated during [`replay`]).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path is not writable.
    /// - Parent directories cannot be created.
    /// - Disk I/O fails during magic header write.
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        MiniStoreOptions::new().open(path).await
    }

    /// Opens a `ministore` with explicit rotation settings.
    ///
    /// See [`MiniStoreOptions`] for configuration details.
    ///
    /// # Errors
    ///
    /// Returns `MiniStoreError::InvalidArgument` if `max_segments == 0`.
    async fn open_with_options<P: AsRef<Path>>(path: P, opts: MiniStoreOptions) -> Result<Self> {
        if opts.max_segments == 0 {
            return Err(MiniStoreError::InvalidArgument("max_segments must be >= 1".into()));
        }

        let path = path.as_ref();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Open file in write+append mode
        let mut file = OpenOptions::new().write(true).create(true).append(true).open(path).await?;

        let metadata = file.metadata().await?;
        if metadata.is_dir() {
            return Err(MiniStoreError::PathIsNotFile { path: path.to_path_buf() });
        }
        let mut current_size = metadata.len();

        // Initialize empty file with magic header
        if metadata.len() == 0 {
            file.write_all(JOURNAL_MAGIC_CURRENT.as_bytes()).await?;
            file.sync_all().await?;
            current_size = JOURNAL_MAGIC_CURRENT.len() as u64;
        }

        Ok(Self {
            base_path: path.to_path_buf(),
            opts,
            journal_file: Some(file),
            current_size,
        })
    }

    /// Appends a serializable record to the journal and ensures it is durably stored.
    ///
    /// The record is serialized as a single JSON line and immediately `fsync`ed to disk.
    /// This operation is **atomic** - either the entire record is written, or nothing is.
    ///
    /// If the active segment exceeds `max_bytes_per_segment`, it is rotated before return.
    ///
    /// # Guarantees
    ///
    /// After this method returns `Ok(())`:
    /// - The record is visible in subsequent [`replay`] calls.
    /// - The record will survive process termination or system crash.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Serialization fails (e.g., unsupported type).
    /// - Disk write fails (e.g., full disk).
    /// - `fsync` fails (e.g., I/O error).
    ///
    /// # Performance
    ///
    /// This is a **slow** operation due to the `fsync`. Use it for critical metadata,
    /// not high-frequency data.
    pub async fn append<R>(&mut self, record: &R) -> Result<()>
    where
        R: Serialize,
    {
        let mut json_bytes = serde_json::to_vec(record)?;
        json_bytes.push(b'\n');

        let file = self.journal_file.as_mut().ok_or(MiniStoreError::Io {
            source: std::io::Error::new(std::io::ErrorKind::NotConnected, "journal closed"),
        })?;

        file.write_all(&json_bytes).await?;
        file.sync_all().await?;

        self.current_size += json_bytes.len() as u64;

        if self.current_size >= self.opts.max_bytes_per_segment {
            self.rotate().await?;
        }

        Ok(())
    }

    /// Replays all records from a journal (including rotated segments) as a `Vec` of strongly-typed values.
    ///
    /// This is a **static method**  it does not require an open `MiniStore` instance.
    ///
    /// # Behavior
    ///
    /// - Scans the directory for files matching `base.001`, `base.002`, etc.
    /// - Reads segments in ascending order (`.001` � `.002` � ... � active file).
    /// - Validates magic header in each file.
    /// - If the journal **does not exist** or is **empty**, returns an empty `Vec`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any segment has a missing or invalid magic header.
    /// - Any line fails to deserialize as type `R`.
    /// - File I/O fails (e.g., permission denied).
    ///
    /// # Example
    ///
    /// ```
    /// # use ministore::MiniStore;
    /// # #[derive(serde::Deserialize, PartialEq, Debug)] struct Event { id: u32 }
    /// # #[tokio::main(flavor = "current_thread")] async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let events: Vec<Event> = MiniStore::replay("/tmp/events.log").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn replay<R, P: AsRef<Path>>(path: P) -> Result<Vec<R>>
    where
        R: DeserializeOwned,
    {
        let path = path.as_ref();
        let dir = path.parent().unwrap_or(Path::new("."));

        if !dir.exists() {
            return Ok(vec![]);
        }

        // Collect existing segments
        let segments = collect_segments(path).await?;        

        let mut current_line_num = 2; // magic = line 1
        let mut all_records = Vec::new();

        // Read archived segments in order
        for (_, seg_path) in segments {
            let lines_read =
                read_records_from_file(&seg_path, &mut all_records, current_line_num).await?;
            current_line_num += lines_read;
        }

        // Read active journal file (without suffix), if it exists and is non-empty
        if path.exists() && tokio::fs::metadata(path).await?.len() > 0 {
            read_records_from_file(path, &mut all_records, current_line_num).await?;
        }

        Ok(all_records)
    }

    /// Performs rotation of the active journal segment.
    ///
    /// Steps:
    /// 1. `fsync` and close the active file.
    /// 2. Rename it to `base.001`, `base.002`, etc.
    /// 3. Delete oldest segments if count > `max_segments - 1`.
    /// 4. Create a new active file with magic header.
    ///
    /// This method is called automatically from `append()` when size limit is reached.
    ///
    /// # Errors
    ///
    /// Returns I/O errors if rename, create, or delete fails.
    async fn rotate(&mut self) -> Result<()> {
        let old_file = self.journal_file.take().ok_or(MiniStoreError::Io {
            source: std::io::Error::new(std::io::ErrorKind::NotConnected, "journal closed"),
        })?;

        // 1. fsync and close
        old_file.sync_all().await?;
        drop(old_file);

        // 2. Collect existing segments
        let mut segments = collect_segments(self.base_path.as_path()).await?;

        // 3. Determine next segment number
        let next_num = segments.last().map(|(n, _)| *n + 1).unwrap_or(1);

        // 4. Rename active file
        let new_segment_path =
            PathBuf::from(format!("{}.{:06}", self.base_path.display(), next_num));
        tokio::fs::rename(&self.base_path, &new_segment_path).await?;

        // 5. Enforce max_segments: keep only (max_segments - 1) old segments
        segments.push((next_num, new_segment_path));
        segments.sort_by_key(|(n, _)| *n);

        if segments.len() > (self.opts.max_segments - 1) {
            let to_remove = segments.len() - (self.opts.max_segments - 1);
            for (_, path) in segments.drain(..to_remove) {
                tokio::fs::remove_file(path).await.ok();
            }
        }

        // 6. Create new active file
        let mut new_file =
            OpenOptions::new().write(true).create_new(true).open(&self.base_path).await?;

        new_file.write_all(JOURNAL_MAGIC_CURRENT.as_bytes()).await?;
        new_file.sync_all().await?;

        self.journal_file = Some(new_file);
        self.current_size = JOURNAL_MAGIC_CURRENT.len() as u64;

        Ok(())
    }

    /// Returns a stream (line iterator) over the records in a single journal file.
    ///
    /// Each line is parsed on-demand as `Result<T, MiniStoreError>`.
    /// This avoids loading the entire journal into memory.
    ///
    /// **Note**: this function reads **only one file**, not rotated segments.
    /// Use [`replay`] to read the full journal history.
    pub async fn stream<T>(path: impl AsRef<Path>) -> Result<JournalStream<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let file = File::open(path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        validate_magic_header(&mut lines).await?;
        Ok(JournalStream {
            lines,
            line_number: 2, // magic = line 1, start at 2
            _phantom: PhantomData,
        })
    }
}

/// Validates the magic header of a journal file.
///
/// Expects the first line to start with `// MINISTORE JOURNAL v`.
///
/// # Errors
///
/// Returns `MiniStoreError::MissingInitialState` if header is missing or invalid.
async fn validate_magic_header(lines: &mut Lines<BufReader<File>>) -> Result<()> {
    let magic = lines.next_line().await?.ok_or(MiniStoreError::MissingInitialState)?;
    if !magic.starts_with(JOURNAL_MAGIC_PREFIX) {
        return Err(MiniStoreError::MissingInitialState);
    }
    Ok(())
}

/// Collects all rotated journal segments matching the base path.
///
/// Returns a sorted list of `(segment_number, path)` pairs.
async fn collect_segments(base: &Path) -> Result<Vec<(u32, PathBuf)>> {
    let mut segments = Vec::new();
    let dir = base.parent().unwrap_or(Path::new("."));
    let file_name = base.file_name().unwrap_or_default();

    let mut entries = tokio::fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if ext.chars().all(|c| c.is_ascii_digit()) {
                if path.file_stem() == Some(file_name) {
                    if let Ok(num) = ext.parse::<u32>() {
                        segments.push((num, path));
                    }
                }
            }
        }
    }
    segments.sort_by_key(|(n, _)| *n);
    Ok(segments)
}

/// Reads all records from a single journal file (including magic header validation)
/// and appends them to the provided vector.
///
/// Returns the number of data lines read (excluding the magic header),
/// or an error if the file is invalid.
async fn read_records_from_file<R: DeserializeOwned>(
    path: &Path,
    records: &mut Vec<R>,
    mut line_num: usize, // starting line number for error reporting (usually 2)
) -> Result<usize> {
    let file = OpenOptions::new().read(true).open(path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Validate magic header (line 1)
    let magic = lines.next_line().await?.ok_or(MiniStoreError::MissingInitialState)?;
    if !magic.starts_with(JOURNAL_MAGIC_PREFIX) {
        return Err(MiniStoreError::MissingInitialState);
    }

    let mut count = 0;
    while let Some(line) = lines.next_line().await? {
        match serde_json::from_str(&line) {
            Ok(record) => {
                records.push(record);
                count += 1;
            },
            Err(e) => {
                return Err(MiniStoreError::Deserialize { line: line_num, source: e });
            },
        }
        line_num += 1;
    }

    Ok(count)
}

/// A stream over records in a single journal file.
///
/// Records are parsed on-demand. Does **not** include rotated segments.
pub struct JournalStream<T> {
    lines: Lines<BufReader<File>>,
    line_number: u64, // start at 2 (after header)
    _phantom: PhantomData<T>,
}

impl<T> JournalStream<T>
where
    T: for<'de> Deserialize<'de>,
{
    /// Asynchronously yields the next record from the journal.
    ///
    /// Returns `None` when the file ends.
    /// Returns `Some(Err(...))` on I/O or deserialization error.
    pub async fn next(&mut self) -> Option<Result<T>> {
        match self.lines.next_line().await {
            Ok(Some(line)) => match serde_json::from_str(&line) {
                Ok(t) => {
                    let record = Ok(t);
                    self.line_number += 1;
                    Some(record)
                },
                Err(e) => {
                    let err = MiniStoreError::Deserialize {
                        line: self.line_number as usize,
                        source: e,
                    };
                    self.line_number += 1;
                    Some(Err(err))
                },
            },
            Ok(None) => None,
            Err(e) => Some(Err(MiniStoreError::Io { source: e })),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serde::{Deserialize, Serialize};
    use tempfile::NamedTempFile;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    enum TestMutation {
        Set { value: u32 },
        Inc { by: u32 },
    }

    #[tokio::test]
    async fn test_ministore_append_replay() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.jsonl");

        // Append records
        let mut store = MiniStore::open(&path).await.unwrap();
        store.append(&TestMutation::Set { value: 10 }).await.unwrap();
        store.append(&TestMutation::Inc { by: 5 }).await.unwrap();

        // Replay
        let records: Vec<TestMutation> = MiniStore::replay(&path).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0], TestMutation::Set { value: 10 });
        assert_eq!(records[1], TestMutation::Inc { by: 5 });
    }

    #[tokio::test]
    async fn test_ministore_empty() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("empty.jsonl");

        let records: Vec<TestMutation> = MiniStore::replay(&path).await.unwrap();
        assert!(records.is_empty());
    }

    #[tokio::test]
    async fn test_stream_success() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", JOURNAL_MAGIC_CURRENT).unwrap();
        write!(file, "{}\n", serde_json::to_string(&TestMutation::Set { value: 10 }).unwrap())
            .unwrap();
        write!(file, "{}\n", serde_json::to_string(&TestMutation::Inc { by: 5 }).unwrap()).unwrap();
        file.flush().unwrap();

        let mut stream: JournalStream<TestMutation> = MiniStore::stream(file.path()).await.unwrap();
        let mut records: Vec<TestMutation> = Vec::new();

        while let Some(result) = stream.next().await {
            records.push(result.unwrap());
        }

        assert_eq!(records.len(), 2);
        assert_eq!(records[0], TestMutation::Set { value: 10 });
        assert_eq!(records[1], TestMutation::Inc { by: 5 });
    }

    #[tokio::test]
    async fn test_stream_empty_journal() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", JOURNAL_MAGIC_CURRENT).unwrap();
        file.flush().unwrap();

        let mut stream: JournalStream<TestMutation> = MiniStore::stream(file.path()).await.unwrap();
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_stream_missing_magic_header() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}\n", serde_json::to_string(&TestMutation::Set { value: 1 }).unwrap())
            .unwrap();
        file.flush().unwrap();

        let result = MiniStore::stream::<TestMutation>(file.path()).await;
        assert!(matches!(result, Err(MiniStoreError::MissingInitialState)));
    }

    #[tokio::test]
    async fn test_stream_invalid_json() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", JOURNAL_MAGIC_CURRENT).unwrap();
        write!(file, "invalid json\n").unwrap();
        file.flush().unwrap();

        let mut stream: JournalStream<TestMutation> = MiniStore::stream(file.path()).await.unwrap();
        let result = stream.next().await.unwrap();
        assert!(result.is_err());
        // Check that error is indeed a deserialization error
        if let Err(MiniStoreError::Deserialize { line: 2, .. }) = result {
            // OK
        } else {
            panic!("Expected Deserialize error on line 2");
        }
    }

    #[tokio::test]
    async fn test_stream_nonexistent_file() {
        let path = tempfile::tempdir().unwrap().path().join("nonexistent.jsonl");
        let result = MiniStore::stream::<TestMutation>(&path).await;
        // Expect I/O error (file not found)
        assert!(result.is_err());
        // Exact error depends on OS, but definitely not MissingInitialState
        assert!(!matches!(result, Err(MiniStoreError::MissingInitialState)));
    }

    #[tokio::test]
    async fn test_replay_order_with_rotation_and_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rot.log");

        // Very small segment + only 3 files total (2 archived + 1 active)
        let opts = MiniStoreOptions::new()
            .max_bytes_per_segment(100) // ~3-4 records per segment
            .max_segments(3); // � at most ~8-12 records retained

        let mut store = MiniStore::open_with_options(&path, opts).await.unwrap();

        // Write many records some will be dropped
        let total_written: usize = 20;
        for i in 0..total_written as u32 {
            store.append(&TestMutation::Set { value: i }).await.unwrap();
        }

        // Replay
        let records: Vec<TestMutation> = MiniStore::replay(&path).await.unwrap();

        // Should retain only the most recent records
        assert!(records.len() < total_written);
        assert!(!records.is_empty());

        // Critical check: order among retained records
        for (idx, record) in records.iter().enumerate() {
            // Values should be sequential: ..., 17, 18, 19
            let expected_value = (total_written - records.len() + idx) as u32;
            assert_eq!(
                record,
                &TestMutation::Set { value: expected_value },
                "At replay index {}: expected {}, got record {:?}",
                idx,
                expected_value,
                record
            );
        }

        // Final record must be the latest written
        if let Some(TestMutation::Set { value: last }) = records.last() {
            assert_eq!(*last, (total_written - 1) as u32);
        }
    }

    #[tokio::test]
    async fn test_replay_order_within_limits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rot.log");

        // Sufficiently large limit to retain everything
        let opts = MiniStoreOptions::new()
            .max_bytes_per_segment(200) // ~4-5 records per segment
            .max_segments(10);

        let mut store = MiniStore::open_with_options(&path, opts).await.unwrap();

        let total: usize = 12;
        for i in 0..total as u32 {
            store.append(&TestMutation::Set { value: i }).await.unwrap();
        }

        // Replay
        let records: Vec<TestMutation> = MiniStore::replay(&path).await.unwrap();

        // Verify count
        assert_eq!(records.len(), total);

        // Critical check: order
        for (idx, record) in records.iter().enumerate() {
            assert_eq!(
                record,
                &TestMutation::Set { value: idx as u32 },
                "Record at position {} has wrong value (expected {}, got {})",
                idx,
                idx,
                if let TestMutation::Set { value } = record { *value } else { u32::MAX }
            );
        }
    }
}
