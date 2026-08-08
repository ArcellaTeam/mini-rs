use std::io::{Read, Write};

use serde::{Serialize, de::DeserializeOwned};

use crate::error::{MiniSnapError, Result};

/// A codec used to serialize and deserialize snapshot envelopes.
///
/// The codec controls the file extension used for each snapshot file and how
/// state is encoded on disk.
pub trait Codec {
    /// Encode a value into the provided writer.
    fn encode<W, T>(&self, wr: &mut W, val: &T) -> Result<()>
    where
        W: Write + ?Sized,
        T: Serialize + ?Sized;

    /// Decode a value from the provided reader.
    fn decode<R, T>(&self, rd: R) -> Result<T>
    where
        R: Read,
        T: DeserializeOwned;

    /// Optional file extension for snapshot files.
    ///
    /// When empty, snapshot files are stored without an explicit extension.
    fn ext(&self) -> &str {
        ""
    }
}


#[cfg(feature = "json")]
pub mod json {
    use super::*;

    /// JSON codec using `serde_json`.
    #[derive(Default)]
    pub struct JsonCodec;

    impl Codec for JsonCodec {
        fn encode<W, T>(&self, wr: &mut W, val: &T) -> Result<()>
        where
            W: Write + ?Sized,
            T: Serialize + ?Sized,
        {
            serde_json::to_writer_pretty(wr, val)?;
            Ok(())
        }

        fn decode<R, T>(&self, rd: R) -> Result<T>
        where
            R: Read,
            T: DeserializeOwned,
        {
            let res = serde_json::from_reader(rd)?;
            Ok(res)
        }

        fn ext(&self) -> &'static str {
            "json"
        }
    }

    impl From<serde_json::Error> for MiniSnapError {
        fn from(err: serde_json::Error) -> Self {
            Self::Codec { source: err.into() }
        }
    }
}

#[cfg(feature = "rmp")]
pub mod rmp {
    use super::*;

    /// MessagePack codec using `rmp-serde`.
    #[derive(Default)]
    pub struct RmpCodec;

    impl Codec for RmpCodec {
        fn encode<W, T>(&self, wr: &mut W, val: &T) -> Result<()>
        where
            W: Write + ?Sized,
            T: Serialize + ?Sized,
        {
            rmp_serde::encode::write(wr, val)?;
            Ok(())
        }

        fn decode<R, T>(&self, rd: R) -> Result<T>
        where
            R: Read,
            T: DeserializeOwned,
        {
            Ok(rmp_serde::decode::from_read(rd)?)
        }

        fn ext(&self) -> &str {
            ""
        }
    }

    impl From<rmp_serde::encode::Error> for MiniSnapError {
        fn from(err: rmp_serde::encode::Error) -> Self {
            Self::Codec { source: err.into() }
        }
    }

    impl From<rmp_serde::decode::Error> for MiniSnapError {
        fn from(err: rmp_serde::decode::Error) -> Self {
            Self::Codec { source: err.into() }
        }
    }
}
