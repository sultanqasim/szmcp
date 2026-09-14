//! Minimal, hand-written Rust bindings for the Xapian **2.x** search engine.
//!
//! The focus is a read-only search path suitable for serving full-text
//! search over ZIM archives, plus just enough write support
//! ([`WritableDatabase`] and mutable [`Document`]s) to build and test
//! databases:
//!
//! - [`Database::open`] for a glass database on disk (directory or
//!   single-file), and [`Database::open_at`] to open a *single-file glass
//!   database embedded at an offset inside a larger file* - the zero-copy
//!   path for the `fulltext/xapian` item inside a ZIM archive. Xapian 2.x
//!   removed the 1.x `open_memview` constructor; `open_at` (an `fd`
//!   positioned at the database's first byte, memory-mapped by Xapian
//!   itself) is the replacement.
//! - [`QueryParser`] with prefixes, stemming and [`Operator`] control.
//!   Note: in Xapian 2.x the default combining operator for `parse_query`
//!   is **[`Operator::Or`]** (1.x used `And`); use
//!   [`QueryParser::set_default_op`] to change it.
//! - [`Stem`] to stem individual words with the same language an index was
//!   built with (the ZIM full-text indexes store unprefixed stems).
//! - [`Enquire`] / [`MSet`] with relevance sorting, random access to
//!   matches (docid, weight, percent, rank) and per-match [`Document`]s.
//!
//! Everything is synchronous. Error handling: Xapian exceptions are
//! captured in a per-thread buffer; failed calls return [`Err`][Error]
//! carrying the Xapian message ([`Error::msg`]).
//!
//! ## Thread-safety
//!
//! - [`Database`], [`Document`], [`Query`], [`MSet`]: `Send + Sync`
//!   (read-only Xapian objects are reference-counted handles; concurrent
//!   reads from multiple threads are the normal glass-backend use case).
//!   [`Document`] is an exception: `Send` only, because reading its
//!   data/value slots reuses a C++-side scratch buffer.
//! - [`Enquire`], [`QueryParser`], [`WritableDatabase`]: `Send` only - they
//!   hold mutable C++ state.
//!
//! ## Example
//!
//! ```no_run
//! use xapian2::{Database, Enquire, Operator, QueryParser};
//!
//! # fn main() -> xapian2::Result<()> {
//! let db = Database::open("/path/to/glass/db")?;
//!
//! let mut qp = QueryParser::new()?;
//! qp.set_stemmer("english")?;
//! qp.set_default_op(Operator::And)?;
//! let query = qp.parse_query("quantum computing")?;
//!
//! let mut enquire = Enquire::new(&db)?;
//! enquire.set_query(&query)?;
//! enquire.set_sort_by_relevance();
//!
//! let mset = enquire.get_mset(0, 20, 0)?;
//! for m in mset.iter() {
//!     let mut doc = db.get_document(m.docid)?;
//!     println!("{} {} {}", m.docid, m.weight, String::from_utf8_lossy(&doc.data()?));
//! }
//! # Ok(())
//! # }
//! ```

use std::ffi::{CStr, CString};
use std::fmt;
use std::io::SeekFrom;
use std::os::raw::{c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;

mod ffi {
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        pub fn xapian2_take_error() -> *const c_char;

        // Database
        pub fn xapian2_db_open(path: *const c_char, flags: c_int) -> *mut c_void;
        pub fn xapian2_db_open_fd(fd: c_int, flags: c_int) -> *mut c_void;
        pub fn xapian2_db_doccount(db: *mut c_void) -> u32;
        pub fn xapian2_db_get_document(db: *mut c_void, did: u32) -> *mut c_void;
        pub fn xapian2_db_compact(db: *mut c_void, output: *const c_char) -> c_int;
        pub fn xapian2_db_compact_single_file(db: *mut c_void, output: *const c_char) -> c_int;
        pub fn xapian2_db_free(db: *mut c_void);

        // Document
        pub fn xapian2_doc_new() -> *mut c_void;
        pub fn xapian2_doc_set_data(d: *mut c_void, data: *const c_char, len: u32) -> c_int;
        pub fn xapian2_doc_add_term(d: *mut c_void, term: *const c_char, len: u32, increment: u32) -> c_int;
        pub fn xapian2_doc_set_value(d: *mut c_void, slot: u32, value: *const c_char, len: u32) -> c_int;
        pub fn xapian2_doc_id(d: *mut c_void) -> u32;
        pub fn xapian2_doc_termlist_count(d: *mut c_void) -> u32;
        pub fn xapian2_doc_data(d: *mut c_void, out_len: *mut u32) -> *const c_char;
        pub fn xapian2_doc_value(d: *mut c_void, slot: u32, out_len: *mut u32) -> *const c_char;
        pub fn xapian2_doc_free(d: *mut c_void);

        // WritableDatabase
        pub fn xapian2_wdb_open(path: *const c_char, flags: c_int) -> *mut c_void;
        pub fn xapian2_wdb_add_document(db: *mut c_void, d: *mut c_void) -> u32;
        pub fn xapian2_wdb_commit(db: *mut c_void) -> c_int;
        pub fn xapian2_wdb_free(db: *mut c_void);

        // Stem
        pub fn xapian2_stem_new(language: *const c_char) -> *mut c_void;
        pub fn xapian2_stem_apply(
            s: *mut c_void,
            word: *const c_char,
            len: u32,
            out_len: *mut u32,
        ) -> *const c_char;
        pub fn xapian2_stem_free(s: *mut c_void);

        // QueryParser
        pub fn xapian2_qp_new() -> *mut c_void;
        pub fn xapian2_qp_set_default_op(qp: *mut c_void, op: c_int) -> c_int;
        pub fn xapian2_qp_add_prefix(qp: *mut c_void, field: *const c_char, prefix: *const c_char) -> c_int;
        pub fn xapian2_qp_add_boolean_prefix(qp: *mut c_void, field: *const c_char, prefix: *const c_char) -> c_int;
        pub fn xapian2_qp_set_stemmer(qp: *mut c_void, language: *const c_char) -> c_int;
        pub fn xapian2_qp_set_stemming_strategy(qp: *mut c_void, strategy: c_int) -> c_int;
        pub fn xapian2_qp_set_database(qp: *mut c_void, db: *mut c_void) -> c_int;
        pub fn xapian2_qp_parse_query(
            qp: *mut c_void,
            query: *const c_char,
            flags: u32,
            default_prefix: *const c_char,
        ) -> *mut c_void;
        pub fn xapian2_qp_free(qp: *mut c_void);

        // Query
        pub fn xapian2_query_term(term: *const c_char, len: u32, wqf: u32) -> *mut c_void;
        pub fn xapian2_query_match_all() -> *mut c_void;
        pub fn xapian2_query_combine(op: c_int, a: *mut c_void, b: *mut c_void) -> *mut c_void;
        pub fn xapian2_query_free(q: *mut c_void);

        // Enquire
        pub fn xapian2_enquire_new(db: *mut c_void) -> *mut c_void;
        pub fn xapian2_enquire_set_query(e: *mut c_void, q: *mut c_void, query_length: u32) -> c_int;
        pub fn xapian2_enquire_set_sort_by_relevance(e: *mut c_void);
        pub fn xapian2_enquire_get_mset(
            e: *mut c_void,
            first: u32,
            maxitems: u32,
            atleast: u32,
        ) -> *mut c_void;
        pub fn xapian2_enquire_free(e: *mut c_void);

        // MSet
        pub fn xapian2_mset_size(m: *mut c_void) -> u32;
        pub fn xapian2_mset_docid(m: *mut c_void, i: u32) -> u32;
        pub fn xapian2_mset_weight(m: *mut c_void, i: u32) -> f64;
        pub fn xapian2_mset_percent(m: *mut c_void, i: u32) -> i32;
        pub fn xapian2_mset_rank(m: *mut c_void, i: u32) -> u32;
        pub fn xapian2_mset_termfreq(m: *mut c_void, term: *const c_char, len: u32) -> u32;
        pub fn xapian2_mset_convert_to_percent(m: *mut c_void, weight: f64) -> i32;
        pub fn xapian2_mset_get_document(m: *mut c_void, i: u32) -> *mut c_void;
        pub fn xapian2_mset_free(m: *mut c_void);
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// An error returned by Xapian.
///
/// [`Error::msg`] is the human-readable Xapian error (for example
/// `Document 42 not found (DocNotFoundError)`), taken from the per-thread
/// error buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    /// Create an error with a custom message (used for I/O and validation
    /// failures that never reached Xapian).
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }

    /// The error message.
    pub fn msg(&self) -> &str {
        &self.0
    }

    /// Read the message Xapian stored for the most recent failure on this
    /// thread, or `None` when there was no failure.
    fn last() -> Option<String> {
        // SAFETY: the pointer is a valid NUL-terminated C string owned by
        // the C++ shim, valid until the next call that may fail; we copy it
        // out immediately.
        let msg = unsafe { CStr::from_ptr(ffi::xapian2_take_error()) }
            .to_string_lossy()
            .into_owned();
        if msg.is_empty() {
            None
        } else {
            Some(msg)
        }
    }

    /// The Xapian error from the last failed call on this thread, or a
    /// generic message when there was none.
    fn last_error(fallback: &str) -> Self {
        match Self::last() {
            Some(m) => Self(m),
            None => Self(fallback.into()),
        }
    }

    fn from_status(status: c_int) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(Self::last_error("unknown Xapian error"))
        }
    }

    fn from_ptr(ptr: *mut c_void, fallback: &str) -> Result<NonNull<c_void>> {
        if ptr.is_null() {
            Err(Self::last_error(fallback))
        } else {
            // SAFETY: ptr is non-null by the guard above.
            Ok(unsafe { NonNull::new_unchecked(ptr) })
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

// `?` converts `xapian2::Error` into `anyhow::Error` via anyhow's blanket
// impl over `std::error::Error + Send + Sync + 'static`.

/// Crate result type: `Result<T, xapian2::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

macro_rules! handle_debug {
    ($($t:ty),* $(,)?) => {
        $(
            impl fmt::Debug for $t {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.debug_tuple(stringify!($t)).field(&self.ptr).finish()
                }
            }
        )*
    };
}

handle_debug!(Database, Document, WritableDatabase, Query, QueryParser, Enquire, MSet, Stem);

// ---------------------------------------------------------------------------
// Flags / operators
// ---------------------------------------------------------------------------

/// Flags for opening databases.
///
/// Xapian 2.x read-only databases take no special flags (0 is correct); the
/// surviving constants are backend selectors and write options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DbFlags(pub u32);

impl DbFlags {
    /// Default: no flags (read-only, auto backend).
    pub const NONE: Self = Self(0);
    /// Use the glass backend.
    pub const GLASS: Self = Self(0x100);
    /// Use the stub backend.
    pub const STUB: Self = Self(0x300);
    /// Writable databases only: don't attempt to fsync.
    pub const NO_SYNC: Self = Self(0x04);
}

/// Query operators, matching `Xapian::Query::op` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Operator {
    And = 0,
    Or = 1,
    AndNot = 2,
    Xor = 3,
    AndMaybe = 4,
    Filter = 5,
    Near = 6,
    Phrase = 7,
    EliteSet = 10,
    Synonym = 13,
    Max = 14,
    Wildcard = 15,
}

impl Operator {
    fn as_c(self) -> c_int {
        self as c_int
    }
}

/// Feature flags for [`QueryParser::parse_query_with`], matching
/// `Xapian::QueryParser::FLAG_*` values.
pub mod parse_flags {
    /// Boolean `AND`/`OR`/parentheses handling.
    pub const FLAG_BOOLEAN: u32 = 1;
    /// Phrase handling with `""`.
    pub const FLAG_PHRASE: u32 = 2;
    /// `+`/`-` (love/hate) handling.
    pub const FLAG_LOVEHATE: u32 = 4;
    /// Boolean operators in any case.
    pub const FLAG_BOOLEAN_ANY_CASE: u32 = 8;
    /// Wildcard handling.
    pub const FLAG_WILDCARD: u32 = 16;
    /// Partial-word handling.
    pub const FLAG_PARTIAL: u32 = 64;
    /// Spelling correction (needs [`QueryParser::set_database`]).
    pub const FLAG_SPELLING_CORRECTION: u32 = 128;
    /// Synonym handling (needs [`QueryParser::set_database`]).
    pub const FLAG_SYNONYM: u32 = 256;
    /// Automatic database synonyms.
    pub const FLAG_AUTO_SYNONYMS: u32 = 512;
    /// Xapian 2.x default: phrases, booleans and love/hate.
    pub const FLAG_DEFAULT: u32 = FLAG_PHRASE | FLAG_BOOLEAN | FLAG_LOVEHATE;
}

/// How the [`QueryParser`] stems query terms, matching
/// `Xapian::QueryParser::stem_strategy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StemStrategy {
    /// Don't stem query terms.
    None = 0,
    /// Xapian's default: stem lowercase terms to `Z`-prefixed stems, leave
    /// capitalized terms unstemmed.
    Some = 1,
    /// Stem all terms to *unprefixed* stems.
    ///
    /// This is how libzim builds the full-text indexes embedded in openZIM
    /// archives (a TermGenerator with `STEM_ALL`), so queries against those
    /// indexes must be parsed with this strategy too.
    All = 2,
    /// Stem all terms to `Z`-prefixed stems.
    AllZ = 3,
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// A read-only Xapian database.
///
/// In Xapian 2.x a `Database` is a reference-counted handle; sharing one
/// across threads for concurrent searches is the normal use case
/// (`Send + Sync`). To search with multiple threads, create one [`Enquire`]
/// per thread from the same `Database` - each `Enquire` holds its own
/// reference to the database.
pub struct Database {
    ptr: NonNull<c_void>,
}

impl Database {
    /// Open a database at `path` (a glass directory, or a single-file
    /// glass database). Read-only, no flags.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_flags(path, DbFlags::NONE)
    }

    /// Open a database at `path` with the given flags.
    pub fn open_with_flags(path: impl AsRef<Path>, flags: DbFlags) -> Result<Self> {
        let c_path = cstr(&path.as_ref().to_string_lossy())?;
        // SAFETY: c_path is a valid NUL-terminated string for the duration
        // of the call; Xapian copies it.
        let ptr = unsafe { ffi::xapian2_db_open(c_path.as_ptr(), flags.0 as c_int) };
        Error::from_ptr(ptr, "failed to open database").map(|ptr| Self { ptr })
    }

    /// Open a *single-file glass database embedded at `offset` inside the
    /// file at `path`*, without copying it.
    ///
    /// This is the zero-copy replacement for Xapian 1.x's
    /// `Database::open_memview`: the file is opened read-only, positioned
    /// at `offset`, and the fd is handed to Xapian, which takes ownership
    /// of it and memory-maps the database region itself.
    ///
    /// Constraints:
    /// - the database must be a *single-file* glass database (ZIM
    ///   `fulltext/xapian` / `title/xapian` items are), and
    /// - the whole database region must lie within this one OS file. For
    ///   chunked ZIM archives (`.zimaa`, `.zimab`, ...) point `path` at
    ///   the chunk containing the index and `offset` at the index's
    ///   position within that chunk.
    ///
    /// On failure the fd is closed by Xapian.
    pub fn open_at(path: impl AsRef<Path>, offset: u64, flags: DbFlags) -> Result<Self> {
        use std::io::Seek;
        #[cfg(unix)]
        use std::os::fd::IntoRawFd;
        #[cfg(windows)]
        use std::os::windows::io::IntoRawHandle;

        let mut file = std::fs::File::open(path.as_ref()).map_err(|e| Error::new(e.to_string()))?;
        file.seek(SeekFrom::Start(offset)).map_err(|e| Error::new(e.to_string()))?;

        // SAFETY: Xapian takes ownership of the fd (closes it on success
        // and failure) and only ever reads through it.
        #[cfg(unix)]
        let ptr = unsafe { ffi::xapian2_db_open_fd(file.into_raw_fd() as c_int, flags.0 as c_int) };
        #[cfg(windows)]
        let ptr = unsafe {
            // Unsupported: Xapian needs an fd-like handle; unix-only for now.
            let _ = file.into_raw_handle();
            return Err(Self("open_at is only supported on unix".into()));
        };
        Error::from_ptr(ptr, "failed to open embedded database").map(|ptr| Self { ptr })
    }

    /// The number of documents in the database.
    pub fn doc_count(&self) -> u32 {
        // SAFETY: the handle is valid for the lifetime of `self`.
        unsafe { ffi::xapian2_db_doccount(self.handle()) }
    }

    /// Fetch the document with the given id.
    ///
    /// Fails with `InvalidArgumentError` for id 0 and
    /// `DocNotFoundError` for unknown ids.
    pub fn get_document(&self, id: u32) -> Result<Document> {
        // SAFETY: see doc_count.
        let ptr = unsafe { ffi::xapian2_db_get_document(self.handle(), id) };
        Error::from_ptr(ptr, "failed to fetch document").map(|ptr| Document { ptr })
    }

    /// Produce a compacted glass *directory* at `output`.
    pub fn compact_to(&self, output: impl AsRef<Path>) -> Result<()> {
        let c_out = cstr(&output.as_ref().to_string_lossy())?;
        // SAFETY: see doc_count.
        let status = unsafe { ffi::xapian2_db_compact(self.handle(), c_out.as_ptr()) };
        Error::from_status(status)
    }

    /// Produce a compacted **single-file** glass database in the file
    /// `output` (created or truncated).
    ///
    /// Single-file databases are what ZIM archives store as their
    /// `fulltext/xapian` item, and what [`Database::open_at`] can open.
    pub fn compact_single_file(&self, output: impl AsRef<Path>) -> Result<()> {
        let c_out = cstr(&output.as_ref().to_string_lossy())?;
        // SAFETY: see doc_count.
        let status = unsafe { ffi::xapian2_db_compact_single_file(self.handle(), c_out.as_ptr()) };
        Error::from_status(status)
    }

    fn handle(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // SAFETY: the handle was allocated by the shim.
        unsafe { ffi::xapian2_db_free(self.ptr.as_ptr()) };
    }
}

// SAFETY: Xapian read-only databases are reference-counted handles and glass
// databases support concurrent readers; all exposed methods take &self and
// are read-only.
unsafe impl Send for Database {}
unsafe impl Sync for Database {}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// A Xapian document (the data blob, term list and value slots).
///
/// Documents obtained from a database carry their id ([`Document::id`]).
/// [`Document::new`] creates an empty, mutable document for indexing.
///
/// `Send` but not `Sync`: reading the data/value slots reuses a C++-side
/// scratch buffer, so use one `Document` from one thread at a time.
pub struct Document {
    ptr: NonNull<c_void>,
}

impl Document {
    /// Create an empty document (for writing/indexing).
    pub fn new() -> Result<Self> {
        // SAFETY: no arguments.
        let ptr = unsafe { ffi::xapian2_doc_new() };
        Error::from_ptr(ptr, "failed to create document").map(|ptr| Self { ptr })
    }

    /// The document id, if this document came from a database.
    pub fn id(&self) -> Option<u32> {
        // SAFETY: the handle is valid for the lifetime of `self`.
        let id = unsafe { ffi::xapian2_doc_id(self.handle()) };
        (id != 0).then_some(id)
    }

    /// The number of terms in the document.
    pub fn termlist_count(&self) -> u32 {
        // SAFETY: the handle is valid for the lifetime of `self`.
        unsafe { ffi::xapian2_doc_termlist_count(self.handle()) }
    }

    /// The document's data blob.
    ///
    /// Xapian 2.x returns document data by value, so this is one C++-side
    /// copy of the blob (a Wikipedia article's data is an HTML page; the
    /// copy is negligible next to the search cost).
    pub fn data(&mut self) -> Result<Vec<u8>> {
        let mut len = 0u32;
        // SAFETY: the handle is valid; `out_len` is written before return.
        let ptr = unsafe { ffi::xapian2_doc_data(self.handle(), &mut len) };
        if ptr.is_null() {
            return Err(Error::last_error("failed to read document data"));
        }
        // SAFETY: the shim guarantees `len` bytes at `ptr`, valid until the
        // next call on this document.
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize).to_vec() })
    }

    /// The document's data blob as UTF-8.
    pub fn data_str(&mut self) -> Result<String> {
        String::from_utf8(self.data()?).map_err(|e| Error::new(e.to_string()))
    }

    /// The value stored in `slot` (Xapian value slots are 0-based), or
    /// empty if none is set.
    pub fn value(&mut self, slot: u32) -> Result<Vec<u8>> {
        let mut len = 0u32;
        // SAFETY: see data().
        let ptr = unsafe { ffi::xapian2_doc_value(self.handle(), slot, &mut len) };
        if ptr.is_null() {
            return Err(Error::last_error("failed to read document value"));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize).to_vec() })
    }

    /// Set the document's data blob.
    pub fn set_data(&mut self, data: impl AsRef<[u8]>) -> Result<()> {
        let data = data.as_ref();
        // SAFETY: `data` is a valid byte slice; the shim copies it.
        let status = unsafe {
            ffi::xapian2_doc_set_data(self.handle(), data.as_ptr() as *const _, data.len() as u32)
        };
        Error::from_status(status)
    }

    /// Add `term` to the document, incrementing its within-document
    /// frequency by `increment`.
    pub fn add_term(&mut self, term: &str, increment: u32) -> Result<()> {
        let bytes = term.as_bytes();
        // SAFETY: `bytes` is a valid byte slice; the shim copies it.
        let status = unsafe {
            ffi::xapian2_doc_add_term(self.handle(), bytes.as_ptr() as *const _, bytes.len() as u32, increment)
        };
        Error::from_status(status)
    }

    /// Set the value stored in `slot`.
    pub fn set_value(&mut self, slot: u32, value: impl AsRef<[u8]>) -> Result<()> {
        let value = value.as_ref();
        // SAFETY: `value` is a valid byte slice; the shim copies it.
        let status = unsafe {
            ffi::xapian2_doc_set_value(self.handle(), slot, value.as_ptr() as *const _, value.len() as u32)
        };
        Error::from_status(status)
    }

    fn handle(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for Document {
    fn drop(&mut self) {
        // SAFETY: the handle was allocated by the shim.
        unsafe { ffi::xapian2_doc_free(self.ptr.as_ptr()) };
    }
}

// SAFETY: a Document's C++ object is only mutated through `&mut self`
// methods (scratch buffer); moving it between threads is safe.
unsafe impl Send for Document {}

// ---------------------------------------------------------------------------
// Stem
// ---------------------------------------------------------------------------

/// A stemming algorithm ([`Xapian::Stem`]), e.g. `Stem::new("english")`.
///
/// Standalone stemmer, for stemming words the same way an index was built:
/// the full-text indexes embedded in openZIM archives store unprefixed
/// Porter2 stems (libzim indexes with `STEM_ALL`), so code that builds
/// query terms by hand must stem with the index's language to match.
///
/// `Send` only: `apply` reuses a C++-side scratch buffer, so use one `Stem`
/// from one thread at a time (mirrors [`Document`]).
pub struct Stem {
    ptr: NonNull<c_void>,
}

impl Stem {
    /// Create a stemmer for `language` (e.g. `"english"`); `"none"` gives a
    /// no-op stemmer. Fails with `InvalidArgumentError` for an unknown
    /// language.
    pub fn new(language: &str) -> Result<Self> {
        let lang = cstr(language)?;
        // SAFETY: `lang` is a valid NUL-terminated string for the call.
        let ptr = unsafe { ffi::xapian2_stem_new(lang.as_ptr()) };
        Error::from_ptr(ptr, "failed to create stemmer").map(|ptr| Self { ptr })
    }

    /// Stem `word` ("elephants" -> "eleph" with the English stemmer). An
    /// empty word comes back unchanged, like Xapian itself.
    pub fn apply(&mut self, word: &str) -> Result<String> {
        let bytes = word.as_bytes();
        let mut len = 0u32;
        // SAFETY: `bytes` is a valid byte slice; the shim copies the word
        // and writes `out_len` before returning.
        let ptr = unsafe {
            ffi::xapian2_stem_apply(
                self.handle(),
                bytes.as_ptr() as *const _,
                bytes.len() as u32,
                &mut len,
            )
        };
        if ptr.is_null() {
            return Err(Error::last_error("failed to stem word"));
        }
        // SAFETY: the shim guarantees `len` bytes at `ptr`, valid until the
        // next call on this stemmer.
        let stemmed =
            unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) }.to_vec();
        String::from_utf8(stemmed).map_err(|e| Error::new(e.to_string()))
    }

    fn handle(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for Stem {
    fn drop(&mut self) {
        // SAFETY: the handle was allocated by the shim.
        unsafe { ffi::xapian2_stem_free(self.ptr.as_ptr()) };
    }
}

// SAFETY: a Stem's C++ object is only mutated through `&mut self` methods
// (scratch buffer); moving it between threads is safe.
unsafe impl Send for Stem {}

// ---------------------------------------------------------------------------
// WritableDatabase (minimal: build/test databases)
// ---------------------------------------------------------------------------

/// A writable Xapian database - just enough to create databases for tests
/// or tooling. `Send` only: Xapian write transactions are not shareable.
pub struct WritableDatabase {
    ptr: NonNull<c_void>,
}

impl WritableDatabase {
    /// Create or open a writable database at `path` (a directory, for the
    /// default glass backend).
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let c_path = cstr(&path.as_ref().to_string_lossy())?;
        // SAFETY: see Database::open_with_flags.
        let ptr = unsafe { ffi::xapian2_wdb_open(c_path.as_ptr(), 0) };
        Error::from_ptr(ptr, "failed to open writable database").map(|ptr| Self { ptr })
    }

    /// Add a document; returns the assigned document id.
    pub fn add_document(&mut self, doc: &Document) -> Result<u32> {
        // SAFETY: both handles are valid for the duration of the call.
        let id = unsafe { ffi::xapian2_wdb_add_document(self.handle(), doc.handle()) };
        if id == 0 {
            Error::from_status(-1)?;
        }
        Ok(id)
    }

    /// Commit all pending changes to disk.
    pub fn commit(&mut self) -> Result<()> {
        // SAFETY: the handle is valid for the lifetime of `self`.
        let status = unsafe { ffi::xapian2_wdb_commit(self.handle()) };
        Error::from_status(status)
    }

    fn handle(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for WritableDatabase {
    fn drop(&mut self) {
        // SAFETY: the handle was allocated by the shim.
        unsafe { ffi::xapian2_wdb_free(self.ptr.as_ptr()) };
    }
}

unsafe impl Send for WritableDatabase {}

// ---------------------------------------------------------------------------
// Query / QueryParser
// ---------------------------------------------------------------------------

/// A parsed query, ready to hand to [`Enquire::set_query`].
pub struct Query {
    ptr: NonNull<c_void>,
}

impl Query {
    /// A query for a single indexed term.
    ///
    /// An empty term yields a match-all query.
    pub fn term(term: &str) -> Result<Self> {
        let bytes = term.as_bytes();
        // SAFETY: `bytes` is a valid byte slice; the shim copies it.
        let ptr =
            unsafe { ffi::xapian2_query_term(bytes.as_ptr() as *const _, bytes.len() as u32, 1) };
        Error::from_ptr(ptr, "failed to build term query").map(|ptr| Self { ptr })
    }

    /// A query matching all documents.
    pub fn match_all() -> Result<Self> {
        // SAFETY: no arguments.
        let ptr = unsafe { ffi::xapian2_query_match_all() };
        Error::from_ptr(ptr, "failed to build match-all query").map(|ptr| Self { ptr })
    }

    /// Combine two queries with the given operator.
    pub fn combine(op: Operator, a: &Query, b: &Query) -> Result<Self> {
        // SAFETY: both handles are valid for the duration of the call.
        let ptr = unsafe { ffi::xapian2_query_combine(op.as_c(), a.handle(), b.handle()) };
        Error::from_ptr(ptr, "failed to combine queries").map(|ptr| Self { ptr })
    }

    fn handle(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for Query {
    fn drop(&mut self) {
        // SAFETY: the handle was allocated by the shim.
        unsafe { ffi::xapian2_query_free(self.ptr.as_ptr()) };
    }
}

// SAFETY: Xapian 2.x Query objects are reference-counted and immutable once
// built; combining borrows sub-queries read-only.
unsafe impl Send for Query {}
unsafe impl Sync for Query {}

/// Builds [`Query`]s from user input strings.
///
/// `Send` only: a parser is a stateful, mutable C++ object - create one per
/// request, or share one behind a lock.
pub struct QueryParser {
    ptr: NonNull<c_void>,
}

impl QueryParser {
    /// Create a parser with Xapian 2.x defaults: default combining
    /// operator [`Operator::Or`], no stemmer, no prefixes.
    pub fn new() -> Result<Self> {
        // SAFETY: no arguments.
        let ptr = unsafe { ffi::xapian2_qp_new() };
        Error::from_ptr(ptr, "failed to create query parser").map(|ptr| Self { ptr })
    }

    /// Set the operator used to combine query items when the user input
    /// doesn't specify one ([`Operator::Or`] is the Xapian 2.x default).
    ///
    /// Accepted values: `And`, `Or`, `Near`, `Phrase`, `EliteSet`,
    /// `Synonym`, `Max`.
    pub fn set_default_op(&mut self, op: Operator) -> Result<()> {
        // SAFETY: the handle is valid for the lifetime of `self`.
        let status = unsafe { ffi::xapian2_qp_set_default_op(self.handle(), op.as_c()) };
        Error::from_status(status)
    }

    /// Map queries for field `field` to terms prefixed with `prefix`,
    /// e.g. `add_prefix("description", "XD")` makes `description:foo`
    /// search for `XD:foo`.
    pub fn add_prefix(&mut self, field: &str, prefix: &str) -> Result<()> {
        let field = cstr(field)?;
        let prefix = cstr(prefix)?;
        // SAFETY: both are valid NUL-terminated strings for the call.
        let status =
            unsafe { ffi::xapian2_qp_add_prefix(self.handle(), field.as_ptr(), prefix.as_ptr()) };
        Error::from_status(status)
    }

    /// Map queries for field `field` to *boolean* terms prefixed with
    /// `prefix`, so `field:value` acts as a filter.
    pub fn add_boolean_prefix(&mut self, field: &str, prefix: &str) -> Result<()> {
        let field = cstr(field)?;
        let prefix = cstr(prefix)?;
        // SAFETY: see add_prefix.
        let status = unsafe {
            ffi::xapian2_qp_add_boolean_prefix(self.handle(), field.as_ptr(), prefix.as_ptr())
        };
        Error::from_status(status)
    }

    /// Set the stemmer language (e.g. `"english"`). Use `"none"` to
    /// disable. Fails with `InvalidArgumentError` for an unknown language.
    pub fn set_stemmer(&mut self, language: &str) -> Result<()> {
        let lang = cstr(language)?;
        // SAFETY: see add_prefix.
        let status = unsafe { ffi::xapian2_qp_set_stemmer(self.handle(), lang.as_ptr()) };
        Error::from_status(status)
    }

    /// Set how query terms are stemmed (default: [`StemStrategy::Some`]).
    pub fn set_stemming_strategy(&mut self, strategy: StemStrategy) -> Result<()> {
        // SAFETY: the handle is valid for the lifetime of the call.
        let status = unsafe {
            ffi::xapian2_qp_set_stemming_strategy(self.handle(), strategy as c_int)
        };
        Error::from_status(status)
    }

    /// Associate the database with the parser (needed for
    /// [`parse_flags::FLAG_SPELLING_CORRECTION`] and synonym flags).
    pub fn set_database(&mut self, db: &Database) -> Result<()> {
        // SAFETY: both handles are valid for the duration of the call.
        let status = unsafe { ffi::xapian2_qp_set_database(self.handle(), db.handle()) };
        Error::from_status(status)
    }

    /// Parse `query` with the Xapian 2.x default flags
    /// ([`parse_flags::FLAG_DEFAULT`]) and no default prefix.
    ///
    /// Malformed user input (for example unbalanced parentheses) returns
    /// `Err` with the Xapian parse-error message.
    pub fn parse_query(&mut self, query: &str) -> Result<Query> {
        self.parse_query_with(query, parse_flags::FLAG_DEFAULT, "")
    }

    /// Parse `query` with explicit feature `flags` and `default_prefix`
    /// (the prefix prepended to terms when no field is given).
    pub fn parse_query_with(&mut self, query: &str, flags: u32, default_prefix: &str) -> Result<Query> {
        let query = cstr(query)?;
        let default_prefix = cstr(default_prefix)?;
        // SAFETY: both are valid NUL-terminated strings for the call.
        let ptr = unsafe {
            ffi::xapian2_qp_parse_query(self.handle(), query.as_ptr(), flags, default_prefix.as_ptr())
        };
        Error::from_ptr(ptr, "failed to parse query").map(|ptr| Query { ptr })
    }

    fn handle(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for QueryParser {
    fn drop(&mut self) {
        // SAFETY: the handle was allocated by the shim.
        unsafe { ffi::xapian2_qp_free(self.ptr.as_ptr()) };
    }
}

unsafe impl Send for QueryParser {}

// ---------------------------------------------------------------------------
// Enquire / MSet
// ---------------------------------------------------------------------------

/// Runs searches against a database.
///
/// The constructor copies the database handle (reference counted), so an
/// `Enquire` is independent of the lifetime of the `&Database` it was
/// created from, and the [`MSet`]s it returns are self-contained snapshots
/// with no ties to the `Enquire` or `Database`.
///
/// `Send` only: an `Enquire` holds mutable query/sort state. For
/// concurrent requests use one `Enquire` per thread (or worker) against a
/// shared `Database`.
pub struct Enquire {
    ptr: NonNull<c_void>,
}

impl Enquire {
    /// Create an enquire for the given database.
    pub fn new(db: &Database) -> Result<Self> {
        // SAFETY: the database handle is valid for the duration of the
        // call; the C++ Enquire copies it.
        let ptr = unsafe { ffi::xapian2_enquire_new(db.handle()) };
        Error::from_ptr(ptr, "failed to create enquire").map(|ptr| Self { ptr })
    }

    /// Set the query to run (no explicit query length; the parser's terms
    /// determine it).
    pub fn set_query(&mut self, query: &Query) -> Result<()> {
        // SAFETY: both handles are valid for the duration of the call.
        let status =
            unsafe { ffi::xapian2_enquire_set_query(self.handle(), query.handle(), 0) };
        Error::from_status(status)
    }

    /// Sort results by descending relevance.
    ///
    /// This is Xapian's default, so the call is optional; it exists for
    /// explicitness.
    pub fn set_sort_by_relevance(&mut self) {
        // SAFETY: the handle is valid for the lifetime of `self`; the call
        // cannot throw.
        unsafe { ffi::xapian2_enquire_set_sort_by_relevance(self.handle()) };
    }

    /// Run the current query.
    ///
    /// Returns at most `maxitems` matches starting at zero-based position
    /// `first`; Xapian stops early once it has examined `atleast` matching
    /// documents (pass 0 to let it decide: best precision at the cost of
    /// speed).
    pub fn get_mset(&self, first: u32, maxitems: u32, atleast: u32) -> Result<MSet> {
        // SAFETY: the handle is valid for the lifetime of `self`;
        // `get_mset` is a const C++ method.
        let ptr =
            unsafe { ffi::xapian2_enquire_get_mset(self.handle(), first, maxitems, atleast) };
        Error::from_ptr(ptr, "failed to run query").map(|ptr| MSet { ptr })
    }

    fn handle(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for Enquire {
    fn drop(&mut self) {
        // SAFETY: the handle was allocated by the shim.
        unsafe { ffi::xapian2_enquire_free(self.ptr.as_ptr()) };
    }
}

unsafe impl Send for Enquire {}

/// One match from an [`MSet`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match {
    /// The matched document's id.
    pub docid: u32,
    /// The match weight (not a percentage).
    pub weight: f64,
    /// The weight as a 0-100 percentage (see [`MSet::convert_to_percent`]).
    pub percent: i32,
    /// The match's absolute rank (its position in the full result set,
    /// counting matches before the requested window).
    pub rank: u32,
}

/// The set of matches for a query - an immutable snapshot with random
/// access, independent of the [`Enquire`] that produced it.
///
/// Positions (`0..size()`) are relative to the set: a set produced with
/// `get_mset(100, 20, 0)` addresses its first element at index 0, whose
/// [`Match::rank`] is 100.
pub struct MSet {
    ptr: NonNull<c_void>,
}

impl MSet {
    /// The number of matches in this set.
    pub fn size(&self) -> u32 {
        // SAFETY: the handle is valid for the lifetime of `self`; every
        // accessor below is a const read of the C++ MSet.
        unsafe { ffi::xapian2_mset_size(self.handle()) }
    }

    /// The docid at position `i` in the set.
    pub fn docid(&self, i: u32) -> u32 {
        // SAFETY: see size().
        unsafe { ffi::xapian2_mset_docid(self.handle(), i) }
    }

    /// The weight at position `i`.
    pub fn weight(&self, i: u32) -> f64 {
        // SAFETY: see size().
        unsafe { ffi::xapian2_mset_weight(self.handle(), i) }
    }

    /// The weight at position `i` as a percentage.
    pub fn percent(&self, i: u32) -> i32 {
        // SAFETY: see size().
        unsafe { ffi::xapian2_mset_percent(self.handle(), i) }
    }

    /// The absolute rank at position `i`.
    pub fn rank(&self, i: u32) -> u32 {
        // SAFETY: see size().
        unsafe { ffi::xapian2_mset_rank(self.handle(), i) }
    }

    /// The number of documents in which `term` occurs (as seen by this
    /// match set).
    pub fn termfreq(&self, term: &str) -> u32 {
        let bytes = term.as_bytes();
        // SAFETY: `bytes` is a valid byte slice; the shim copies it.
        unsafe {
            ffi::xapian2_mset_termfreq(self.handle(), bytes.as_ptr() as *const _, bytes.len() as u32)
        }
    }

    /// Convert an absolute weight to a 0-100 percentage, accounting for
    /// weighted query terms.
    pub fn convert_to_percent(&self, weight: f64) -> i32 {
        // SAFETY: see size().
        unsafe { ffi::xapian2_mset_convert_to_percent(self.handle(), weight) }
    }

    /// The [`Document`] at position `i` (with its id set).
    pub fn document(&self, i: u32) -> Result<Document> {
        // SAFETY: see size().
        let ptr = unsafe { ffi::xapian2_mset_get_document(self.handle(), i) };
        Error::from_ptr(ptr, "failed to fetch matched document").map(|ptr| Document { ptr })
    }

    /// Iterate the matches in the set, best first.
    pub fn iter(&self) -> MSetIter<'_> {
        MSetIter { mset: self, pos: 0 }
    }

    fn handle(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for MSet {
    fn drop(&mut self) {
        // SAFETY: the handle was allocated by the shim.
        unsafe { ffi::xapian2_mset_free(self.ptr.as_ptr()) };
    }
}

// SAFETY: an MSet is an immutable snapshot; all accessors are const reads.
unsafe impl Send for MSet {}
unsafe impl Sync for MSet {}

/// Iterator over an [`MSet`]'s matches, best first.
#[derive(Clone)]
pub struct MSetIter<'mset> {
    mset: &'mset MSet,
    pos: u32,
}

impl Iterator for MSetIter<'_> {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        if self.pos >= self.mset.size() {
            return None;
        }
        let m = Match {
            docid: self.mset.docid(self.pos),
            weight: self.mset.weight(self.pos),
            percent: self.mset.percent(self.pos),
            rank: self.mset.rank(self.pos),
        };
        self.pos += 1;
        Some(m)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.mset.size() - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for MSetIter<'_> {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::new("input string contains an interior NUL byte"))
}
