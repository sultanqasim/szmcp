//! Pure-Rust ZIM 6.x container writer, byte-layout-compatible with what
//! libzim 9.8.2's `zim::writer::Creator` produces.
//!
//! Mirrors libzim's writer internals (verified against libzim 9.8.2):
//! - `creator.cpp` — entry lifecycle, finish order, file layout, header,
//!   checksum (`writeLastParts` / `fillHeader`).
//! - `dirent.cpp` + `tinyString.h` — binary dirent layout and the
//!   path/title tiny-string packing (title omitted when equal to the path).
//! - `cluster.cpp` — blob offset tables, extended (u64) offsets, zstd frames.
//! - `counterHandler.cpp` — the `M/Counter` metadata.
//! - the title listing provider (`TitleListingProvider`).
//!
//! Structural rules replicated here:
//! - Dirents live in a `BTreeMap` keyed by `(namespace byte, path)`, so map
//!   iteration order == the archive's entry order (sorted by namespace byte,
//!   then path bytes — exactly libzim's `comparePath`/`strcmp`).
//! - A redirection to a path that does not exist yet first creates a
//!   *placeholder* dirent; a later `add_item` with the same `(ns, path)`
//!   replaces it in place (`addOrUpdate` semantics).
//! - `finish` creates the `X/listing/titleOrdered/v1` and `W/mainPage` (when a
//!   main path is set and resolvable) dirents, drops dangling redirects (and
//!   their placeholders) plus redirect loops/blind chains, assigns entry
//!   indexes by sorted position, sorts + remaps the MIME list, adds the
//!   M/Counter and title-listing contents, then writes the file: clusters,
//!   dirents, path pointer table, cluster pointer table, 80-byte header last,
//!   and finally the MD5 checksum of everything before it.
//! - Clusters: at most one open compressed and one uncompressed cluster at a
//!   time; a cluster is closed when it holds at least one blob and its size
//!   (offset table + blob bytes) plus the incoming blob would reach 2 MiB.
//!   Compressed clusters hold one zstd frame (level 19) of the offset table +
//!   blob bytes; uncompressed clusters store those bytes verbatim.
//!
//! The only API deviation from the plan sketch: [`ZimCreator::add_item`] takes
//! an explicit `front_article` flag (libzim's FRONT_ARTICLE hint, which drives
//! the `X/listing/titleOrdered/v1` contents; zim2zim sets it on every item).

use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use md5::Digest;

use crate::zimcommon::{
    CLUSTER_EXTENDED_BIT, CLUSTER_UNCOMPRESSED, CLUSTER_ZSTD, HEADER_SIZE, MIME_REDIRECT,
    NO_LAYOUT_PAGE, NO_TITLE_PTR_POS, ZimHeader,
};

/// The ZIM major version this writer produces (libzim 9.8.2's default).
const ZIM_MAJOR: u16 = 6;
const ZIM_MINOR_VERSION: u16 = 3;
/// First cluster offset; the bytes between the MIME list and here stay zero.
const CLUSTER_BASE_OFFSET: u64 = 2048;
/// libzim's `DEFAULT_CLUSTER_SIZE`: clusters are closed once their data would
/// reach this size.
const CLUSTER_TARGET_SIZE: u64 = 2 * 1024 * 1024;
/// Mimetype of the `X/listing/titleOrdered/v1` entry.
const LISTING_MIME: &str = "application/octet-stream+zimlisting";

/// Blobs at or above this size stream straight to disk as their own
/// uncompressed cluster instead of accumulating in the open cluster's RAM
/// buffer until finish (embedded Xapian databases are the only such blobs).
const BIG_BLOB_THRESHOLD: u64 = 1024 * 1024;

/// Which open cluster an item's blob landed in (compressed or not).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Compressed,
    Uncompressed,
}

/// An open cluster: blobs are appended one by one; `blob_ends` holds the
/// running end offset of every blob (starting with 0), `data` the
/// concatenated bytes.
#[derive(Debug)]
struct OpenCluster {
    compress: bool,
    /// End offset of each blob so far; `blob_ends[i]` is the end of blob i
    /// (equivalently the start of blob i+1). Empty blobs still grow this.
    blob_ends: Vec<u64>,
    data: Vec<u8>,
}

impl OpenCluster {
    fn new(compress: bool) -> Self {
        OpenCluster { compress, blob_ends: vec![0], data: Vec::new() }
    }

    /// Number of blobs added so far (libzim's `Cluster::count`).
    fn count(&self) -> u32 {
        (self.blob_ends.len() - 1) as u32
    }

    fn data_size(&self) -> u64 {
        self.blob_ends[self.blob_ends.len() - 1]
    }

    /// Whether any blob offset exceeds 32 bits (libzim's `isExtended`).
    fn is_extended(&self) -> bool {
        self.data_size() > u32::MAX as u64
    }

    /// libzim's `Cluster::size`: offset table + blob bytes. The table has one
    /// entry per blob plus the end sentinel, each `width` bytes wide.
    fn size(&self) -> u64 {
        let width = if self.is_extended() { 8 } else { 4 };
        self.blob_ends.len() as u64 * width + self.data_size()
    }

    /// Append one blob. Empty blobs still grow the offset table.
    fn push(&mut self, content: &[u8]) {
        self.blob_ends.push(self.data_size() + content.len() as u64);
        self.data.extend_from_slice(content);
    }

    /// Write the cluster: the info byte, then the blob offset table (each
    /// entry gets the table's byte size added — libzim's `delta`) and the blob
    /// bytes. Compressed clusters hold ONE zstd frame (level 19, like libzim's
    /// `ZSTD_INFO::init_stream_encoder`) over the table + blob bytes; the info
    /// Write the cluster, returning its serialized size (bytes written).
    fn write_to(&self, w: &mut impl Write) -> io::Result<u64> {
        let extended = self.is_extended();
        let width = if extended { 8 } else { 4 };
        let delta = self.blob_ends.len() as u64 * width;
        let info = if extended { CLUSTER_EXTENDED_BIT } else { 0 }
            + if self.compress { CLUSTER_ZSTD } else { CLUSTER_UNCOMPRESSED };

        // blob_ends doubles as libzim's `blobOffsets`: entry i is the start
        // offset of blob i within (table + data); the last entry is the total.
        // Entries are u32 unless the cluster is extended.
        let mut table = Vec::with_capacity(self.blob_ends.len() * width as usize);
        for &off in &self.blob_ends {
            if extended {
                table.extend_from_slice(&(off + delta).to_le_bytes());
            } else {
                table.extend_from_slice(&(((off + delta) as u32).to_le_bytes()));
            }
        }
        w.write_all(&[info])?;
        let mut written = 1u64;
        if self.compress {
            let mut enc = zstd::stream::Encoder::new(Vec::new(), 19)?;
            enc.write_all(&table)?;
            enc.write_all(&self.data)?;
            let comp = enc.finish()?;
            w.write_all(&comp)?;
            written += comp.len() as u64;
        } else {
            w.write_all(&table)?;
            w.write_all(&self.data)?;
            written += table.len() as u64 + self.data.len() as u64;
        }
        Ok(written)
    }
}

/// Location of an item's blob: which kind of open cluster, which generation
/// of it, and the blob index within the cluster. Returned by
/// [`ZimCreator::add_blob`] and carried in [`DirentOut::Item`]; the final
/// archive cluster index is resolved by the writer at write time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobRef {
    pub compress: bool,
    pub generation: u32,
    pub blob: u32,
}

/// One dirent handed to the streaming writer: everything the archive body
/// needs, already resolved except the blob's cluster index (resolved at
/// write time) and the MIME id (resolved by string lookup).
#[derive(Debug)]
pub enum DirentOut {
    Item { ns: u8, path: String, title: String, mime: String, blob: BlobRef },
    Redirect { ns: u8, path: String, title: String, target_idx: u32 },
}

/// A dirent's serializable fields, fully resolved.
enum DirentBytes {
    Item { mime_idx: u16, cluster: u32, blob: u32 },
    Redirect { target: u32 },
}

/// Streaming file state: the archive body is written to the output file as
/// it is produced (closed clusters stream immediately, dirents as they are
/// emitted), so RAM never holds content or dirent bytes. The MIME list and
/// the pointer tables are patched/written at the very end.
struct WriteState {
    out: io::BufWriter<std::fs::File>,
    /// Absolute write position (starts at CLUSTER_BASE_OFFSET).
    pos: u64,
    /// Absolute offset of every closed cluster, in close order.
    cluster_offsets: Vec<u64>,
    /// Relative offset of every emitted dirent (the path pointer table).
    dirent_offsets: Vec<u64>,
    /// Absolute offset/end of the dirent section.
    dirent_start: Option<u64>,
    dirent_end: u64,
    /// Streaming mode: tail dirents registered before `begin_write`, in
    /// sorted order, emitted interleaved with (and after) the C stream.
    tails: Vec<((u8, String), Dirent)>,
    tail_next: usize,
    /// Ordinal of the W/mainPage dirent among the emitted dirents.
    main_page_idx: Option<u32>,
}

impl WriteState {
    /// Serialize one resolved dirent and record its path pointer.
    fn emit(&mut self, ns: u8, path: &str, title: &str, k: &DirentBytes) -> io::Result<()> {
        let start = *self.dirent_start.get_or_insert(self.pos);
        if ns == b'W' && path == "mainPage" {
            self.main_page_idx = Some(self.dirent_offsets.len() as u32);
        }
        self.dirent_offsets.push(self.pos - start);
        write_dirent(&mut self.out, ns, path, title, k)?;
        let head = match k {
            DirentBytes::Item { .. } => 16,
            DirentBytes::Redirect { .. } => 12,
        };
        // pathTitle: path, NUL, then the title only when it differs from the
        // path — one terminating NUL is always written (libzim's
        // PathTitleTinyString::concat writes path NUL [title] NUL).
        let title_part = if title != path { title.len() as u64 } else { 0 };
        self.pos += head + path.len() as u64 + 1 + title_part + 1;
        self.dirent_end = self.pos;
        Ok(())
    }
}

#[derive(Clone)]
enum DirentKind {
    Item {
        /// MIME id in insertion order (libzim's `getMimeTypeIdx`), remapped to
        /// the sorted position at finish time.
        mime_idx: u16,
        /// Which cluster holds the blob and the blob index within it.
        blob: BlobRef,
    },
    Redirect {
        /// (namespace, path) of the target dirent.
        target: (u8, String),
    },
    /// A redirect whose target entry index is already resolved (the
    /// streaming path's W/mainPage, which targets a ranked member).
    RedirectIdx(u32),
    /// Redirect target that was never filled in (libzim's `isPlaceholder`).
    Placeholder,
}

#[derive(Clone)]
struct Dirent {
    kind: DirentKind,
    /// Stored title (may be empty; such dirents behave as if titled with their
    /// path, matching libzim's tiny-string reader).
    title: String,
    front_article: bool,
    /// Final entry index, set at finish time.
    idx: u32,
    /// Dirents dropped by the dangling-redirect / blind-chain cleanup.
    removed: bool,
}

impl Dirent {
    fn is_redirect(&self) -> bool {
        matches!(self.kind, DirentKind::Redirect { .. } | DirentKind::Placeholder)
    }
    /// The title a reader sees: the stored title, or the path when none was
    /// stored (libzim's `PathTitleTinyString::getTitle`).
    fn effective_title<'a>(&'a self, path: &'a str) -> &'a str {
        if self.title.is_empty() {
            path
        } else {
            &self.title
        }
    }
}

/// A ZIM archive under construction.
pub struct ZimCreator {
    out_path: PathBuf,
    uuid: [u8; 16],
    /// (namespace byte, path) -> dirent; iteration order is the entry order
    /// (BTreeMap's natural order == `zimcommon::dirent_order`).
    dirents: BTreeMap<(u8, String), Dirent>,
    /// MIME string by insertion-order id.
    mime_by_idx: Vec<String>,
    /// The same strings in insertion order, frozen when the list is sorted
    /// (dirents registered before the freeze carry insertion-order ids).
    mime_insertion: Vec<String>,
    /// The main entry path (for the `W/mainPage` redirect at finish; record
    /// mode resolves it against the dirent map).
    main_path: String,
    /// Streaming mode: the already-resolved target entry index of the
    /// `W/mainPage` redirect (`None`: no main page).
    main_target: Option<u32>,
    /// Streaming mode: the caller-built `X/listing/titleOrdered/v1` bytes,
    /// added to the uncompressed cluster at `finish_write`.
    listing_bytes: Vec<u8>,
    comp_cluster: OpenCluster,
    uncomp_cluster: OpenCluster,
    /// Number of clusters closed per compression kind (the "generation" of the
    /// currently open clusters). Closed clusters stream straight to the
    /// output file, so only their offsets survive.
    comp_generation: u32,
    uncomp_generation: u32,
    /// Generation -> archive cluster index, per slot.
    comp_gen_idx: Vec<u32>,
    uncomp_gen_idx: Vec<u32>,
    /// Mime string -> count for `M/Counter` (sorted like libzim's std::map).
    mime_counter: BTreeMap<String, u64>,
    /// Streaming file state, opened lazily on the first cluster close (or at
    /// `begin_write`) and gone once the file is complete.
    w: Option<WriteState>,
}

/// 16 random bytes from `/dev/urandom` (falling back to a time/pid hash if
/// unavailable — libzim's `Uuid()` also generates random bytes).
fn random_uuid() -> [u8; 16] {
    let mut uuid = [0u8; 16];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut uuid).is_ok() {
            return uuid;
        }
    }
    // Extremely unlikely fallback: mix time and pid through a splitmix-like mix.
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15)
        ^ ((std::process::id() as u64) << 32);
    for b in uuid.iter_mut() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_add(0x94D049BB133111EB);
        *b = (z ^ (z >> 31)) as u8;
    }
    uuid
}

/// `zim::stripMimeParameters`: cut at the first `;`, space or tab.
fn strip_mime_parameters(raw: &str) -> &str {
    match raw.find(|c| c == ';' || c == ' ' || c == '\t') {
        Some(i) => &raw[..i],
        None => raw,
    }
}

/// libzim's `isCompressibleMimetype`.
fn is_compressible_mimetype(mime: &str) -> bool {
    mime.starts_with("text")
        || mime.contains("+xml")
        || mime.contains("+json")
        || mime == "application/javascript"
        || mime == "application/json"
}

impl ZimCreator {
    /// Start a new archive that will be written to `out_path` by [`finish`].
    pub fn new(out_path: impl Into<PathBuf>) -> io::Result<Self> {
        Ok(ZimCreator {
            out_path: out_path.into(),
            uuid: random_uuid(),
            dirents: BTreeMap::new(),
            mime_by_idx: Vec::new(),
            mime_insertion: Vec::new(),
            main_path: String::new(),
            main_target: None,
            listing_bytes: Vec::new(),
            comp_cluster: OpenCluster::new(true),
            uncomp_cluster: OpenCluster::new(false),
            comp_generation: 0,
            uncomp_generation: 0,
            comp_gen_idx: Vec::new(),
            uncomp_gen_idx: Vec::new(),
            mime_counter: BTreeMap::new(),
            w: None,
        })
    }

    /// Set the main entry path. May be called at any point before [`finish`];
    /// if `C/<main_path>` exists when finishing, a `W/mainPage` redirection is
    /// created and the header's mainPage field set to its index.
    pub fn set_main_path(&mut self, main_path: &str) {
        self.main_path = main_path.to_string();
    }

    /// libzim's `getMimeTypeIdx`: assign insertion-order ids to MIME strings.
    fn get_mime_idx(&mut self, mime: &str) -> io::Result<u16> {
        if let Some(i) = self.mime_by_idx.iter().position(|m| m == mime) {
            return Ok(i as u16);
        }
        let idx = self
            .mime_by_idx
            .len()
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many distinct mime types"))?;
        if idx >= u16::MAX {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "too many distinct mime types"));
        }
        self.mime_by_idx.push(mime.to_string());
        Ok(idx)
    }

    /// libzim's `addOrUpdate`: replace a placeholder in place, error on a real
    /// dirent conflict, insert otherwise.
    fn add_or_update(&mut self, ns: u8, path: &str, data: Dirent) -> io::Result<()> {
        let key = (ns, path.to_string());
        match self.dirents.get(&key) {
            Some(existing) if !matches!(data.kind, DirentKind::Placeholder) => {
                if !matches!(existing.kind, DirentKind::Placeholder) {
                    return Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        format!(
                            "Impossible to add {}/{}: a dirent with that path already exists",
                            ns as char, path
                        ),
                    ));
                }
            }
            _ => {}
        }
        self.dirents.insert(key, data);
        Ok(())
    }

    /// libzim's `ensureDirentCanBeAdded`: fail early on a dirent conflict.
    fn ensure_dirent_can_be_added(&self, ns: u8, path: &str) -> io::Result<()> {
        match self.dirents.get(&(ns, path.to_string())) {
            Some(existing) if !matches!(existing.kind, DirentKind::Placeholder) => Err(
                io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!(
                        "Impossible to add {}/{}: a dirent with that path already exists",
                        ns as char, path
                    ),
                ),
            ),
            _ => Ok(()),
        }
    }

    /// libzim's `addItemData`: pick the open cluster for `compress`, closing it
    /// first when it already holds blobs and would grow past the 2 MiB target,
    /// then append the blob. Returns the blob's location: which kind of open
    /// cluster, which generation of it, and the blob index within the cluster.
    fn add_item_data(&mut self, compress: bool, content: &[u8]) -> io::Result<(bool, u32, u32)> {
        let item_size = content.len() as u64;
        if compress {
            if self.comp_cluster.count() > 0
                && self.comp_cluster.size() + item_size >= CLUSTER_TARGET_SIZE
            {
                self.close_cluster(Slot::Compressed)?;
            }
            let blob = self.comp_cluster.count();
            self.comp_cluster.push(content);
            Ok((true, self.comp_generation, blob))
        } else {
            if self.uncomp_cluster.count() > 0
                && self.uncomp_cluster.size() + item_size >= CLUSTER_TARGET_SIZE
            {
                self.close_cluster(Slot::Uncompressed)?;
            }
            let blob = self.uncomp_cluster.count();
            self.uncomp_cluster.push(content);
            Ok((false, self.uncomp_generation, blob))
        }
    }

    /// Stream a blob of known `len` from `reader` into the uncompressed open
    /// cluster (used for embedded Xapian databases, which can be large).
    /// Blobs at or above [`BIG_BLOB_THRESHOLD`] stream straight to disk as
    /// their own uncompressed cluster instead of sitting in RAM until finish.
    fn add_item_streaming(&mut self, len: u64, reader: &mut dyn Read) -> io::Result<(bool, u32, u32)> {
        if len >= BIG_BLOB_THRESHOLD {
            self.close_cluster(Slot::Uncompressed)?;
            self.ensure_open()?;
            let extended = len > u32::MAX as u64;
            let width: u64 = if extended { 8 } else { 4 };
            // One blob: the offset table holds the 0 start and the end.
            let table: Vec<u8> = [0u64, len]
                .iter()
                .flat_map(|&off| {
                    let v = off + 2 * width;
                    if extended { v.to_le_bytes().to_vec() } else { (v as u32).to_le_bytes().to_vec() }
                })
                .collect();
            let ws = self.w.as_mut().unwrap();
            let offset = ws.pos;
            let archive_idx = ws.cluster_offsets.len() as u32;
            ws.out.write_all(&[CLUSTER_UNCOMPRESSED])?;
            ws.out.write_all(&table)?;
            let mut copied = 0u64;
            let mut buf = vec![0u8; 1024 * 1024];
            while copied < len {
                let want = (len - copied).min(buf.len() as u64) as usize;
                let n = reader.read(&mut buf[..want])?;
                if n == 0 {
                    break;
                }
                copied += n as u64;
                ws.out.write_all(&buf[..n])?;
            }
            if copied != len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("xapian index shorter than expected ({copied} of {len} bytes)"),
                ));
            }
            ws.pos += 1 + table.len() as u64 + len;
            ws.cluster_offsets.push(offset);
            let generation = self.uncomp_generation;
            self.uncomp_gen_idx.push(archive_idx);
            self.uncomp_generation += 1;
            return Ok((false, generation, 0));
        }
        if self.uncomp_cluster.count() > 0
            && self.uncomp_cluster.size() + len >= CLUSTER_TARGET_SIZE
        {
            self.close_cluster(Slot::Uncompressed)?;
        }
        let blob = self.uncomp_cluster.count();
        self.uncomp_cluster.blob_ends.push(self.uncomp_cluster.data_size() + len);
        let mut copied = 0u64;
        let mut buf = vec![0u8; 1024 * 1024];
        let data = &mut self.uncomp_cluster.data;
        while copied < len {
            let want = (len - copied).min(buf.len() as u64) as usize;
            let n = reader.read(&mut buf[..want])?;
            if n == 0 {
                break;
            }
            copied += n as u64;
            data.extend_from_slice(&buf[..n]);
        }
        if copied != len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("xapian index shorter than expected ({copied} of {len} bytes)"),
            ));
        }
        Ok((false, self.uncomp_generation, blob))
    }

    /// The final archive cluster index of a blob. Closed generations map
    /// through the generation tables; the two open clusters close at the end
    /// of the file write (compressed first, a no-op when empty), so an open
    /// generation's index is the next free position.
    fn resolve_blob(&self, blob: BlobRef, closed: usize) -> io::Result<u32> {
        match blob.compress {
            true if (blob.generation as usize) < self.comp_gen_idx.len() => {
                Ok(self.comp_gen_idx[blob.generation as usize])
            }
            true if blob.generation == self.comp_generation => Ok(closed as u32),
            false if (blob.generation as usize) < self.uncomp_gen_idx.len() => {
                Ok(self.uncomp_gen_idx[blob.generation as usize])
            }
            false if blob.generation == self.uncomp_generation => {
                Ok(closed as u32 + u32::from(self.comp_cluster.count() > 0))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "blob references an unknown cluster generation",
            )),
        }
    }

    /// Open the output file and write the 80-byte header placeholder plus
    /// the sparse MIME-list gap (the real MIME list is patched in at the
    /// end). Idempotent; also used by the record path.
    fn ensure_open(&mut self) -> io::Result<()> {
        if self.w.is_some() {
            return Ok(());
        }
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.out_path)?;
        let mut out = io::BufWriter::new(f);
        out.write_all(&[0u8; HEADER_SIZE as usize])?;
        out.write_all(&vec![0u8; (CLUSTER_BASE_OFFSET - HEADER_SIZE) as usize])?;
        self.w = Some(WriteState {
            out,
            pos: CLUSTER_BASE_OFFSET,
            cluster_offsets: Vec::new(),
            dirent_offsets: Vec::new(),
            dirent_start: None,
            dirent_end: CLUSTER_BASE_OFFSET,
            tails: Vec::new(),
            tail_next: 0,
            main_page_idx: None,
        });
        Ok(())
    }

    /// Close a cluster (it must hold at least one blob) and open a fresh
    /// one: its bytes stream straight to the output file (clusters never
    /// accumulate in RAM), which keeps the byte layout identical to writing
    /// them all at finish — same close order, same offsets.
    fn close_cluster(&mut self, slot: Slot) -> io::Result<()> {
        let count = match slot {
            Slot::Compressed => self.comp_cluster.count(),
            Slot::Uncompressed => self.uncomp_cluster.count(),
        };
        if count == 0 {
            return Ok(());
        }
        let archive_idx = self
            .w
            .as_ref()
            .map(|w| w.cluster_offsets.len())
            .unwrap_or(0) as u32;
        let cluster = match slot {
            Slot::Compressed => {
                self.comp_gen_idx.push(archive_idx);
                self.comp_generation += 1;
                std::mem::replace(&mut self.comp_cluster, OpenCluster::new(true))
            }
            Slot::Uncompressed => {
                self.uncomp_gen_idx.push(archive_idx);
                self.uncomp_generation += 1;
                std::mem::replace(&mut self.uncomp_cluster, OpenCluster::new(false))
            }
        };
        self.ensure_open()?;
        let ws = self.w.as_mut().unwrap();
        let offset = ws.pos;
        let size = cluster.write_to(&mut ws.out)?;
        ws.cluster_offsets.push(offset);
        ws.pos += size;
        Ok(())
    }

    /// Sort the MIME list (libzim's resolveMimeTypes). Dirents registered
    /// before the freeze carry insertion-order ids: `mime_insertion` maps
    /// them back to their string, whose sorted position is the final id.
    fn freeze_mimes(&mut self) {
        self.mime_insertion = std::mem::take(&mut self.mime_by_idx);
        self.mime_by_idx = self.mime_insertion.clone();
        self.mime_by_idx.sort();
    }

    /// The MIME id of an already-registered string (its sorted position).
    fn mime_idx_of(&self, mime: &str) -> io::Result<u16> {
        self.mime_by_idx
            .binary_search_by(|m| m.as_str().cmp(mime))
            .map(|i| i as u16)
            .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("mime type {mime:?} not registered before the archive was written"),
            )
        })
    }

    /// The insertion-order id of a registered string (tails registered after
    /// the freeze carry insertion-order ids; emission re-derives the sorted
    /// id from the string).
    fn mime_insertion_idx(&self, mime: &str) -> io::Result<u16> {
        self.mime_insertion
            .iter()
            .position(|m| m == mime)
            .map(|i| i as u16)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("mime type {mime:?} not registered before the archive was written"),
                )
            })
    }

    /// Add a content item (libzim's `Creator::addItem`). An empty `mime` gets
    /// libzim's "application/octet-stream" fallback. `front_article` is the
    /// FRONT_ARTICLE hint: only front articles appear in the
    /// `X/listing/titleOrdered/v1` listing.
    pub fn add_item(
        &mut self,
        path: &str,
        title: &str,
        mime: &str,
        compress: bool,
        front_article: bool,
        content: Vec<u8>,
    ) -> io::Result<()> {
        self.add_item_in_namespace(b'C', path, title, mime, compress, front_article, content)
    }

    /// [`add_item`] in an explicit namespace: the reader's `Entry` and the
    /// writer's dirent key both pair a namespace byte with a path, so the
    /// generic form mirrors [`crate::zim::Entry`] one-to-one (used by the
    /// synthetic test archives, which carry `M/` metadata and old-scheme
    /// namespace entries). libzim's user-facing `addItem` is C-only; the
    /// `M/Counter` tally therefore counts C-namespace items only.
    pub(crate) fn add_item_in_namespace(
        &mut self,
        ns: u8,
        path: &str,
        title: &str,
        mime: &str,
        compress: bool,
        front_article: bool,
        content: Vec<u8>,
    ) -> io::Result<()> {
        let mut dirent_mime = mime;
        if dirent_mime.is_empty() {
            eprintln!("WARNING: mimetype missing for {path}");
            dirent_mime = "application/octet-stream";
        }
        let mime_idx = self.get_mime_idx(dirent_mime)?;
        // libzim's ensureDirentCanBeAdded: a real dirent over a real dirent is
        // an error, before any blob lands in a cluster.
        self.ensure_dirent_can_be_added(ns, path)?;
        let blob = self.add_item_data(compress, &content)?;
        // libzim's CounterHandler::handle(dirent, item): C-namespace items
        // counted by their parameter-stripped mime; empty mimes skipped.
        if ns == b'C' {
            let clean = strip_mime_parameters(mime);
            if !mime.is_empty() && !clean.is_empty() {
                *self.mime_counter.entry(clean.to_string()).or_insert(0) += 1;
            }
        }
        self.add_or_update(
            ns,
            path,
            Dirent {
                kind: DirentKind::Item {
                    mime_idx,
                    blob: BlobRef {
                        compress,
                        generation: blob.1,
                        blob: blob.2,
                    },
                },
                title: title.to_string(),
                front_article,
                idx: 0,
                removed: false,
            },
        )
    }

    /// Add one blob to the appropriate open cluster WITHOUT creating a
    /// dirent: the streaming convert path assigns dirents later (phase 3),
    /// carrying the returned ref in its per-article records.
    pub fn add_blob(&mut self, compress: bool, content: &[u8]) -> io::Result<BlobRef> {
        let (compress, generation, blob) = self.add_item_data(compress, content)?;
        Ok(BlobRef { compress, generation, blob })
    }

    /// Streaming mode: set the already-resolved target entry index of the
    /// `W/mainPage` redirect (the caller ranks members; `None` drops the
    /// main-page dirent, e.g. when nothing was converted).
    pub fn set_main_page_target(&mut self, target_idx: Option<u32>) {
        self.main_target = target_idx;
    }

    /// Streaming mode: hand over the caller-built `X/listing/titleOrdered/v1`
    /// bytes (added to the uncompressed cluster at `finish_write`).
    pub fn set_listing_bytes(&mut self, bytes: Vec<u8>) {
        self.listing_bytes = bytes;
    }

    /// Register a MIME string early so the list can be finalized (sorted)
    /// before any dirent is streamed with it.
    pub fn register_mime(&mut self, mime: &str) -> io::Result<()> {
        self.get_mime_idx(mime).map(|_| ())
    }

    /// Streaming mode: freeze the MIME list (everything the archive will use
    /// is registered by now — the counter and listing mimes are added here),
    /// open the output file and park the tail dirents registered so far.
    /// Dirents are then emitted with [`ZimCreator::emit_dirent`];
    /// [`ZimCreator::finish_write`] completes the file.
    pub fn begin_write(&mut self) -> io::Result<()> {
        self.get_mime_idx("text/plain")?;
        self.get_mime_idx(LISTING_MIME)?;
        self.freeze_mimes();
        self.ensure_open()?;
        let tails: Vec<((u8, String), Dirent)> =
            std::mem::take(&mut self.dirents).into_iter().collect();
        let ws = self.w.as_mut().unwrap();
        ws.tails = tails;
        Ok(())
    }

    /// Streaming mode: emit one dirent into the archive body. C-namespace
    /// items are tallied for M/Counter (libzim counts them by their
    /// parameter-stripped mime). Tail dirents sorting before this one are
    /// flushed first (the C stream always sorts before M/W/X in practice).
    pub fn emit_dirent(&mut self, d: DirentOut) -> io::Result<()> {
        if let DirentOut::Item { ns, mime, .. } = &d {
            if *ns == b'C' {
                let clean = strip_mime_parameters(mime);
                if !mime.is_empty() && !clean.is_empty() {
                    *self.mime_counter.entry(clean.to_string()).or_insert(0) += 1;
                }
            }
        }
        let (ns, path) = match &d {
            DirentOut::Item { ns, path, .. } | DirentOut::Redirect { ns, path, .. } => (*ns, path),
        };
        self.flush_tails_before(ns, path)?;
        self.emit_resolved(d)
    }

    /// Emit one dirent without the counter tally (the record path tallied
    /// through `add_item` already).
    fn emit_resolved(&mut self, d: DirentOut) -> io::Result<()> {
        self.ensure_open()?;
        match d {
            DirentOut::Item { ns, path, title, mime, blob } => {
                let closed = self.w.as_ref().map(|w| w.cluster_offsets.len()).unwrap_or(0);
                let kind = DirentBytes::Item {
                    mime_idx: self.mime_idx_of(&mime)?,
                    cluster: self.resolve_blob(blob, closed)?,
                    blob: blob.blob,
                };
                let ws = self.w.as_mut().unwrap();
                ws.emit(ns, &path, &title, &kind)
            }
            DirentOut::Redirect { ns, path, title, target_idx } => {
                let ws = self.w.as_mut().unwrap();
                ws.emit(ns, &path, &title, &DirentBytes::Redirect { target: target_idx })
            }
        }
    }

    /// Flush the pending tail dirents sorting before `(ns, path)`.
    fn flush_tails_before(&mut self, ns: u8, path: &str) -> io::Result<()> {
        loop {
            let at = {
                let ws = self.w.as_ref().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "begin_write first")
                })?;
                match ws.tails.get(ws.tail_next) {
                    Some((key, _)) if key.0 < ns || (key.0 == ns && key.1.as_str() < path) => {
                        ws.tail_next
                    }
                    _ => return Ok(()),
                }
            };
            let (key, d) = {
                let ws = self.w.as_ref().unwrap();
                (ws.tails[at].0.clone(), ws.tails[at].1.clone())
            };
            self.emit_tail_dirent(&key, &d)?;
        }
    }

    /// Resolve and emit one tail dirent (a pre-registered metadata,
    /// illustration or Xapian-index dirent).
    fn emit_tail_dirent(&mut self, key: &(u8, String), d: &Dirent) -> io::Result<()> {
        let kind = match &d.kind {
            DirentKind::Item { mime_idx, blob } => {
                let mime = self
                    .mime_insertion
                    .get(*mime_idx as usize)
                    .cloned()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "tail dirent mime out of range")
                    })?;
                let closed = self.w.as_ref().map(|w| w.cluster_offsets.len()).unwrap_or(0);
                DirentBytes::Item {
                    mime_idx: self.mime_idx_of(&mime)?,
                    cluster: self.resolve_blob(*blob, closed)?,
                    blob: blob.blob,
                }
            }
            DirentKind::Redirect { .. } | DirentKind::Placeholder => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unresolved tail redirect {key:?} in the streaming path"),
                ));
            }
            DirentKind::RedirectIdx(t) => DirentBytes::Redirect { target: *t },
        };
        self.ensure_open()?;
        let ws = self.w.as_mut().unwrap();
        ws.emit(key.0, &key.1, &d.title, &kind)
    }

    /// Streaming mode: add the M/Counter, X/listing and W/mainPage tail
    /// dirents (their blobs too — counter content first, then the listing,
    /// the content order libzim's finish produces), emit every remaining
    /// tail dirent and complete the file (pointer tables, header, checksum).
    pub fn finish_write(&mut self) -> io::Result<()> {
        // M/Counter content: "mime=count;..." over the C-item tally,
        // COMPRESSED (text/plain), like the record path's finish.
        let counter_content = self
            .mime_counter
            .iter()
            .map(|(m, c)| format!("{m}={c}"))
            .collect::<Vec<_>>()
            .join(";");
        let (compress, generation, blob) = self.add_item_data(true, counter_content.as_bytes())?;
        // Tails registered after the freeze still carry insertion-order ids;
        // emission re-derives the sorted id from the string.
        let counter_mime_idx = self.mime_insertion_idx("text/plain")?;
        let counter = ((b'M', "Counter".to_string()), Dirent {
            kind: DirentKind::Item {
                mime_idx: counter_mime_idx,
                blob: BlobRef { compress, generation, blob },
            },
            title: String::new(),
            front_article: false,
            idx: 0,
            removed: false,
        });

        // X/listing/titleOrdered/v1 content: the caller-built u32 LE entry
        // indexes, UNCOMPRESSED.
        let listing = std::mem::take(&mut self.listing_bytes);
        let (compress, generation, blob) = self.add_item_data(false, &listing)?;
        let listing_mime_idx = self.mime_insertion_idx(LISTING_MIME)?;
        let listing = ((b'X', "listing/titleOrdered/v1".to_string()), Dirent {
            kind: DirentKind::Item {
                mime_idx: listing_mime_idx,
                blob: BlobRef { compress, generation, blob },
            },
            title: String::new(),
            front_article: false,
            idx: 0,
            removed: false,
        });

        // W/mainPage when the caller resolved a target.
        let main_page = self
            .main_target
            .map(|t| ((b'W', "mainPage".to_string()), Dirent {
                kind: DirentKind::RedirectIdx(t),
                title: String::new(),
                front_article: false,
                idx: 0,
                removed: false,
            }));

        {
            let ws = self.w.as_mut().unwrap();
            ws.tails.push(counter);
            ws.tails.push(listing);
            if let Some(mp) = main_page {
                ws.tails.push(mp);
            }
            ws.tails.sort_by(|a, b| a.0.cmp(&b.0));
        }
        self.flush_all_tails()?;
        self.finish_tail()
    }

    /// Emit every remaining tail dirent (sorted; the counter/listing
    /// main-page ones merged in by `finish_write`).
    fn flush_all_tails(&mut self) -> io::Result<()> {
        loop {
            let more = {
                let ws = self.w.as_ref().unwrap();
                ws.tail_next < ws.tails.len()
            };
            if !more {
                return Ok(());
            }
            let (key, d) = {
                let ws = self.w.as_ref().unwrap();
                (ws.tails[ws.tail_next].0.clone(), ws.tails[ws.tail_next].1.clone())
            };
            self.w.as_mut().unwrap().tail_next += 1;
            self.emit_tail_dirent(&key, &d)?;
        }
    }

    /// Complete the streamed file: patch the MIME list into the header gap,
    /// close the two open clusters (compressed first, like libzim), then
    /// write the path pointer table, cluster pointer table, header and MD5
    /// checksum.
    fn finish_tail(&mut self) -> io::Result<()> {
        // MIME list at offset 80 (the area was written as zeros).
        let mut mime_blob = Vec::new();
        for mime in &self.mime_by_idx {
            mime_blob.extend_from_slice(mime.as_bytes());
            mime_blob.push(0);
        }
        mime_blob.push(0);
        assert!(
            HEADER_SIZE + mime_blob.len() as u64 <= CLUSTER_BASE_OFFSET,
            "mime type list too big"
        );
        {
            let ws = self.w.as_mut().unwrap();
            ws.out.flush()?;
            ws.out.seek(SeekFrom::Start(HEADER_SIZE))?;
            ws.out.write_all(&mime_blob)?;
            let pad = CLUSTER_BASE_OFFSET - HEADER_SIZE - mime_blob.len() as u64;
            ws.out.write_all(&vec![0u8; pad as usize])?;
            ws.out.seek(SeekFrom::Start(ws.pos))?;
        }
        self.close_cluster(Slot::Compressed)?;
        self.close_cluster(Slot::Uncompressed)?;
        let ws = self.w.as_mut().unwrap();
        let dirent_start = ws.dirent_start.unwrap_or(ws.pos);
        let path_ptr_pos = ws.dirent_end;
        // The path pointer table follows the dirent section; the cluster
        // pointer table follows it. `at` tracks the true append position
        // (ws.pos is not advanced by these writes).
        let mut at = ws.pos;
        for &off in &ws.dirent_offsets {
            ws.out.write_all(&(dirent_start + off).to_le_bytes())?;
            at += 8;
        }
        let cluster_ptr_pos = at;
        for &off in &ws.cluster_offsets {
            ws.out.write_all(&off.to_le_bytes())?;
            at += 8;
        }
        let checksum_pos = at;
        let header = ZimHeader {
            major: ZIM_MAJOR,
            minor: ZIM_MINOR_VERSION,
            uuid: self.uuid,
            entry_count: ws.dirent_offsets.len() as u32,
            cluster_count: ws.cluster_offsets.len() as u32,
            url_ptr_pos: path_ptr_pos,
            title_ptr_pos: NO_TITLE_PTR_POS,
            cluster_ptr_pos,
            mime_list_pos: HEADER_SIZE,
            main_page: ws.main_page_idx.unwrap_or(u32::MAX),
            layout_page: NO_LAYOUT_PAGE,
            checksum_pos,
        }
        .serialize();
        ws.out.flush()?;
        ws.out.seek(SeekFrom::Start(0))?;
        ws.out.write_all(&header)?;
        // MD5 over everything before checksumPos (libzim's writeChecksum):
        // re-read the flushed file, then append the digest.
        ws.out.flush()?;
        let mut rf = std::fs::File::open(&self.out_path)?;
        let mut hasher = md5::Md5::new();
        let mut left = checksum_pos;
        let mut buf = vec![0u8; 65536];
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            rf.read_exact(&mut buf[..want])?;
            hasher.update(&buf[..want]);
            left -= want as u64;
        }
        let digest: [u8; 16] = hasher.finalize().into();
        ws.out.seek(SeekFrom::Start(checksum_pos))?;
        ws.out.write_all(&digest)?;
        ws.out.flush()?;
        Ok(())
    }

    /// Add a redirection. If `target_path` was not added yet, a placeholder
    /// dirent is created for it and becomes a real item when `add_item` is
    /// later called with that path.
    pub fn add_redirection(
        &mut self,
        path: &str,
        title: &str,
        target_path: &str,
        front_article: bool,
    ) -> io::Result<()> {
        // libzim: addOrUpdate(Dirent(NS::C, targetPath)) — creates the target
        // placeholder only when no dirent exists there yet.
        let target_key = (b'C', target_path.to_string());
        if !self.dirents.contains_key(&target_key) {
            self.dirents.insert(
                target_key,
                Dirent { kind: DirentKind::Placeholder, title: String::new(), front_article: false, idx: 0, removed: false },
            );
        }
        self.add_or_update(
            b'C',
            path,
            Dirent {
                kind: DirentKind::Redirect { target: (b'C', target_path.to_string()) },
                title: title.to_string(),
                front_article,
                idx: 0,
                removed: false,
            },
        )
    }

    /// Add an `M/<name>` metadata entry (libzim's `Creator::addMetadata`).
    /// The MIME type decides compression via libzim's `isCompressibleMimetype`
    /// ("text..."-prefix, "+xml"/"+json" substrings, javascript/json).
    pub fn add_metadata(&mut self, name: &str, content: &[u8], mime: &str) -> io::Result<()> {
        let mime_idx = self.get_mime_idx(mime)?;
        self.ensure_dirent_can_be_added(b'M', name)?;
        let compress = is_compressible_mimetype(mime);
        let blob = self.add_item_data(compress, content)?;
        self.add_or_update(
            b'M',
            name,
            Dirent {
                kind: DirentKind::Item {
                    mime_idx,
                    blob: BlobRef { compress, generation: blob.1, blob: blob.2 },
                },
                title: String::new(),
                front_article: false,
                idx: 0,
                removed: false,
            },
        )
    }

    /// Add an illustration (`M/Illustration_{size}x{size}@1` metadata with the
    /// image/png mimetype — never compressed, like libzim's addIllustration).
    pub fn add_illustration(&mut self, size: u32, content: &[u8]) -> io::Result<()> {
        let name = format!("Illustration_{size}x{size}@1");
        self.add_metadata(&name, content, "image/png")
    }

    /// Add an embedded Xapian index as `X/<name>` with the uncompressed blob
    /// holding the single-file database (libzim stores fulltext/title indexes
    /// UNCOMPRESSED, `application/octet-stream+xapian`).
    pub fn add_xapian_index(&mut self, name: &str, single_file_db: &Path) -> io::Result<()> {
        // Stream the DB file into the uncompressed open cluster in bounded
        // chunks (embedded indexes can be large).
        let mut file = std::fs::File::open(single_file_db).map_err(|e| {
            io::Error::new(e.kind(), format!("cannot open xapian index {single_file_db:?}: {e}"))
        })?;
        let len = file.metadata()?.len();
        self.add_xapian_blob(name, len, &mut file)
    }

    /// [`add_xapian_index`] from in-memory bytes (the synthetic test
    /// archives embed hand-built databases).
    #[cfg(test)]
    pub fn add_xapian_index_bytes(&mut self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let mut cursor = std::io::Cursor::new(bytes);
        self.add_xapian_blob(name, bytes.len() as u64, &mut cursor)
    }

    /// Shared tail of the two `add_xapian_*` variants: stream `len` bytes
    /// from `reader` into the uncompressed open cluster and file the
    /// `X/<name>` dirent.
    fn add_xapian_blob(&mut self, name: &str, len: u64, reader: &mut dyn Read) -> io::Result<()> {
        let mime_idx = self.get_mime_idx("application/octet-stream+xapian")?;
        self.ensure_dirent_can_be_added(b'X', name)?;
        let (compress, generation, blob) = self.add_item_streaming(len, reader)?;
        self.add_or_update(
            b'X',
            name,
            Dirent {
                kind: DirentKind::Item {
                    mime_idx,
                    blob: BlobRef { compress, generation, blob },
                },
                title: String::new(),
                front_article: false,
                idx: 0,
                removed: false,
            },
        )
    }

    /// Finish the archive (libzim's `finishZimCreation` + `writeLastParts`):
    /// create the listing/main-page/counter dirents, clean up dangling
    /// redirects, assign entry indexes, sort the MIME list, append the
    /// M/Counter and title-listing blobs, close the open clusters, write the
    /// file and its checksum.
    pub fn finish(&mut self) -> io::Result<()> {
        // 1. The title listing entry ("application/octet-stream+zimlisting"),
        //    always created (its blob may be empty).
        let listing_mime = self.get_mime_idx(LISTING_MIME)?;
        self.dirents.insert(
            (b'X', "listing/titleOrdered/v1".to_string()),
            Dirent {
                kind: DirentKind::Item {
                    mime_idx: listing_mime,
                    blob: BlobRef { compress: false, generation: 0, blob: 0 },
                },
                title: String::new(),
                front_article: false,
                idx: 0,
                removed: false,
            },
        );

        // 2. W/mainPage redirection when a main path is set and C/<main_path>
        //    exists (even as a placeholder, like libzim's findDirent lookup;
        //    the cleanup below drops it again in that case).
        if !self.main_path.is_empty() && self.dirents.contains_key(&(b'C', self.main_path.clone()))
        {
            self.dirents.insert(
                (b'W', "mainPage".to_string()),
                Dirent {
                    kind: DirentKind::Redirect { target: (b'C', self.main_path.clone()) },
                    title: String::new(),
                    front_article: false,
                    idx: 0,
                    removed: false,
                },
            );
        }

        // 3. M/Counter dirent, created unconditionally (libzim's
        //    CounterHandler::createDirents; mime "text/plain" -> compressed).
        let counter_mime = self.get_mime_idx("text/plain")?;
        self.dirents.insert(
            (b'M', "Counter".to_string()),
            Dirent {
                kind: DirentKind::Item {
                    mime_idx: counter_mime,
                    blob: BlobRef { compress: true, generation: 0, blob: 0 },
                },
                title: String::new(),
                front_article: false,
                idx: 0,
                removed: false,
            },
        );

        // 4. Fix the dirents before any data or index is assigned.
        self.detect_dangling_redirects();
        self.remove_loops_and_blind_chains();
        self.drop_removed_redirects();

        // Entry indexes: position in sorted (namespace, path) order.
        for (i, (_, d)) in self.dirents.iter_mut().enumerate() {
            d.idx = i as u32;
        }

        // 5. MIME list resolved: SORTED alphabetically (libzim's
        //    resolveMimeTypes). Dirent ids resolve by string lookup when the
        //    dirents are streamed out below.
        self.freeze_mimes();

        // 6. M/Counter content ("mime=count;..."), COMPRESSED (text/plain).
        let counter_content = self
            .mime_counter
            .iter()
            .map(|(m, c)| format!("{m}={c}"))
            .collect::<Vec<_>>()
            .join(";");
        let (compress, generation, blob_num) = self.add_item_data(true, counter_content.as_bytes())?;
        if let Some(Dirent { kind: DirentKind::Item { blob, .. }, .. }) =
            self.dirents.get_mut(&(b'M', "Counter".to_string()))
        {
            *blob = BlobRef { compress, generation, blob: blob_num };
        }

        // 7. X/listing/titleOrdered/v1 content: u32 LE entry index of every
        // front-article dirent, sorted by (title bytes, path bytes) — libzim
        // std::sorts by title only; the tie-break keeps us deterministic.
        // UNCOMPRESSED cluster.
        let mut articles: Vec<(&str, &str, u32)> = Vec::new();
        for ((_, path), d) in &self.dirents {
            if d.front_article {
                articles.push((d.effective_title(path), path, d.idx));
            }
        }
        articles.sort_by(|a, b| (a.0.as_bytes(), a.1.as_bytes()).cmp(&(b.0.as_bytes(), b.1.as_bytes())));
        let mut listing_blob = Vec::with_capacity(articles.len() * 4);
        for (_, _, idx) in &articles {
            listing_blob.extend_from_slice(&idx.to_le_bytes());
        }
        let lb = self.add_item_data(false, &listing_blob)?;
        if let Some(Dirent { kind: DirentKind::Item { blob, .. }, .. }) = self
            .dirents
            .get_mut(&(b'X', "listing/titleOrdered/v1".to_string()))
        {
            *blob = BlobRef { compress: lb.0, generation: lb.1, blob: lb.2 };
        }

        // 8. Close the open clusters (compressed first, then the uncompressed
        // one — libzim's order; they stream to the file), then stream every
        // dirent out in sorted order and complete the file.
        self.close_cluster(Slot::Compressed)?;
        self.close_cluster(Slot::Uncompressed)?;

        // Target entry indexes: post-cleanup map position (sorted order).
        let idx_of: HashMap<&(u8, String), u32> =
            self.dirents.keys().enumerate().map(|(i, k)| (k, i as u32)).collect();
        let mut items = Vec::with_capacity(self.dirents.len());
        for ((ns, path), d) in &self.dirents {
            items.push(match &d.kind {
                DirentKind::Item { mime_idx, blob } => {
                    let mime = self.mime_insertion[*mime_idx as usize].clone();
                    DirentOut::Item {
                        ns: *ns,
                        path: path.clone(),
                        title: d.title.clone(),
                        mime,
                        blob: *blob,
                    }
                }
                DirentKind::Redirect { target } => DirentOut::Redirect {
                    ns: *ns,
                    path: path.clone(),
                    title: d.title.clone(),
                    target_idx: *idx_of.get(target).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("dangling redirect {}/{path} remains", *ns as char),
                        )
                    })?,
                },
                DirentKind::RedirectIdx(target) => DirentOut::Redirect {
                    ns: *ns,
                    path: path.clone(),
                    title: d.title.clone(),
                    target_idx: *target,
                },
                DirentKind::Placeholder => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unfilled placeholder {}/{path} remains", *ns as char),
                    ));
                }
            });
        }
        self.write_stream(items.into_iter())
    }

    /// Shared file tail: stream `dirents` (already in final sorted order,
    /// everything resolved) into the output, then the pointer tables,
    /// header and checksum. Serves the record path (materialized dirents)
    /// and — piecewise — the streaming path.
    fn write_stream(&mut self, dirents: impl Iterator<Item = DirentOut>) -> io::Result<()> {
        self.ensure_open()?;
        for d in dirents {
            self.emit_resolved(d)?;
        }
        self.finish_tail()
    }

    /// libzim's `detectDanglingRedirects`: a redirection whose target is still
    /// a placeholder (never filled in) is invalid — both it and the
    /// placeholder are marked removed.
    fn detect_dangling_redirects(&mut self) {
        let keys: Vec<(u8, String)> = self.dirents.keys().cloned().collect();
        let mut dangling: Vec<(u8, String)> = Vec::new();
        for key in &keys {
            match &self.dirents[key].kind {
                DirentKind::Redirect { target } => {
                    if let Some(t) = self.dirents.get(target) {
                        if matches!(t.kind, DirentKind::Placeholder) {
                            dangling.push(key.clone());
                            dangling.push(target.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        for key in dangling {
            if let Some(d) = self.dirents.get_mut(&key) {
                d.removed = true;
            }
        }
    }

    /// libzim's `removeLoopsAndBlindChainsOfRedirects`: walk each unresolved
    /// redirect chain assigning temporary indexes; a walk that hits a dead end
    /// (placeholder or removed dirent) or a loop marks the WHOLE chain for
    /// removal (markBlindRedirectChainForRemoval).
    fn remove_loops_and_blind_chains(&mut self) {
        let keys: Vec<(u8, String)> = self.dirents.keys().cloned().collect();
        let mut chain_idx: HashMap<(u8, String), u32> = HashMap::new();
        let mut index: u32 = 1;
        for key in &keys {
            let d = &self.dirents[key];
            if d.removed || !d.is_redirect() || chain_idx.get(key).copied().unwrap_or(0) != 0 {
                continue;
            }
            let start = index;
            let mut cur = key.clone();
            let mut dead = false;
            loop {
                chain_idx.insert(cur.clone(), index);
                index += 1;
                let next = match &self.dirents[&cur].kind {
                    DirentKind::Redirect { target } => Some(target.clone()),
                    // A stream-mode main page never takes part in the cleanup.
                    DirentKind::RedirectIdx(_) => break,
                    DirentKind::Placeholder => None,
                    DirentKind::Item { .. } => break,
                };
                let next = match next {
                    Some(n) => n,
                    None => {
                        dead = true;
                        break;
                    }
                };
                let seen = chain_idx.get(&next).copied().unwrap_or(0);
                if self.dirents.get(&next).map(|d| d.removed).unwrap_or(false) || seen >= start {
                    dead = true;
                    break;
                }
                cur = next;
            }
            if dead {
                // markBlindRedirectChainForRemoval: walk the chain, dropping
                // every live dirent until a removed one is met.
                let mut cur = key.clone();
                loop {
                    match self.dirents.get_mut(&cur) {
                        Some(d) if !d.removed => d.removed = true,
                        _ => break,
                    }
                    cur = match &self.dirents[&cur].kind {
                        DirentKind::Redirect { target } => target.clone(),
                        _ => break,
                    };
                }
            }
        }
    }

    /// Remove all dirents marked by the cleanup phases.
    fn drop_removed_redirects(&mut self) {
        self.dirents.retain(|_, d| !d.removed);
    }
}

/// Serialize one dirent (libzim's `Dirent::write`): the 12/16-byte head, then
/// the path (NUL-terminated), then the title — omitted when equal to the
/// path (libzim's PathTitleTinyString::concat) — and a final NUL.
fn write_dirent(
    out: &mut impl Write,
    ns: u8,
    path: &str,
    title: &str,
    k: &DirentBytes,
) -> io::Result<()> {
    match k {
        DirentBytes::Item { mime_idx, cluster, blob } => {
            out.write_all(&mime_idx.to_le_bytes())?;
            out.write_all(&[0])?; // parameter size
            out.write_all(&[ns])?;
            out.write_all(&0u32.to_le_bytes())?; // revision
            out.write_all(&cluster.to_le_bytes())?;
            out.write_all(&blob.to_le_bytes())?;
        }
        DirentBytes::Redirect { target } => {
            out.write_all(&MIME_REDIRECT.to_le_bytes())?;
            out.write_all(&[0])?; // parameter size
            out.write_all(&[ns])?;
            out.write_all(&0u32.to_le_bytes())?; // revision
            out.write_all(&target.to_le_bytes())?;
        }
    }
    // pathTitle: path, NUL, then the title only when it differs from the path.
    out.write_all(path.as_bytes())?;
    out.write_all(&[0])?;
    if title != path {
        out.write_all(title.as_bytes())?;
    }
    out.write_all(&[0])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zim::{Target, Zim};
    use crate::zimcommon::{u16le, u32le, u64le, ZIM_MAGIC};

    /// The header fields of a written archive, read straight from the bytes
    /// (the byte-level check on top of the shared `ZimHeader` type; the two
    /// old-scheme fields exist only in the raw layout).
    fn raw_header(bytes: &[u8]) -> (u32, u16, u16, [u8; 16], u32, u32, u64, u64, u64, u64, u32, u32, u64) {
        let h = &bytes[..80];
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&h[8..24]);
        (
            u32le(&h[0..4]),   // magic
            u16le(&h[4..6]),   // major
            u16le(&h[6..8]),   // minor
            uuid,
            u32le(&h[24..28]), // entry count
            u32le(&h[28..32]), // cluster count
            u64le(&h[32..40]), // pathPtrPos
            u64le(&h[40..48]), // titleIdxPos
            u64le(&h[48..56]), // clusterPtrPos
            u64le(&h[56..64]), // mimeListPos
            u32le(&h[64..68]), // mainPage
            u32le(&h[68..72]), // layoutPage
            u64le(&h[72..80]), // checksumPos
        )
    }

    /// Entry (mime id, namespace, url, title, target) of `idx`, as the reader
    /// parses it from the written archive.
    fn entry_of(zim: &Zim, idx: u32) -> (u16, u8, String, String, Target) {
        let e = zim.get_entry(idx).unwrap();
        (e.mime, e.namespace, e.url.clone(), e.title.clone(), e.target)
    }

    fn blob_of(zim: &Zim, e: &Target) -> Vec<u8> {
        match e {
            Target::Cluster(c, b) => zim.read_blob(*c, *b).unwrap(),
            t => panic!("expected content target, got {t:?}"),
        }
    }

    /// The standard small archive used by several tests:
    /// C/a -> C/b (redirect, front article), C/b item "B title" markdown,
    /// C/c item (non-front), M/Name metadata, M/Counter, W/mainPage,
    /// X/listing/titleOrdered/v1.
    fn build_small_archive(out: &std::path::Path) {
        let mut zc = ZimCreator::new(out).unwrap();
        zc.set_main_path("b");
        zc.add_item("b", "B title", "text/markdown", true, true, b"body of b".to_vec()).unwrap();
        zc.add_item("c", "", "text/html;raw=true", false, false, b"<html>c</html>".to_vec()).unwrap();
        zc.add_redirection("a", "A", "b", true).unwrap();
        zc.add_metadata("Name", b"szmcp-test", "text/plain;charset=UTF-8").unwrap();
        zc.add_illustration(48, &[1, 2, 3, 4, 5]).unwrap();
        zc.finish().unwrap();
    }

    #[test]
    fn small_archive_structure_and_reader_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("small.zim");
        build_small_archive(&out_path);
        let bytes = std::fs::read(&out_path).unwrap();

        // ---- header field layout ----
        let (magic, major, minor, _uuid, entries, clusters, _path_ptr, title_idx_pos, _cluster_ptr, mime_list_pos, main_page, layout_page, checksum_pos) =
            raw_header(&bytes);
        assert_eq!(magic, ZIM_MAGIC);
        assert_eq!((major, minor), (6, 3));
        assert_eq!(entries, 8, "3 C + M/Counter + M/Illustration + M/Name + W/mainPage + X/listing");
        assert_eq!(clusters, 2, "one comp + one uncomp cluster");
        assert_eq!(title_idx_pos, u64::MAX);
        assert_eq!(layout_page, u32::MAX);
        assert_eq!(mime_list_pos, 80);
        assert_eq!(main_page, 6, "W/mainPage entry index");
        // the MIME list is NUL-terminated strings followed by a final NUL;
        // the rest of the 80..2048 area stays zero (libzim's sparse gap)
        let mut mime_end = 80usize;
        while bytes[mime_end] != 0 {
            mime_end += bytes[mime_end..].iter().position(|&b| b == 0).unwrap() + 1;
        }
        mime_end += 1; // the final empty NUL terminating the list
        assert!(bytes[mime_end..2048].iter().all(|&b| b == 0), "gap is zeros");
assert_eq!(mime_end, 196, "mime list = 6 types + terminator");
        assert_eq!(checksum_pos as usize, bytes.len() - 16, "checksum right after cluster ptrs");

        // checksum: MD5 over everything before checksumPos == the last 16 bytes
        let digest: [u8; 16] = md5::Md5::digest(&bytes[..bytes.len() - 16]).into();
        assert_eq!(&bytes[bytes.len() - 16..], &digest[..]);

        // open with the crate's reader; entry order = sorted (ns byte, path)
        let z = Zim::open(&out_path).unwrap();
        // 0 C/a, 1 C/b, 2 C/c, 3 M/Counter, 4 M/Illustration_48x48@1,
        // 5 M/Name, 6 W/mainPage, 7 X/listing/titleOrdered/v1
        let (_, ns, url, title, target) = entry_of(&z, 0);
        assert_eq!((ns, url.as_str()), (b'C', "a"));
        assert_eq!(title, "A");
        assert_eq!(target, Target::Redirect(1), "redirect target = C/b's entry index");
        // the terminal article behind the redirect
        let e_b = z.get_entry(1).unwrap();
        assert_eq!((e_b.namespace, e_b.url, e_b.title), (b'C', "b".to_string(), "B title".to_string()));
        assert_eq!(blob_of(&z, &e_b.target), b"body of b".to_vec());
        let (_, ns_c, url_c, title_c, t_c) = entry_of(&z, 2);
        assert_eq!((ns_c, url_c.as_str(), title_c.as_str()), (b'C', "c", ""));
        assert_eq!(blob_of(&z, &t_c), b"<html>c</html>".to_vec());
        // M/Counter exists, is "text/plain", compressed, with the right count
        let (_, _, url_ct, title_ct, t_ct) = entry_of(&z, 3);
        assert_eq!((url_ct.as_str(), title_ct.as_str()), ("Counter", ""));
        assert_eq!(z.mime_type(z.get_entry(3).unwrap().mime), Some("text/plain"));
        assert_eq!(blob_of(&z, &t_ct), b"text/html=1;text/markdown=1".to_vec());
        // metadata M/Name (illustration sorts between Counter and Name)
        let idx_name = z.find_entry(b'M', "Name").unwrap().unwrap();
        assert_eq!(idx_name, 5);
        assert_eq!(blob_of(&z, &z.get_entry(idx_name).unwrap().target), b"szmcp-test".to_vec());
        // illustration: image/png, uncompressed
        let ill_idx = z.find_entry(b'M', "Illustration_48x48@1").unwrap().unwrap();
        assert_eq!(z.mime_type(z.get_entry(ill_idx).unwrap().mime), Some("image/png"));
        let (_, _, _, _, t_ill) = entry_of(&z, ill_idx);
        assert_eq!(blob_of(&z, &t_ill), vec![1u8, 2, 3, 4, 5]);
        // W/mainPage redirect targets C/b (idx 1); empty stored title reads
        // back as ""
        let (_, ns_w, url_w, title_w, t_w) = entry_of(&z, 6);
        assert_eq!((ns_w, url_w.as_str(), title_w.as_str()), (b'W', "mainPage", ""));
        assert!(matches!(t_w, Target::Redirect(1)));
        // X/listing/titleOrdered/v1: front articles (C/a "A", C/b "B title")
        // sorted by title bytes -> [a, b]
        let (_, ns_x, url_x, _, t_x) = entry_of(&z, 7);
        assert_eq!((ns_x, url_x.as_str()), (b'X', "listing/titleOrdered/v1"));
        let listing = blob_of(&z, &t_x);
        let idxs: Vec<u32> = listing
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(idxs, vec![0, 1], "front articles sorted by title");
    }

    #[test]
    fn dangling_redirect_and_its_placeholder_are_removed() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("dangling.zim");
        let mut zc = ZimCreator::new(&out_path).unwrap();
        zc.add_item("kept", "Kept", "text/plain", false, true, b"kept body".to_vec()).unwrap();
        zc.add_redirection("gone", "Gone", "never-added", true).unwrap();
        zc.finish().unwrap();

        let bytes = std::fs::read(&out_path).unwrap();
        let (_, _, _, _, entries, _, _, _, _, _, main_page, _, _) = raw_header(&bytes);
        // C/gone and its placeholder C/never-added are dropped; remaining:
        // C/kept (0), M/Counter (1), X/listing/titleOrdered/v1 (2)
        assert_eq!(entries, 3);
        assert_eq!(main_page, u32::MAX, "no main path set");

        let z = Zim::open(&out_path).unwrap();
        assert_eq!(z.entry_count(), 3);
        assert_eq!(z.find_entry(b'C', "gone").unwrap(), None, "dangling redirect dropped");
        assert_eq!(z.find_entry(b'C', "never-added").unwrap(), None, "placeholder dropped");
        assert_eq!(z.find_entry(b'C', "kept").unwrap(), Some(0));
        let (_, ns0, url0, _, _) = entry_of(&z, 0);
        assert_eq!((ns0, url0.as_str()), (b'C', "kept"));
        let (_, ns1, url1, _, _) = entry_of(&z, 1);
        assert_eq!((ns1, url1.as_str()), (b'M', "Counter"));
        let (_, ns2, url2, _, _) = entry_of(&z, 2);
        assert_eq!((ns2, url2.as_str()), (b'X', "listing/titleOrdered/v1"));
        assert_eq!(blob_of(&z, &z.get_entry(0).unwrap().target), b"kept body".to_vec());
        // the listing has no front articles left? kept IS a front article
        let listing = blob_of(&z, &z.get_entry(2).unwrap().target);
        let idxs: Vec<u32> = listing
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(idxs, vec![0], "kept is a front article");
    }

    #[test]
    fn dangling_redirect_chain_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("chain.zim");
        let mut zc = ZimCreator::new(&out_path).unwrap();
        zc.add_item("real", "Real", "text/plain", false, true, b"real body".to_vec()).unwrap();
        // b -> dangling placeholder (target never added): b and its target go;
        // then a -> b becomes a blind chain (its terminal was removed) and is
        // dropped too.
        zc.add_redirection("b", "B", "missing-target", true).unwrap();
        zc.add_redirection("a", "A", "b", true).unwrap();
        zc.finish().unwrap();

        let z = Zim::open(&out_path).unwrap();
        assert_eq!(z.entry_count(), 3, "C/real + M/Counter + X/listing (a, b, missing-target all dropped)");
        assert_eq!(z.find_entry(b'C', "a").unwrap(), None);
        assert_eq!(z.find_entry(b'C', "b").unwrap(), None);
        assert_eq!(z.find_entry(b'C', "missing-target").unwrap(), None);
        let (_, ns, url, _, _) = entry_of(&z, 0);
        assert_eq!((ns, url.as_str()), (b'C', "real"));
    }

    #[test]
    fn redirect_chain_through_two_redirects_resolves() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("resolve.zim");
        let mut zc = ZimCreator::new(&out_path).unwrap();
        zc.add_item("end", "End", "text/plain", false, false, b"end body".to_vec()).unwrap();
        zc.add_redirection("mid", "Mid", "end", true).unwrap();
        zc.add_redirection("start", "Start", "mid", true).unwrap();
        zc.finish().unwrap();

        let z = Zim::open(&out_path).unwrap();
        // sorted C paths: "end" < "mid" < "start" -> 0 C/end, 1 C/mid, 2 C/start
        assert_eq!(z.find_entry(b'C', "end").unwrap(), Some(0));
        assert_eq!(z.find_entry(b'C', "mid").unwrap(), Some(1));
        assert_eq!(z.find_entry(b'C', "start").unwrap(), Some(2));
        let e_start = z.get_entry(2).unwrap();
        let idx_mid = match e_start.target {
            Target::Redirect(t) => t,
            t => panic!("start should be a redirect, got {t:?}"),
        };
        let e_mid = z.get_entry(idx_mid).unwrap();
        assert_eq!((e_mid.namespace, e_mid.url.as_str()), (b'C', "mid"));
        let idx_end = match e_mid.target {
            Target::Redirect(t) => t,
            t => panic!("mid should be a redirect, got {t:?}"),
        };
        let e_end = z.get_entry(idx_end).unwrap();
        assert_eq!((e_end.namespace, e_end.url.as_str()), (b'C', "end"));
        assert_eq!(blob_of(&z, &e_end.target), b"end body".to_vec());
    }

    #[test]
    fn redirect_target_added_after_the_redirect_fills_placeholder() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("fill.zim");
        let mut zc = ZimCreator::new(&out_path).unwrap();
        // the target is added AFTER the redirect
        zc.add_redirection("first", "First", "later", true).unwrap();
        zc.add_item("later", "Later", "text/plain", false, true, b"later body".to_vec()).unwrap();
        zc.finish().unwrap();

        let z = Zim::open(&out_path).unwrap();
        assert_eq!(z.entry_count(), 4, "C/first, C/later, M/Counter, X/listing");
        assert_eq!(z.find_entry(b'C', "first").unwrap(), Some(0));
        assert_eq!(z.find_entry(b'C', "later").unwrap(), Some(1));
        let e = z.get_entry(0).unwrap();
        let Target::Redirect(t) = e.target else {
            panic!("placeholder must have been filled, redirect kept");
        };
        let e_later = z.get_entry(t).unwrap();
        assert_eq!((e_later.namespace, e_later.url.as_str()), (b'C', "later"));
        assert_eq!(e_later.title, "Later");
        assert_eq!(blob_of(&z, &e_later.target), b"later body".to_vec());
    }

    #[test]
    fn big_items_split_across_clusters() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("split.zim");
        let mut zc = ZimCreator::new(&out_path).unwrap();
        // two ~1.5 MiB compressed items: the first fills the open comp cluster,
        // the second would push it past 2 MiB -> a second comp cluster opens.
        let blob1 = vec![b'a'; 1_500_000];
        let blob2 = vec![b'y'; 1_500_000];
        zc.add_item("alpha", "Alpha", "text/plain", true, true, blob1.clone()).unwrap();
        zc.add_item("beta", "Beta", "text/plain", true, true, blob2.clone()).unwrap();
        zc.finish().unwrap();

        let bytes = std::fs::read(&out_path).unwrap();
        let (_, _, _, _, _, clusters, _, _, _, _, _, _, _) = raw_header(&bytes);
        // the M/Counter blob forces a second compressed cluster, and the
        // listing an uncompressed one; the two items must be in DIFFERENT ones
        assert_eq!(clusters, 3, "comp(alpha) + comp(beta)+counter + uncomp(listing)");

        let z = Zim::open(&out_path).unwrap();
        // sorted C paths: "alpha" < "beta" -> 0 C/alpha, 1 C/beta
        let e1 = z.get_entry(0).unwrap();
        let e2 = z.get_entry(1).unwrap();
        let Target::Cluster(c1, b1) = e1.target else { panic!("alpha has content") };
        let Target::Cluster(c2, b2) = e2.target else { panic!("beta should have content") };
        assert_ne!(c1, c2, "the two items land in different clusters");
        assert_eq!(b1, 0);
        assert_eq!(b2, 0);
        assert_eq!(z.read_blob(c1, b1).unwrap(), blob1);
        assert_eq!(z.read_blob(c2, b2).unwrap(), blob2);
    }

    #[test]
    fn empty_content_item_still_gets_a_blob() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("empty.zim");
        let mut zc = ZimCreator::new(&out_path).unwrap();
        zc.add_item("empty", "", "text/plain", false, true, Vec::new()).unwrap();
        zc.add_item("nonempty", "N", "text/plain", false, true, b"data".to_vec()).unwrap();
        zc.finish().unwrap();

        let z = Zim::open(&out_path).unwrap();
        // sorted C paths: "empty" < "nonempty" -> 0, 1
        let e0 = z.get_entry(0).unwrap();
        assert_eq!((e0.namespace, e0.url.as_str()), (b'C', "empty"));
        let Target::Cluster(c, b) = e0.target else { panic!("empty item has content") };
        assert_eq!(z.read_blob(c, b).unwrap(), Vec::<u8>::new());
        // two blobs in the same cluster: empty first, data second
        let e1 = z.get_entry(1).unwrap();
        let Target::Cluster(c2, b2) = e1.target else { panic!() };
        assert_eq!(c2, c, "same cluster");
        assert_eq!(b2, 1);
        assert_eq!(z.read_blob(c2, b2).unwrap(), b"data".to_vec());
    }

    #[test]
    fn front_article_flag_controls_the_listing() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("front.zim");
        let mut zc = ZimCreator::new(&out_path).unwrap();
        // "alpha" NOT front; "gamma" front with empty title (effective title =
        // its path); "zeta" a front REDIRECT whose title "Mid title" sorts
        // before "gamma" byte-wise ('M' < 'g').
        zc.add_item("alpha", "AAA", "text/plain", false, false, b"1".to_vec()).unwrap();
        zc.add_item("gamma", "", "text/plain", false, true, b"g body".to_vec()).unwrap();
        zc.add_redirection("zeta", "Mid title", "gamma", true).unwrap();
        zc.finish().unwrap();

        let z = Zim::open(&out_path).unwrap();
        // sorted C paths: alpha(0), gamma(1), zeta(2)
        let listing_idx = z.find_entry(b'X', "listing/titleOrdered/v1").unwrap().unwrap();
        let listing = blob_of(&z, &z.get_entry(listing_idx).unwrap().target);
        let idxs: Vec<u32> = listing
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // front articles: C/zeta (title "Mid title", idx 2) and C/gamma
        // (effective title "gamma", idx 1); sorted by title bytes:
        // "Mid title" < "gamma" (uppercase before lowercase)
        assert_eq!(idxs, vec![2, 1]);
    }

    #[test]
    fn xapian_index_blob_is_stored_uncompressed_and_equal() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("xapian.zim");
        // a fake "single-file database" (any bytes; the writer stores it
        // verbatim in one uncompressed blob)
        let db_path = dir.path().join("title.idx");
        let db_bytes: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&db_path, &db_bytes).unwrap();
        let mut zc = ZimCreator::new(&out_path).unwrap();
        zc.add_item("a", "A", "text/plain", false, true, b"body".to_vec()).unwrap();
        zc.add_xapian_index("title/xapian", &db_path).unwrap();
        zc.finish().unwrap();
        let z = Zim::open(&out_path).unwrap();
        let idx = z.find_entry(b'X', "title/xapian").unwrap().unwrap();
        let e = z.get_entry(idx).unwrap();
        assert_eq!(z.mime_type(e.mime), Some("application/octet-stream+xapian"));
        let Target::Cluster(c, b) = e.target else { panic!("xapian index has content") };
        // uncompressed cluster: the reader stores file_offset for the blob
        let blob = z.read_blob(c, b).unwrap();
        assert_eq!(blob, db_bytes);
    }
}

