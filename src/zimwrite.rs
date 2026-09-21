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
/// of it, and the blob index within the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlobRef {
    compress: bool,
    generation: u32,
    blob: u32,
}

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
    /// Redirect target that was never filled in (libzim's `isPlaceholder`).
    Placeholder,
}

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
    /// The main entry path (for the `W/mainPage` redirect at finish).
    main_path: String,
    comp_cluster: OpenCluster,
    uncomp_cluster: OpenCluster,
    /// Closed clusters in close order; the vec index is the cluster's archive
    /// index. Compressed/uncompressed clusters close interleaved, so each
    /// open cluster's final index is only known at its own close.
    closed_clusters: Vec<OpenCluster>,
    /// Number of clusters closed per compression kind (the "generation" of the
    /// currently open clusters).
    comp_generation: u32,
    uncomp_generation: u32,
    /// Generation -> archive cluster index, per slot.
    comp_gen_idx: Vec<u32>,
    uncomp_gen_idx: Vec<u32>,
    /// Mime string -> count for `M/Counter` (sorted like libzim's std::map).
    mime_counter: BTreeMap<String, u64>,
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
            main_path: String::new(),
            comp_cluster: OpenCluster::new(true),
            uncomp_cluster: OpenCluster::new(false),
            closed_clusters: Vec::new(),
            comp_generation: 0,
            uncomp_generation: 0,
            comp_gen_idx: Vec::new(),
            uncomp_gen_idx: Vec::new(),
            mime_counter: BTreeMap::new(),
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
    fn add_item_data(&mut self, compress: bool, content: &[u8]) -> (bool, u32, u32) {
        let item_size = content.len() as u64;
        if compress {
            if self.comp_cluster.count() > 0
                && self.comp_cluster.size() + item_size >= CLUSTER_TARGET_SIZE
            {
                self.close_cluster(Slot::Compressed);
            }
            let blob = self.comp_cluster.count();
            self.comp_cluster.push(content);
            (true, self.comp_generation, blob)
        } else {
            if self.uncomp_cluster.count() > 0
                && self.uncomp_cluster.size() + item_size >= CLUSTER_TARGET_SIZE
            {
                self.close_cluster(Slot::Uncompressed);
            }
            let blob = self.uncomp_cluster.count();
            self.uncomp_cluster.push(content);
            (false, self.uncomp_generation, blob)
        }
    }

    /// Stream a blob of known `len` from `reader` into the uncompressed open
    /// cluster (used for embedded Xapian databases, which can be large).
    fn add_item_streaming(&mut self, len: u64, reader: &mut dyn Read) -> io::Result<(bool, u32, u32)> {
        if self.uncomp_cluster.count() > 0
            && self.uncomp_cluster.size() + len >= CLUSTER_TARGET_SIZE
        {
            self.close_cluster(Slot::Uncompressed);
        }
        let blob = self.uncomp_cluster.count();
        self.uncomp_cluster.blob_ends.push(self.uncomp_cluster.data_size() + len);
        let mut copied = 0u64;
        let mut buf = [0u8; 1024 * 1024];
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

    /// Map a location's cluster generation to the final archive cluster index
    /// (all referenced clusters are closed before the file is written).
    fn cluster_number(&self, compress: bool, generation: u32) -> u32 {
        let table = if compress { &self.comp_gen_idx } else { &self.uncomp_gen_idx };
        table[generation as usize]
    }

    /// Close a cluster (it must hold at least one blob) and open a fresh one.
    fn close_cluster(&mut self, slot: Slot) {
        let count = match slot {
            Slot::Compressed => self.comp_cluster.count(),
            Slot::Uncompressed => self.uncomp_cluster.count(),
        };
        if count == 0 {
            return;
        }
        let archive_idx = self.closed_clusters.len() as u32;
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
        self.closed_clusters.push(cluster);
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
        let blob = self.add_item_data(compress, &content);
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
        let blob = self.add_item_data(compress, content);
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

        // 5. MIME list resolved: SORTED alphabetically, dirent mime ids
        // remapped from insertion order to the sorted position.
        let old_mimes = std::mem::take(&mut self.mime_by_idx);
        let mut sorted_mimes = old_mimes.clone();
        sorted_mimes.sort();
        let mapping: Vec<u16> = old_mimes
            .iter()
            .map(|m| sorted_mimes.binary_search(m).unwrap_or(0) as u16)
            .collect();
        self.mime_by_idx = sorted_mimes;
        for Dirent { kind, .. } in self.dirents.values_mut() {
            if let DirentKind::Item { mime_idx, .. } = kind {
                *mime_idx = mapping[*mime_idx as usize];
            }
        }

        // 6. M/Counter content ("mime=count;..."), COMPRESSED (text/plain).
        let counter_content = self
            .mime_counter
            .iter()
            .map(|(m, c)| format!("{m}={c}"))
            .collect::<Vec<_>>()
            .join(";");
        let (compress, generation, blob_num) = self.add_item_data(true, counter_content.as_bytes());
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
        let lb = self.add_item_data(false, &listing_blob);
        if let Some(Dirent { kind: DirentKind::Item { blob, .. }, .. }) = self
            .dirents
            .get_mut(&(b'X', "listing/titleOrdered/v1".to_string()))
        {
            *blob = BlobRef { compress: lb.0, generation: lb.1, blob: lb.2 };
        }

        // 8. Close the open clusters (compressed first, then the uncompressed
        // one — libzim's order), then write the file.
        self.close_cluster(Slot::Compressed);
        self.close_cluster(Slot::Uncompressed);
        self.write_file()
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

    fn write_file(&mut self) -> io::Result<()> {
        // Dirent bytes (all known here); offsets are relative to the section.
        let mut dirent_bytes: Vec<u8> = Vec::new();
        let mut dirent_offsets: Vec<u64> = Vec::with_capacity(self.dirents.len());
        for ((ns, path), d) in &self.dirents {
            dirent_offsets.push(dirent_bytes.len() as u64);
            write_dirent(&mut dirent_bytes, *ns, path, d, self)?;
        }

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

        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.out_path)?;
        let mut out = io::BufWriter::new(&mut f);
        // Placeholder header (overwritten at the very end, like libzim's
        // writeLastParts seek(0) + header.write).
        out.write_all(&[0u8; HEADER_SIZE as usize])?;
        out.write_all(&mime_blob)?;
        out.write_all(&vec![0u8; (CLUSTER_BASE_OFFSET - HEADER_SIZE - mime_blob.len() as u64) as usize])?;

        // Clusters at CLUSTER_BASE_OFFSET, in close order.
        let mut cluster_offsets: Vec<u64> = Vec::with_capacity(self.closed_clusters.len());
        let mut pos = CLUSTER_BASE_OFFSET;
        for c in &self.closed_clusters {
            cluster_offsets.push(pos);
            pos += c.write_to(&mut out)? as u64;
        }

        // Dirents, then the path pointer table (u64 LE offsets per dirent).
        let dirent_start = pos;
        out.write_all(&dirent_bytes)?;
        let path_ptr_pos = dirent_start + dirent_bytes.len() as u64;
        for &off in &dirent_offsets {
            out.write_all(&(dirent_start + off).to_le_bytes())?;
        }

        // Cluster pointer table (u64 LE absolute offsets per cluster).
        let cluster_ptr_pos = path_ptr_pos + self.dirents.len() as u64 * 8;
        for &off in &cluster_offsets {
            out.write_all(&off.to_le_bytes())?;
        }
        let checksum_pos = cluster_ptr_pos + self.closed_clusters.len() as u64 * 8;

        let main_page = self
            .dirents
            .get(&(b'W', "mainPage".to_string()))
            .map(|d| d.idx)
            .unwrap_or(u32::MAX);
        let header = ZimHeader {
            major: ZIM_MAJOR,
            minor: ZIM_MINOR_VERSION,
            uuid: self.uuid,
            entry_count: self.dirents.len() as u32,
            cluster_count: self.closed_clusters.len() as u32,
            url_ptr_pos: path_ptr_pos,
            title_ptr_pos: NO_TITLE_PTR_POS,
            cluster_ptr_pos,
            mime_list_pos: HEADER_SIZE,
            main_page,
            layout_page: NO_LAYOUT_PAGE,
            checksum_pos,
        }
        .serialize();

        // Write the real header at 0 (libzim writes it last), flush, then
        // compute the checksum by re-reading the file from 0 (libzim's
        // writeChecksum) and append the 16-byte digest.
        out.flush()?;
        drop(out);
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&header)?;
        let mut hasher = md5::Md5::new();
        let mut left = checksum_pos;
        let mut buf = [0u8; 65536];
        f.seek(SeekFrom::Start(0))?;
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            f.read_exact(&mut buf[..want])?;
            hasher.update(&buf[..want]);
            left -= want as u64;
        }
        let digest: [u8; 16] = hasher.finalize().into();
        f.seek(SeekFrom::Start(checksum_pos))?;
        f.write_all(&digest)?;
        f.sync_all()?;
        Ok(())
    }
}

/// Serialize one dirent (libzim's `Dirent::write`): the 12/16-byte head, then
/// the path (NUL-terminated), then the title — omitted when equal to the
/// path (libzim's PathTitleTinyString::concat) — and a final NUL.
fn write_dirent(
    out: &mut Vec<u8>,
    ns: u8,
    path: &str,
    d: &Dirent,
    creator: &ZimCreator,
) -> io::Result<()> {
    match &d.kind {
        DirentKind::Item { mime_idx, blob } => {
            let cluster = creator.cluster_number(blob.compress, blob.generation);
            out.extend_from_slice(&mime_idx.to_le_bytes());
            out.push(0); // parameter size
            out.push(ns);
            out.extend_from_slice(&0u32.to_le_bytes()); // revision
            out.extend_from_slice(&cluster.to_le_bytes());
            out.extend_from_slice(&blob.blob.to_le_bytes());
        }
        DirentKind::Redirect { target } => {
            let target_idx = creator
                .dirents
                .get(target)
                .map(|t| t.idx)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("dangling redirect {}/{} remains", ns as char, path),
                    )
                })?;
            out.extend_from_slice(&MIME_REDIRECT.to_le_bytes());
            out.push(0); // parameter size
            out.push(ns);
            out.extend_from_slice(&0u32.to_le_bytes()); // revision
            out.extend_from_slice(&target_idx.to_le_bytes());
        }
        DirentKind::Placeholder => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unfilled placeholder {}/{} remains", ns as char, path),
            ));
        }
    }
    // pathTitle: path, NUL, then the title only when it differs from the path.
    out.extend_from_slice(path.as_bytes());
    out.push(0);
    if d.title != path {
        out.extend_from_slice(d.title.as_bytes());
    }
    out.push(0);
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

