//! Minimal read-only ZIM container reader.
//!
//! Just enough of the [ZIM file format](https://wiki.openzim.org/wiki/ZIM_file_format)
//! to serve articles and open the Xapian full-text index embedded in the
//! archive (`X/fulltext/xapian`) in place, with no copy.
//!
//! All multi-byte integers are little-endian. Archives may be a single
//! `.zim` file or a chunked archive (`.zimaa`, `.zimab`, ...).

use memmap2::Mmap;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use xapian2::Database as XapianDatabase;

const ZIM_MAGIC: u32 = 72173914;
/// Directory-entry mimetype sentinels.
const MIME_REDIRECT: u16 = 0xffff;
const MIME_LINKTARGET: u16 = 0xfffe;
const MIME_DELETED: u16 = 0xfffd;
const MAX_REDIRECT_HOPS: u32 = 50;
/// Upper bound on how much of a directory entry we ever need to read.
const DIRENT_WINDOW: u64 = 64 * 1024;

/// One OS file (the whole archive, or one chunk) mapped into memory.
struct Part {
    path: PathBuf,
    mmap: Mmap,
    /// Virtual offset of the start of this part within the archive.
    start: u64,
}

/// Read view over a (possibly chunked) ZIM archive.
struct Store {
    parts: Vec<Part>,
    len: u64,
}

impl Store {
    /// Open the archive rooted at `path`. If `path` is an existing file it is
    /// the whole archive; otherwise the chunked form (`path` + "aa", "ab", ...)
    /// is used.
    fn open(path: &Path) -> io::Result<Store> {
        if path.is_file() {
            let file = File::open(path)?;
            let len = file.metadata()?.len();
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("cannot map {}: {e}", path.display())))?;
            return Ok(Store {
                parts: vec![Part { path: path.to_path_buf(), mmap, start: 0 }],
                len,
            });
        }

        // Chunked archive: `path` + two-letter suffix, from "aa" upward.
        let base = path.to_string_lossy().into_owned();
        let mut parts = Vec::new();
        let mut start = 0u64;
        'outer: for i in 0..26u8 {
            for j in 0..26u8 {
                let chunk = PathBuf::from(format!(
                    "{}{}{}",
                    base,
                    (b'a' + i) as char,
                    (b'a' + j) as char
                ));
                if !chunk.is_file() {
                    break 'outer;
                }
                let file = File::open(&chunk)?;
                let len = file.metadata()?.len();
                let mmap = unsafe { Mmap::map(&file) }
                    .map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("cannot map {chunk:?}: {e}")))?;
                parts.push(Part { path: chunk, mmap, start });
                start += len;
            }
        }
        if parts.is_empty() {
            return Err(io::Error::new(ErrorKind::NotFound, format!("no such archive: {}", path.display())));
        }
        Ok(Store { parts, len: start })
    }

    fn len(&self) -> u64 {
        self.len
    }

    /// Read `len` bytes at virtual offset `off`. Returns a borrowed view when
    /// the range lies inside a single part.
    fn read<'a>(&'a self, off: u64, len: u64) -> io::Result<std::borrow::Cow<'a, [u8]>> {
        if off > self.len || len > self.len - off {
            return Err(io::Error::new(ErrorKind::InvalidData, "read out of bounds in ZIM archive"));
        }
        if len == 0 {
            return Ok(std::borrow::Cow::Borrowed(&[]));
        }
        let idx = self.part_index(off).ok_or_else(|| {
            io::Error::new(ErrorKind::InvalidData, "read out of bounds in ZIM archive")
        })?;
        let part = &self.parts[idx];
        let rel = (off - part.start) as usize;
        let part_len = part.mmap.len();
        if rel + len as usize <= part_len {
            return Ok(std::borrow::Cow::Borrowed(&part.mmap[rel..rel + len as usize]));
        }
        // Spans part boundaries: concatenate.
        let mut buf: Vec<u8> = part.mmap[rel..].to_vec();
        for p in &self.parts[idx + 1..] {
            buf.extend_from_slice(&p.mmap[..]);
        }
        if buf.len() < len as usize {
            return Err(io::Error::new(ErrorKind::InvalidData, "read out of bounds in ZIM archive"));
        }
        Ok(std::borrow::Cow::Owned(buf[..len as usize].to_vec()))
    }

    fn part_index(&self, voff: u64) -> Option<usize> {
        if voff >= self.len {
            return None;
        }
        let idx = self.parts.partition_point(|p| p.start > voff).saturating_sub(1);
        let part = &self.parts[idx];
        (voff >= part.start && voff < part.start + part.mmap.len() as u64).then_some(idx)
    }

    /// The (file path, in-file offset) of the OS file holding virtual offset
    /// `voff`, and that file's length.
    fn file_location(&self, voff: u64) -> Option<(&Path, u64, u64)> {
        let idx = self.part_index(voff)?;
        let part = &self.parts[idx];
        Some((part.path.as_path(), voff - part.start, part.mmap.len() as u64))
    }
}

/// Parsed ZIM header (the fields we care about).
#[derive(Debug, Clone)]
pub struct Header {
    pub major: u16,
    pub minor: u16,
    pub entry_count: u32,
    pub cluster_count: u32,
    pub url_ptr_pos: u64,
    pub cluster_ptr_pos: u64,
    pub mime_list_pos: u64,
    pub checksum_pos: u64,
}

/// What a directory entry points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Redirect to another directory entry (by URL index).
    Redirect(u32),
    /// Cluster and blob index of the entry's content.
    Cluster(u32, u32),
    /// No content (deleted or link-target entries).
    None,
}

/// A parsed directory entry.
#[derive(Debug, Clone)]
pub struct Entry {
    pub mime: u16,
    pub namespace: u8,
    pub url: String,
    pub title: String,
    pub target: Target,
}

/// Where a blob's bytes live. `file_offset`/`length` are only set for blobs in
/// uncompressed clusters.
struct BlobLocation {
    compression: u8,
    /// Virtual file offset of the blob's raw bytes (uncompressed clusters only).
    file_offset: Option<u64>,
    length: u64,
}

/// A ZIM archive.
pub struct Zim {
    store: Store,
    pub header: Header,
    mime_types: Vec<String>,
}

#[inline]
fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
#[inline]
fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
#[inline]
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().unwrap())
}

/// Read a NUL-terminated string at `off`; returns (string, offset-after-NUL).
fn cstr(b: &[u8], off: usize) -> io::Result<(&str, usize)> {
    let nul = b[off..]
        .iter()
        .position(|&c| c == 0)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "unterminated string in ZIM entry"))?;
    let s = std::str::from_utf8(&b[off..off + nul])
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "non-UTF-8 string in ZIM entry"))?;
    Ok((s, off + nul + 1))
}

/// The canonical full path of an entry: `<namespace>/<url>` under the new
/// namespace scheme (>=6.1, url excludes the namespace), or the stored url
/// itself under the old scheme (url already includes the namespace).
fn full_path(e: &Entry) -> String {
    let ns = e.namespace as char;
    if e.url.starts_with(&format!("{ns}/")) {
        e.url.clone()
    } else {
        format!("{ns}/{}", e.url)
    }
}

impl Zim {
    /// Open a ZIM archive and parse the header + MIME type list. Pointer
    /// lists, entries, and clusters are read on demand.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Zim> {
        let path = path.as_ref();
        let store = Store::open(path)?;
        let hlen = 80u64.min(store.len());
        let hdr = store.read(0, hlen)?;
        if hlen < 80 || u32le(&hdr[0..4]) != ZIM_MAGIC {
            return Err(io::Error::new(ErrorKind::InvalidData, format!("not a ZIM file: {}", path.display())));
        }
        let header = Header {
            major: u16le(&hdr[4..6]),
            minor: u16le(&hdr[6..8]),
            entry_count: u32le(&hdr[24..28]),
            cluster_count: u32le(&hdr[28..32]),
            url_ptr_pos: u64le(&hdr[32..40]),
            cluster_ptr_pos: u64le(&hdr[48..56]),
            mime_list_pos: u64le(&hdr[56..64]),
            checksum_pos: u64le(&hdr[72..80]),
        };
        if !(5..=6).contains(&header.major) {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("unsupported ZIM version {}.{}", header.major, header.minor),
            ));
        }
        if header.mime_list_pos != 80 {
            return Err(io::Error::new(ErrorKind::InvalidData, "bad mimeListPos in ZIM header"));
        }
        if header.checksum_pos + 16 > store.len()
            || header.url_ptr_pos.saturating_add(8u64 * header.entry_count as u64) > store.len()
            || header.cluster_ptr_pos.saturating_add(8u64 * header.cluster_count as u64) > store.len()
        {
            return Err(io::Error::new(ErrorKind::InvalidData, "inconsistent ZIM header"));
        }

        // MIME type list: NUL-terminated strings, ended by an empty string.
        let mut mime_types = Vec::new();
        let window = store.read(80, DIRENT_WINDOW.min(store.len() - 80))?;
        let mut off = 0usize;
        while off < window.len() {
            let (s, next) = match cstr(&window, off) {
                Ok(x) => x,
                Err(_) => break,
            };
            if s.is_empty() {
                break;
            }
            mime_types.push(s.to_string());
            off = next;
        }

        Ok(Zim { store, header, mime_types })
    }

    pub fn entry_count(&self) -> u32 {
        self.header.entry_count
    }

    /// The MIME type string for mime id `id`, if known.
    pub fn mime_type(&self, id: u16) -> Option<&str> {
        (id < MIME_DELETED).then_some(id as usize).and_then(|i| self.mime_types.get(i).map(String::as_str))
    }

    /// One 8-byte pointer from a pointer list.
    fn ptr(&self, base: u64, idx: u32) -> io::Result<u64> {
        let at = base
            .checked_add(idx as u64 * 8)
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "pointer list out of bounds"))?;
        Ok(u64le(&self.store.read(at, 8)?))
    }

    /// Offset of the directory entry at `idx` in the URL pointer list.
    fn dirent_offset(&self, idx: u32) -> io::Result<u64> {
        if idx >= self.header.entry_count {
            return Err(io::Error::new(ErrorKind::InvalidData, "entry index out of bounds"));
        }
        self.ptr(self.header.url_ptr_pos, idx)
    }

    /// Parse the directory entry at `idx`.
    pub fn get_entry(&self, idx: u32) -> io::Result<Entry> {
        let off = self.dirent_offset(idx)?;
        if off >= self.store.len() {
            return Err(io::Error::new(ErrorKind::InvalidData, "entry offset out of bounds"));
        }
        let window = self.store.read(off, DIRENT_WINDOW.min(self.store.len() - off))?;
        if window.len() < 8 {
            return Err(io::Error::new(ErrorKind::InvalidData, "truncated directory entry"));
        }
        let mime = u16le(&window[0..2]);
        let namespace = window[3];
        let target = match mime {
            MIME_REDIRECT => Target::Redirect(u32le(&window[8..12])),
            MIME_LINKTARGET | MIME_DELETED => Target::None,
            _ => Target::Cluster(u32le(&window[8..12]), u32le(&window[12..16])),
        };
        let url_off = match mime {
            MIME_REDIRECT => 12,
            MIME_LINKTARGET | MIME_DELETED => 8,
            _ => 16,
        };
        let (url, url_end) = cstr(&window, url_off)?;
        let (title, _) = cstr(&window, url_end)?;
        Ok(Entry {
            mime,
            namespace,
            url: url.to_string(),
            title: title.to_string(),
            target,
        })
    }

    /// Binary search the URL pointer list for (namespace, url). Entries are
    /// sorted by the namespace byte followed by the stored url bytes.
    pub fn find_entry(&self, namespace: u8, url: &str) -> io::Result<Option<u32>> {
        let mut lo = 0u32;
        let mut hi = self.header.entry_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let entry = self.get_entry(mid)?;
            let ord = (entry.namespace, entry.url.as_bytes()).cmp(&(namespace, url.as_bytes()));
            match ord {
                std::cmp::Ordering::Less => lo = mid + 1,
                _ => hi = mid,
            }
        }
        if lo < self.header.entry_count {
            let entry = self.get_entry(lo)?;
            if entry.namespace == namespace && entry.url == url {
                return Ok(Some(lo));
            }
        }
        Ok(None)
    }

    /// Resolve a user-supplied path to an entry index, tolerating both the
    /// old namespace scheme (stored url includes the namespace, e.g.
    /// "A/index.html") and the new one (>=6.1, stored url excludes it).
    pub fn resolve_path(&self, path: &str) -> io::Result<Option<u32>> {
        let path = path.trim_matches('/');
        if path.is_empty() {
            return Ok(None);
        }
        let mut candidates: Vec<(u8, String)> = Vec::new();
        if let Some((first, rest)) = path.split_once('/') {
            if first.is_ascii() && first.len() == 1 {
                let ns = first.as_bytes()[0];
                candidates.push((ns, rest.to_string())); // new scheme
                candidates.push((ns, path.to_string())); // old scheme
            }
        }
        for ns in [b'C', b'A'] {
            candidates.push((ns, path.to_string()));
            candidates.push((ns, format!("{}/{}", ns as char, path)));
        }
        for (ns, url) in candidates {
            if let Some(idx) = self.find_entry(ns, &url)? {
                return Ok(Some(idx));
            }
        }
        Ok(None)
    }

    fn cluster_range(&self, cluster: u32) -> io::Result<(u64, u64)> {
        if cluster >= self.header.cluster_count {
            return Err(io::Error::new(ErrorKind::InvalidData, "cluster index out of bounds"));
        }
        let start = self.ptr(self.header.cluster_ptr_pos, cluster)?;
        let end = if cluster + 1 < self.header.cluster_count {
            self.ptr(self.header.cluster_ptr_pos, cluster + 1)?
        } else {
            self.header.checksum_pos
        };
        if start >= end {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster range"));
        }
        Ok((start, end))
    }

    /// Locate a blob. Offset/length are only reported for blobs in
    /// uncompressed clusters (where the bytes sit verbatim in the file).
    fn locate_blob(&self, cluster: u32, blob: u32) -> io::Result<BlobLocation> {
        let (start, end) = self.cluster_range(cluster)?;
        let info = self.store.read(start, 1)?[0];
        let compression = info & 0x0f;
        if compression != 0 && compression != 1 {
            return Ok(BlobLocation { compression, file_offset: None, length: 0 });
        }
        let sz = if (info & 0x10) != 0 { 8 } else { 4 }; // 64-bit blob offsets
        let tbl0 = start + 1 + blob as u64 * sz;
        let (s0, s1) = if sz == 8 {
            let a = u64le(&self.store.read(tbl0, 8)?);
            let b = u64le(&self.store.read(tbl0 + 8, 8)?);
            (a, b)
        } else {
            (u32le(&self.store.read(tbl0, 4)?) as u64, u32le(&self.store.read(tbl0 + 4, 4)?) as u64)
        };
        let body_len = end - (start + 1);
        if s0 > body_len || s1 > body_len || s0 > s1 {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
        }
        Ok(BlobLocation {
            compression,
            file_offset: Some(start + 1 + s0),
            length: s1 - s0,
        })
    }

    /// Read a blob's bytes, decompressing the cluster when needed.
    pub fn read_blob(&self, cluster: u32, blob: u32) -> io::Result<Vec<u8>> {
        self.read_blob_prefix(cluster, blob, u64::MAX)
    }

    /// Read at most `max` bytes of a blob (no decompression cost beyond what
    /// the cluster format requires).
    pub fn read_blob_prefix(&self, cluster: u32, blob: u32, max: u64) -> io::Result<Vec<u8>> {
        let loc = self.locate_blob(cluster, blob)?;
        if let Some(voff) = loc.file_offset {
            let take = loc.length.min(max);
            return Ok(self.store.read(voff, take)?[..].to_vec());
        }
        let (start, end) = self.cluster_range(cluster)?;
        let body = self.store.read(start + 1, end - start - 1)?;
        let decompressed = match loc.compression {
            5 => zstd::decode_all(body.as_ref()).map_err(|e| {
                io::Error::new(ErrorKind::InvalidData, format!("zstd decompression failed: {e}"))
            })?,
            4 => {
                // liblzma's auto decoder transparently handles the raw LZMA2
                // streams ZIM clusters store.
                let mut decoder =
                    xz2::stream::Stream::new_auto_decoder(0, 0).map_err(|e| {
                        io::Error::new(ErrorKind::InvalidData, format!("lzma decoder init failed: {e:?}"))
                    })?;
                let mut out = Vec::new();
                let status = decoder
                    .process_vec(body.as_ref(), &mut out, xz2::stream::Action::Finish)
                    .map_err(|e| {
                        io::Error::new(ErrorKind::InvalidData, format!("lzma decompression failed: {e:?}"))
                    })?;
                if !matches!(status, xz2::stream::Status::StreamEnd) {
                    return Err(io::Error::new(ErrorKind::InvalidData, "lzma stream did not end cleanly"));
                }
                out
            }
            other => {
                return Err(io::Error::new(ErrorKind::InvalidData, format!("unknown cluster compression {other}")));
            }
        };
        let info = self.store.read(start, 1)?[0];
        let sz = if (info & 0x10) != 0 { 8 } else { 4 };
        let dec_u64 = |d: &[u8], i: u64| -> io::Result<u64> {
            if sz == 8 {
                Ok(u64le(&d[i as usize..i as usize + 8]))
            } else {
                Ok(u32le(&d[i as usize..i as usize + 4]) as u64)
            }
        };
        if (decompressed.len() as u64) < 2 * sz {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed decompressed cluster"));
        }
        let tbl_size = dec_u64(&decompressed, 0)?;
        if tbl_size < sz || tbl_size % sz != 0 || (tbl_size as usize) > decompressed.len() {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
        }
        let n = tbl_size / sz; // offsets including the end sentinel
        let bi = blob as u64;
        if bi + 1 >= n {
            return Err(io::Error::new(ErrorKind::InvalidData, "blob index out of bounds"));
        }
        let s = dec_u64(&decompressed, bi * sz)?;
        let e = dec_u64(&decompressed, (bi + 1) * sz)?;
        if s > e || (e as usize) > decompressed.len() {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
        }
        let take = (e - s).min(max) as usize;
        Ok(decompressed[s as usize..s as usize + take].to_vec())
    }

    /// Open the archive's full-text Xapian index, preferring the zero-copy
    /// path (opening the in-file glass database at the blob's offset).
    /// Returns `Ok(None)` when the archive carries no full-text index.
    pub fn open_fulltext_xapian(&self) -> io::Result<Option<XapianDatabase>> {
        const CANDIDATES: &[(u8, &str)] = &[
            (b'X', "fulltext/xapian"),
            (b'X', "X/fulltext/xapian"),
            (b'X', "fulltextindex/xapian/FullTextIndex"),
            (b'X', "X/fulltextindex/xapian/FullTextIndex"),
        ];
        let mut idx = None;
        for (ns, url) in CANDIDATES {
            if let Some(i) = self.find_entry(*ns, url)? {
                idx = Some(i);
                break;
            }
        }
        let Some(idx) = idx else { return Ok(None) };
        let entry = self.get_entry(idx)?;
        let Target::Cluster(cluster, blob) = entry.target else {
            return Err(io::Error::new(ErrorKind::InvalidData, "full-text index entry has no content"));
        };
        let loc = self.locate_blob(cluster, blob)?;
        if let Some(voff) = loc.file_offset {
            if let Some((path, off_in_file, file_len)) = self.store.file_location(voff) {
                if off_in_file + loc.length <= file_len {
                    match xapian2::Database::open_at(path, off_in_file, xapian2::DbFlags::NONE) {
                        Ok(db) => return Ok(Some(db)),
                        Err(e) => {
                            tracing::warn!("open_at for embedded Xapian index failed ({e}); copying to temp file");
                        }
                    }
                }
            }
        }
        // Fallback: copy the (decompressed) index to a temp file.
        let bytes = self.read_blob(cluster, blob)?;
        static TMP_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!("szmcp-xapian-{}-{n}.xdb", std::process::id()));
        std::fs::write(&tmp, &bytes).map_err(|e| {
            io::Error::new(ErrorKind::Other, format!("failed to write temp Xapian index: {e}"))
        })?;
        xapian2::Database::open(&tmp)
            .map(Some)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.msg()))
    }
}

/// An article served from an archive: the final (post-redirect) entry plus
/// the full content bytes.
pub struct Article {
    pub title: String,
    pub full_path: String,
    pub mime_type: Option<String>,
    pub bytes: Vec<u8>,
}

/// A ZIM archive plus lazily-opened, cached Xapian search database.
pub struct Archive {
    /// Archive name relative to the ZIM directory (e.g. "wikipedia.zim").
    pub name: String,
    pub zim: Zim,
    xapian: RwLock<Option<Arc<XapianDatabase>>>,
}

impl Archive {
    pub fn new(name: String, zim: Zim) -> Self {
        Self { name, zim, xapian: RwLock::new(None) }
    }

    pub fn article_count(&self) -> u32 {
        self.zim.entry_count()
    }

    /// Whether this archive carries an openable full-text Xapian index.
    pub fn searchable(&self) -> bool {
        self.zim.open_fulltext_xapian().map(|db| db.is_some()).unwrap_or(false)
    }

    /// The full-text Xapian database, opened (and cached) on first use.
    pub fn xapian_db(&self) -> io::Result<Option<Arc<XapianDatabase>>> {
        let mut guard = self.xapian.write().unwrap();
        if let Some(db) = guard.as_ref() {
            return Ok(Some(db.clone()));
        }
        let db = match self.zim.open_fulltext_xapian()? {
            Some(db) => Arc::new(db),
            None => return Ok(None),
        };
        *guard = Some(db.clone());
        Ok(Some(db))
    }

    /// Resolve `path` (following redirects) and read the article's content.
    pub fn get_article(&self, path: &str) -> io::Result<Article> {
        let Some(idx) = self.zim.resolve_path(path)? else {
            return Err(io::Error::new(ErrorKind::NotFound, format!("article not found: {path}")));
        };
        let (entry, bytes) = self.content_of(idx)?;
        Ok(Article {
            title: if entry.title.is_empty() { full_path(&entry) } else { entry.title.clone() },
            full_path: full_path(&entry),
            mime_type: self.zim.mime_type(entry.mime).map(String::from),
            bytes,
        })
    }

    /// Resolve `path`, following redirects, reading at most `max_bytes` of
    /// content (for lightweight previews such as search intros). Returns the
    /// raw entry title (may be empty - the caller applies fallbacks), the
    /// MIME type, and the content prefix.
    pub fn article_preview(
        &self,
        path: &str,
        max_bytes: u64,
    ) -> io::Result<Option<(String, Option<String>, Vec<u8>)>> {
        let Some(idx) = self.zim.resolve_path(path)? else {
            return Ok(None);
        };
        let entry = self.resolve_entry(idx)?;
        let mime = self.zim.mime_type(entry.mime).map(String::from);
        let bytes = match entry.target {
            Target::Cluster(c, b) => self.zim.read_blob_prefix(c, b, max_bytes)?,
            _ => return Ok(None),
        };
        Ok(Some((entry.title, mime, bytes)))
    }

    /// Follow a redirect chain to the terminal entry (bounded).
    fn resolve_entry(&self, idx: u32) -> io::Result<Entry> {
        let mut entry = self.zim.get_entry(idx)?;
        for _ in 0..MAX_REDIRECT_HOPS {
            match entry.target {
                Target::Redirect(next) => entry = self.zim.get_entry(next)?,
                _ => return Ok(entry),
            }
        }
        Err(io::Error::new(ErrorKind::InvalidData, "redirect chain did not terminate"))
    }

    /// Resolve `idx` (following redirects) and read the full content.
    fn content_of(&self, idx: u32) -> io::Result<(Entry, Vec<u8>)> {
        let entry = self.resolve_entry(idx)?;
        let bytes = match entry.target {
            Target::Cluster(c, b) => self.zim.read_blob(c, b)?,
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "entry has no content (redirect loop or deleted entry)"
                ));
            }
        };
        Ok((entry, bytes))
    }
}

/// A set of ZIM archives loaded from a directory.
pub struct ZimLibrary {
    pub root: PathBuf,
    pub archives: Vec<Arc<Archive>>,
}

impl ZimLibrary {
    /// Recursively scan `root` for ZIM files and open them. Files that fail
    /// to open are skipped with a warning.
    pub fn scan(root: &Path) -> io::Result<ZimLibrary> {
        if !root.is_dir() {
            return Err(io::Error::new(ErrorKind::NotFound, format!("not a directory: {}", root.display())));
        }
        let mut names: Vec<String> = Vec::new();
        scan_dir(root, root, &mut names)?;
        names.sort();
        names.dedup();
        let mut archives = Vec::new();
        for name in names {
            match Zim::open(root.join(&name)) {
                Ok(zim) => archives.push(Arc::new(Archive::new(name, zim))),
                Err(e) => eprintln!("szmcp: skipping {}: {e}", name),
            }
        }
        if archives.is_empty() {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("no ZIM files could be loaded from {}", root.display()),
            ));
        }
        Ok(ZimLibrary { root: root.to_path_buf(), archives })
    }
}

/// Map a file name to an archive name, if it is one ("x.zim" -> "x.zim";
/// "x.zimaa" -> "x.zim").
fn zim_archive_name(name: &std::ffi::OsStr) -> Option<String> {
    let s = name.to_string_lossy();
    if let Some(stripped) = s.strip_suffix(".zim") {
        Some(format!("{stripped}.zim"))
    } else if let Some(stripped) = s.strip_suffix(".zimaa") {
        Some(format!("{stripped}.zim"))
    } else {
        None
    }
}

fn scan_dir(dir: &Path, root: &Path, out: &mut Vec<String>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            scan_dir(&path, root, out)?;
        } else if let Some(base) = path.file_name().and_then(zim_archive_name) {
            // Normalize chunk names (x.zimaa) to the base archive name (x.zim).
            let parent = path
                .parent()
                .and_then(|p| p.strip_prefix(root).ok())
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            out.push(if parent.is_empty() {
                base
            } else {
                format!("{parent}/{base}")
            });
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Synthetic ZIM archive builder for tests (adapted from zxr).

    use super::*;
    use std::io::Write;

    pub struct TestEntry {
        pub namespace: u8,
        pub url: &'static str,
        pub title: &'static str,
        pub mime: u16,
        pub body: &'static [u8],
    }

    pub struct TestRedirect {
        pub namespace: u8,
        pub url: &'static str,
        pub title: &'static str,
        pub target_content: usize,
    }

    fn push_zstring(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(s.as_bytes());
        buf.push(0);
    }

    /// Build an uncompressed single-blob cluster body for `data`.
    fn build_cluster(data: &[u8]) -> Vec<u8> {
        let mut c = Vec::new();
        c.push(0u8); // info byte: comp=0 (uncompressed), not extended.
        let first = 8u32; // 2 offsets * 4 bytes.
        c.extend_from_slice(&first.to_le_bytes());
        c.extend_from_slice(&(first + data.len() as u32).to_le_bytes());
        c.extend_from_slice(data);
        c
    }

    /// Assemble a complete in-memory ZIM. `index`, when given, is stored as
    /// an uncompressed `X/fulltext/xapian` content entry (a single-blob
    /// cluster), like a real archive's embedded Xapian full-text index.
    pub fn build_archive(
        mime_types: &[&str],
        content: &[TestEntry],
        redirects: &[TestRedirect],
        main_page_content: usize,
        index: Option<&[u8]>,
    ) -> Vec<u8> {
        let has_index = index.is_some();
        let entry_count = (content.len() + redirects.len() + has_index as usize) as u32;
        let cluster_count = (content.len() + has_index as usize) as u32;

        let mut mime_blob = Vec::new();
        for m in mime_types {
            push_zstring(&mut mime_blob, m);
        }
        mime_blob.push(0);

        enum Logical<'a> {
            Content { e: &'a TestEntry, cluster: u32 },
            Redirect { r: &'a TestRedirect },
            Index { bytes: &'a [u8], cluster: u32 },
        }
        let mut logical: Vec<(u8, &str, Logical)> = Vec::new();
        for (ci, e) in content.iter().enumerate() {
            logical.push((e.namespace, e.url, Logical::Content { e, cluster: ci as u32 }));
        }
        for r in redirects {
            logical.push((r.namespace, r.url, Logical::Redirect { r }));
        }
        if let Some(ix) = index {
            logical.push((
                b'X',
                "fulltext/xapian",
                Logical::Index { bytes: ix, cluster: content.len() as u32 },
            ));
        }
        logical.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

        let mut index_of_url: std::collections::HashMap<(u8, &str), u32> = std::collections::HashMap::new();
        for (i, (ns, url, _)) in logical.iter().enumerate() {
            index_of_url.insert((*ns, *url), i as u32);
        }

        let mut entry_bodies: Vec<Vec<u8>> = Vec::new();
        for (ns, _url, item) in &logical {
            let mut b = Vec::new();
            match item {
                Logical::Content { e, cluster } => {
                    b.extend_from_slice(&e.mime.to_le_bytes());
                    b.push(0);
                    b.push(*ns);
                    b.extend_from_slice(&0u32.to_le_bytes());
                    b.extend_from_slice(&cluster.to_le_bytes());
                    b.extend_from_slice(&0u32.to_le_bytes());
                    push_zstring(&mut b, e.url);
                    push_zstring(&mut b, e.title);
                }
                Logical::Redirect { r } => {
                    let target = index_of_url[&(
                        content[r.target_content].namespace,
                        content[r.target_content].url,
                    )];
                    b.extend_from_slice(&MIME_REDIRECT.to_le_bytes());
                    b.push(0);
                    b.push(*ns);
                    b.extend_from_slice(&0u32.to_le_bytes());
                    b.extend_from_slice(&target.to_le_bytes());
                    push_zstring(&mut b, r.url);
                    push_zstring(&mut b, r.title);
                }
                Logical::Index { bytes, cluster } => {
                    b.extend_from_slice(&0u16.to_le_bytes());
                    b.push(0);
                    b.push(*ns);
                    b.extend_from_slice(&0u32.to_le_bytes());
                    b.extend_from_slice(&cluster.to_le_bytes());
                    b.extend_from_slice(&0u32.to_le_bytes());
                    push_zstring(&mut b, "fulltext/xapian");
                    push_zstring(&mut b, "Xapian index");
                    let _ = bytes;
                }
            }
            entry_bodies.push(b);
        }

        let mut clusters: Vec<Vec<u8>> = content.iter().map(|e| build_cluster(e.body)).collect();
        if let Some(ix) = index {
            clusters.push(build_cluster(ix));
        }

        let mime_pos = 80u64;
        let url_ptr_pos = mime_pos + mime_blob.len() as u64;
        let title_ptr_pos = url_ptr_pos + entry_count as u64 * 8;
        let cluster_ptr_pos = title_ptr_pos + entry_count as u64 * 4;
        let entries_pos = cluster_ptr_pos + cluster_count as u64 * 8;

        let mut entry_offsets = Vec::new();
        let mut cur = entries_pos;
        for b in &entry_bodies {
            entry_offsets.push(cur);
            cur += b.len() as u64;
        }
        let mut cluster_offsets = Vec::new();
        for c in &clusters {
            cluster_offsets.push(cur);
            cur += c.len() as u64;
        }
        let checksum_pos = cur;

        let main_page_idx = index_of_url[&(
            content[main_page_content].namespace,
            content[main_page_content].url,
        )];

        let mut out = vec![0u8; 80];
        out[0..4].copy_from_slice(&ZIM_MAGIC.to_le_bytes());
        out[4..6].copy_from_slice(&6u16.to_le_bytes()); // major
        out[6..8].copy_from_slice(&1u16.to_le_bytes()); // minor
        out[24..28].copy_from_slice(&entry_count.to_le_bytes());
        out[28..32].copy_from_slice(&cluster_count.to_le_bytes());
        out[32..40].copy_from_slice(&url_ptr_pos.to_le_bytes());
        out[40..48].copy_from_slice(&title_ptr_pos.to_le_bytes());
        out[48..56].copy_from_slice(&cluster_ptr_pos.to_le_bytes());
        out[56..64].copy_from_slice(&mime_pos.to_le_bytes());
        out[64..68].copy_from_slice(&main_page_idx.to_le_bytes());
        out[68..72].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
        out[72..80].copy_from_slice(&checksum_pos.to_le_bytes());

        out.extend_from_slice(&mime_blob);
        for off in &entry_offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        // Title pointer list: entry_count u32 indices (URL order is good
        // enough for the tests - szmcp never reads the title ordering).
        for i in 0..entry_count {
            out.extend_from_slice(&i.to_le_bytes());
        }
        for off in &cluster_offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        for b in &entry_bodies {
            out.extend_from_slice(b);
        }
        for c in &clusters {
            out.extend_from_slice(c);
        }
        out.extend_from_slice(&[0u8; 16]);
        out
    }

    fn open_bytes(bytes: &[u8]) -> (Zim, tempfile::NamedTempFile) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        let z = Zim::open(f.path()).unwrap();
        (z, f)
    }

    fn sample_archive() -> (Archive, tempfile::NamedTempFile) {
        let content = [
            TestEntry {
                namespace: b'C',
                url: "Apple",
                title: "Apple",
                mime: 0,
                body: b"<html><head><title>Apple</title></head><body><h1>Apple</h1><h2 id=\"History\">History</h2><p>Apples have been cultivated for 10,000 years.</p><h2 id=\"Uses\">Uses</h2><p>Eaten fresh or cooked.</p></body></html>",
            },
            TestEntry {
                namespace: b'C',
                url: "Banana",
                title: "Banana",
                mime: 0,
                body: b"<html><body><h2>Bananas</h2><p>A delicious fruit.</p></body></html>",
            },
            TestEntry {
                namespace: b'C',
                url: "style.css",
                title: "style.css",
                mime: 1,
                body: b"body{color:red}",
            },
        ];
        let redirects = [TestRedirect {
            namespace: b'C',
            url: "Apple_fruit",
            title: "Apple (fruit)",
            target_content: 0,
        }];
        let bytes = build_archive(&["text/html", "text/css"], &content, &redirects, 0, None);
        let (z, f) = open_bytes(&bytes);
        (Archive::new("sample.zim".to_string(), z), f)
    }

    #[test]
    fn open_and_entry_count() {
        let (a, _f) = sample_archive();
        assert_eq!(a.article_count(), 4);
    }

    #[test]
    fn get_article_content() {
        let (a, _f) = sample_archive();
        let art = a.get_article("Banana").unwrap();
        assert_eq!(art.bytes, b"<html><body><h2>Bananas</h2><p>A delicious fruit.</p></body></html>");
        assert_eq!(art.title, "Banana");
        assert_eq!(art.mime_type.as_deref(), Some("text/html"));
    }

    #[test]
    fn get_article_follows_redirect_and_new_scheme_paths() {
        let (a, _f) = sample_archive();
        // Bare path.
        assert!(a.get_article("Apple").is_ok());
        // Namespaced path (new scheme: ns + bare url).
        assert!(a.get_article("C/Apple").is_ok());
        // Leading slash.
        assert!(a.get_article("/C/Apple_fruit").is_ok());
        let art = a.get_article("C/Apple_fruit").unwrap();
        assert!(art.bytes.starts_with(b"<html><head>"));
        assert_eq!(art.title, "Apple");
        assert_eq!(art.full_path, "C/Apple");
    }

    #[test]
    fn get_article_not_found() {
        let (a, _f) = sample_archive();
        assert!(a.get_article("Nope").is_err());
    }

    #[test]
    fn fulltext_index_absent_reports_none() {
        let (a, _f) = sample_archive();
        assert!(a.zim.open_fulltext_xapian().unwrap().is_none());
    }

    #[test]
    fn preview_reads_only_prefix() {
        let (a, _f) = sample_archive();
        let (title, mime, bytes) = a.article_preview("Apple", 64).unwrap().unwrap();
        assert_eq!(title, "Apple");
        assert_eq!(mime.as_deref(), Some("text/html"));
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn old_scheme_paths() {
        // Old namespace scheme: stored url includes the namespace.
        let content = [TestEntry {
            namespace: b'A',
            url: "A/index.html",
            title: "Home",
            mime: 0,
            body: b"<p>hello old world</p>",
        }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, None);
        let (z, _f) = open_bytes(&bytes);
        let a = Archive::new("old.zim".to_string(), z);
        let art = a.get_article("index.html").unwrap();
        assert_eq!(art.bytes, b"<p>hello old world</p>");
        assert_eq!(art.full_path, "A/index.html");
    }
}
