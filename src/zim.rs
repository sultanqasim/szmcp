//! Minimal read-only ZIM container reader.
//!
//! Just enough of the [ZIM file format](https://wiki.openzim.org/wiki/ZIM_file_format)
//! to serve articles and open the Xapian search indexes embedded in the
//! archive (`X/fulltext/xapian`, plus `X/title/xapian` when present) in
//! place, with no copy.
//!
//! All multi-byte integers are little-endian. Archives may be a single
//! `.zim` file or a chunked archive (`.zimaa`, `.zimab`, ...).

use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use xapian2::Database as XapianDatabase;

use crate::zimcommon::{
    dirent_order, u16le, u32le, u64le, CLUSTER_COMPRESSION_MASK, CLUSTER_EXTENDED_BIT,
    CLUSTER_LZMA, CLUSTER_UNCOMPRESSED, CLUSTER_ZSTD, MIME_DELETED, MIME_LINKTARGET,
    MIME_REDIRECT, ZIM_MAGIC,
};

const MAX_REDIRECT_HOPS: u32 = 50;
/// Upper bound on how much of a directory entry we ever need to read.
const DIRENT_WINDOW: u64 = 64 * 1024;
/// Reads at or above this size bypass the mapping (direct file I/O): the
/// cluster bodies of a multi-GB archive would otherwise map their pages
/// into the reader's resident set as the convert pass walks the archive.
const BIG_READ: u64 = 256 * 1024;

/// One OS file (the whole archive, or one chunk) mapped into memory.
struct Part {
    path: PathBuf,
    file: File,
    mmap: Mmap,
    /// Virtual offset of the start of this part within the archive.
    start: u64,
}

impl Part {
    /// Direct unbuffered read of `len` bytes at in-file offset `off`.
    fn pread(&self, off: u64, len: u64) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; len as usize];
        self.pread_into(off, &mut buf)?;
        Ok(buf)
    }

    fn pread_into(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file
            .read_exact_at(buf, off)
            .map_err(|e| io::Error::new(e.kind(), format!("pread failed: {e}")))
    }
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
                parts: vec![Part { path: path.to_path_buf(), file, mmap, start: 0 }],
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
                parts.push(Part { path: chunk, file, mmap, start });
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
    /// the range lies inside a single part. Reads of [`BIG_READ`] bytes or
    /// more go through direct file I/O instead of the mapping: an 8+ GB
    /// archive's cluster pages would otherwise accumulate in this process's
    /// page-cache-resident RSS as the convert pass touches them (clean
    /// mapped pages the kernel only reclaims under pressure).
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
        if len >= BIG_READ {
            return Ok(std::borrow::Cow::Owned(part.pread(off - part.start, len)?));
        }
        if rel + len as usize <= part_len {
            return Ok(std::borrow::Cow::Borrowed(&part.mmap[rel..rel + len as usize]));
        }
        // Spans part boundaries: pread the needed ranges.
        let mut buf: Vec<u8> = Vec::with_capacity(len as usize);
        let mut at = off;
        let mut need = len;
        for p in &self.parts[idx..] {
            if need == 0 {
                break;
            }
            let avail = p.mmap.len() as u64 - (at - p.start);
            let take = avail.min(need);
            let from = buf.len();
            buf.resize(from + take as usize, 0);
            p.pread_into(at - p.start, &mut buf[from..])?;
            at += take;
            need -= take;
        }
        if need > 0 {
            return Err(io::Error::new(ErrorKind::InvalidData, "read out of bounds in ZIM archive"));
        }
        Ok(std::borrow::Cow::Owned(buf))
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

/// The parsed ZIM file header: the shared [`crate::zimcommon::ZimHeader`],
/// so the reader and the writer speak the same field names.
pub use crate::zimcommon::ZimHeader;

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
///
/// `Zim` is `Sync` (plain fields over read-only `memmap2::Mmap`s): one
/// `&Zim` is shared across the parallel convert pass's worker threads.
pub struct Zim {
    store: Store,
    pub header: ZimHeader,
    mime_types: Vec<String>,
}

/// Uppercase the first character of `s` (the rest is left untouched).
fn upper_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Capitalize every '_'-separated word: "salt_lake_city" -> "Salt_Lake_City",
/// the shape MediaWiki gives article URLs.
fn title_case(s: &str) -> String {
    s.split('_').map(upper_first).collect::<Vec<_>>().join("_")
}

/// Read a blob-table entry (`sz` = 4 or 8 bytes wide) at byte offset `off`.
fn table_int(d: &[u8], off: usize, sz: usize) -> u64 {
    if sz == 8 {
        u64le(&d[off..off + 8])
    } else {
        u32le(&d[off..off + 4]) as u64
    }
}

/// Incremental decoder for a compressed cluster body: decompressed bytes are
/// only produced as `decode_to` asks for them, so reading a blob prefix never
/// decodes the rest of the cluster.
struct ClusterDecoder<'a> {
    stream: Box<dyn io::Read + 'a>,
    buf: Vec<u8>,
}

impl<'a> ClusterDecoder<'a> {
    fn new(compression: u8, body: &'a [u8]) -> io::Result<ClusterDecoder<'a>> {
        let stream: Box<dyn io::Read + 'a> = match compression {
            CLUSTER_ZSTD => Box::new(zstd::stream::read::Decoder::with_buffer(body)?),
            // liblzma's auto decoder transparently handles the LZMA
            // streams ZIM clusters store.
            CLUSTER_LZMA => Box::new(xz2::read::XzDecoder::new(body)),
            other => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("unknown cluster compression {other}"),
                ))
            }
        };
        Ok(ClusterDecoder { stream, buf: Vec::new() })
    }

    /// Decode until at least `len` bytes are buffered. A stream that ends
    /// early (or fails) is a malformed cluster.
    fn decode_to(&mut self, len: u64) -> io::Result<&[u8]> {
        while self.buf.len() < len as usize {
            let filled = self.buf.len();
            // Grow in bounded steps so a bogus length runs into the stream's
            // end (an error) before its bytes are ever buffered.
            self.buf.resize((filled + 1024 * 1024).min(len as usize), 0);
            match self.stream.read(&mut self.buf[filled..]) {
                Ok(0) => {
                    return Err(io::Error::new(ErrorKind::InvalidData, "malformed decompressed cluster"));
                }
                Ok(n) => self.buf.truncate(filled + n),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        format!("cluster decompression failed: {e}"),
                    ));
                }
            }
        }
        Ok(&self.buf[..len as usize])
    }
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
        let header = ZimHeader::parse(&hdr);
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
            let ord = dirent_order(
                (entry.namespace, entry.url.as_bytes()),
                (namespace, url.as_bytes()),
            );
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
        // Trim user-supplied decoration: surrounding whitespace, a leading
        // "./" (how in-page hrefs reference articles) and leading slashes.
        // A trailing slash is NOT trimmed: "C/" is the (nonexistent) C
        // namespace root, and silently trimming it would resolve to
        // whatever article happens to be named "C".
        let path = path.trim().trim_start_matches("./").trim_start_matches('/');
        if path.is_empty() {
            return Ok(None);
        }
        let mut candidates: Vec<(u8, String)> = Vec::new();
        if let Some((first, rest)) = path.split_once('/') {
            if first.is_ascii() && first.len() == 1 && !rest.is_empty() {
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
        let compression = info & CLUSTER_COMPRESSION_MASK;
        if compression != 0 && compression != CLUSTER_UNCOMPRESSED {
            return Ok(BlobLocation { compression, file_offset: None, length: 0 });
        }
        let sz = if (info & CLUSTER_EXTENDED_BIT) != 0 { 8 } else { 4 }; // 64-bit blob offsets
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

    /// Read at most `max` bytes of a blob. Compressed clusters are decoded
    /// incrementally: only the bytes leading up to the wanted prefix are ever
    /// decompressed, not the whole cluster.
    pub fn read_blob_prefix(&self, cluster: u32, blob: u32, max: u64) -> io::Result<Vec<u8>> {
        let loc = self.locate_blob(cluster, blob)?;
        if let Some(voff) = loc.file_offset {
            let take = loc.length.min(max);
            return Ok(self.store.read(voff, take)?[..].to_vec());
        }
        let (start, end) = self.cluster_range(cluster)?;
        let body = self.store.read(start + 1, end - start - 1)?;
        let info = self.store.read(start, 1)?[0];
        let sz = if (info & CLUSTER_EXTENDED_BIT) != 0 { 8 } else { 4 }; // 64-bit blob offsets
        let mut dec = ClusterDecoder::new(loc.compression, body.as_ref())?;
        // The decompressed data starts with the blob-offset table, whose
        // first entry gives the table's own byte size; blob data follows.
        let tbl_size = table_int(dec.decode_to(sz as u64)?, 0, sz);
        if tbl_size < sz as u64 || tbl_size % sz as u64 != 0 {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
        }
        let n = tbl_size / sz as u64; // offsets including the end sentinel
        let bi = blob as u64;
        if bi + 1 >= n {
            return Err(io::Error::new(ErrorKind::InvalidData, "blob index out of bounds"));
        }
        let tbl = dec.decode_to(tbl_size)?;
        let s = table_int(tbl, bi as usize * sz, sz);
        let e = table_int(tbl, (bi as usize + 1) * sz, sz);
        if s > e {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
        }
        let take = (e - s).min(max);
        let data = dec.decode_to(s + take)?;
        Ok(data[s as usize..(s + take) as usize].to_vec())
    }

    /// Parse the directory entry at `idx`, returning only the MIME id and the
    /// target - no path/title allocation. The lean read the parallel convert
    /// pass uses to classify every entry before the (rarer) full reads.
    pub fn entry_head(&self, idx: u32) -> io::Result<(u16, Target)> {
        let off = self.dirent_offset(idx)?;
        if off >= self.store.len() {
            return Err(io::Error::new(ErrorKind::InvalidData, "entry offset out of bounds"));
        }
        let window = self.store.read(off, 16.min(self.store.len() - off))?;
        if window.len() < 8 {
            return Err(io::Error::new(ErrorKind::InvalidData, "truncated directory entry"));
        }
        let mime = u16le(&window[0..2]);
        let target = match mime {
            MIME_REDIRECT => {
                if window.len() < 12 {
                    return Err(io::Error::new(ErrorKind::InvalidData, "truncated directory entry"));
                }
                Target::Redirect(u32le(&window[8..12]))
            }
            MIME_LINKTARGET | MIME_DELETED => Target::None,
            _ => {
                if window.len() < 16 {
                    return Err(io::Error::new(ErrorKind::InvalidData, "truncated directory entry"));
                }
                Target::Cluster(u32le(&window[8..12]), u32le(&window[12..16]))
            }
        };
        Ok((mime, target))
    }

    /// Follow the redirect chain starting at entry `idx` to the first
    /// non-redirect entry (transitively, cycle-safe, hence bounded).
    /// `None` when the chain leaves the entry range or cycles - a dangling
    /// redirect, like `zim2zim.redirect_target`.
    pub fn redirect_terminal(&self, idx: u32) -> Option<u32> {
        let mut seen: HashSet<u32> = HashSet::new();
        let mut cur = idx;
        loop {
            if cur >= self.header.entry_count || !seen.insert(cur) {
                return None;
            }
            let (_, target) = self.entry_head(cur).ok()?;
            match target {
                Target::Redirect(next) => cur = next,
                _ => return Some(cur),
            }
        }
    }

    /// The whole decompressed payload of a cluster: the blob-offset table
    /// followed by the blob bytes (everything after the cluster's info
    /// byte). Uncompressed clusters return their raw body; compressed
    /// clusters are decoded in full (the caller decides whether to share or
    /// cache the result - see convert's `ClusterCache`).
    pub fn decompress_cluster(&self, cluster: u32) -> io::Result<Vec<u8>> {
        let (start, end) = self.cluster_range(cluster)?;
        let info = self.store.read(start, 1)?[0];
        let compression = info & CLUSTER_COMPRESSION_MASK;
        if compression == 0 || compression == CLUSTER_UNCOMPRESSED {
            return Ok(self.store.read(start + 1, end - start - 1)?[..].to_vec());
        }
        let body = self.store.read(start + 1, end - start - 1)?;
        let sz = if (info & CLUSTER_EXTENDED_BIT) != 0 { 8 } else { 4 };
        let mut dec = ClusterDecoder::new(compression, &body)?;
        // The decompressed data starts with the blob-offset table, whose
        // first entry gives the table's own byte size; the LAST entry is the
        // total payload size (each offset carries that size as delta), so
        // the full decode length is known from the table alone.
        let tbl_size = table_int(dec.decode_to(sz as u64)?, 0, sz);
        if tbl_size < sz as u64 || tbl_size % sz as u64 != 0 {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
        }
        let tbl = dec.decode_to(tbl_size)?;
        let total = table_int(tbl, tbl_size as usize - sz, sz);
        if total < tbl_size {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
        }
        Ok(dec.decode_to(total)?.to_vec())
    }

    /// The byte span `(start, end)` of blob `blob` within a cluster's
    /// decompressed payload (the [`decompress_cluster`] indexing space).
    /// Compressed clusters decode only their blob-offset table (a partial
    /// decompression from the frame start); uncompressed clusters read the
    /// table straight from the file.
    pub fn blob_span(&self, cluster: u32, blob: u32) -> io::Result<(u64, u64)> {
        let (start, end) = self.cluster_range(cluster)?;
        let info = self.store.read(start, 1)?[0];
        let compression = info & CLUSTER_COMPRESSION_MASK;
        let sz = if (info & CLUSTER_EXTENDED_BIT) != 0 { 8u64 } else { 4u64 };
        let body_len = end - start - 1;
        let (s, e) = if compression == 0 || compression == CLUSTER_UNCOMPRESSED {
            let tbl0 = start + 1 + blob as u64 * sz;
            let (a, b) = if sz == 8 {
                (
                    u64le(&self.store.read(tbl0, 8)?),
                    u64le(&self.store.read(tbl0 + 8, 8)?),
                )
            } else {
                (
                    u32le(&self.store.read(tbl0, 4)?) as u64,
                    u32le(&self.store.read(tbl0 + 4, 4)?) as u64,
                )
            };
            (a, b)
        } else {
            let body = self.store.read(start + 1, body_len)?;
            let mut dec = ClusterDecoder::new(compression, &body)?;
            let tbl_size = table_int(dec.decode_to(sz)?, 0, sz as usize);
            if tbl_size < sz || tbl_size % sz != 0 {
                return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
            }
            let n = tbl_size / sz; // offsets including the end sentinel
            if blob as u64 + 1 >= n {
                return Err(io::Error::new(ErrorKind::InvalidData, "blob index out of bounds"));
            }
            let tbl = dec.decode_to(tbl_size)?;
            (
                table_int(tbl, blob as usize * sz as usize, sz as usize),
                table_int(tbl, (blob as usize + 1) * sz as usize, sz as usize),
            )
        };
        if s > e {
            return Err(io::Error::new(ErrorKind::InvalidData, "malformed cluster blob table"));
        }
        Ok((s, e))
    }

    /// The bytes of a blob in an UNCOMPRESSED cluster, borrowed from the
    /// mmap where possible (no decompression, no copy). Compressed clusters
    /// are rejected: their callers decode the whole cluster at once.
    pub fn raw_blob(&self, cluster: u32, blob: u32) -> io::Result<std::borrow::Cow<'_, [u8]>> {
        let loc = self.locate_blob(cluster, blob)?;
        let voff = loc.file_offset.ok_or_else(|| {
            io::Error::new(ErrorKind::InvalidData, "compressed cluster: decode it as a whole")
        })?;
        self.store.read(voff, loc.length)
    }

    /// Whether the cluster's payload is compressed (zstd/lzma). Uncompressed
    /// cluster blobs are direct mmap views (see [`raw_blob`]); the parallel
    /// convert pass caches decompressed compressed clusters only.
    pub fn cluster_is_compressed(&self, cluster: u32) -> io::Result<bool> {
        let (start, _) = self.cluster_range(cluster)?;
        let info = self.store.read(start, 1)?[0];
        let compression = info & CLUSTER_COMPRESSION_MASK;
        Ok(compression != 0 && compression != CLUSTER_UNCOMPRESSED)
    }

    /// Directory names under which a ZIM may embed its two search indexes:
    /// the current openZIM layout first, then the historical variants.
    const FULLTEXT_INDEX_NAMES: &[(u8, &str)] = &[
        (b'X', "fulltext/xapian"),
        (b'X', "X/fulltext/xapian"),
        (b'X', "fulltextindex/xapian/FullTextIndex"),
        (b'X', "X/fulltextindex/xapian/FullTextIndex"),
    ];
    const TITLE_INDEX_NAMES: &[(u8, &str)] = &[
        (b'X', "title/xapian"),
        (b'X', "X/title/xapian"),
        (b'X', "titleindex/xapian/TitleIndex"),
        (b'X', "X/titleindex/xapian/TitleIndex"),
    ];

    /// The directory index of an embedded Xapian index stored under one of
    /// `candidates`, probed in order (a cheap directory probe; no database
    /// is opened).
    fn index_entry(&self, candidates: &[(u8, &str)]) -> io::Result<Option<u32>> {
        for (ns, url) in candidates {
            if let Some(i) = self.find_entry(*ns, url)? {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    /// Directory names under which a ZIM may store its `Language` metadata
    /// (current openZIM layout first, historical variant second).
    const LANGUAGE_METADATA_NAMES: &[(u8, &str)] =
        &[(b'M', "Language"), (b'M', "M/Language")];

    /// The directory index of the full-text Xapian index entry, if the
    /// archive carries one.
    fn fulltext_index_entry(&self) -> io::Result<Option<u32>> {
        self.index_entry(Self::FULLTEXT_INDEX_NAMES)
    }

    /// The directory index of the title Xapian index entry, if the archive
    /// carries one.
    fn title_index_entry(&self) -> io::Result<Option<u32>> {
        self.index_entry(Self::TITLE_INDEX_NAMES)
    }

    /// The archive's `Language` metadata (its first code, lowercased):
    /// libzim stems the embedded search index with the stemmer chosen from
    /// this metadata, so the query side must use the same language.
    /// `None` when the archive carries none. A directory probe plus one
    /// small blob read.
    fn language_metadata(&self) -> io::Result<Option<String>> {
        for (ns, url) in Self::LANGUAGE_METADATA_NAMES {
            let Some(idx) = self.find_entry(*ns, url)? else { continue };
            let entry = self.get_entry(idx)?;
            let Target::Cluster(cluster, blob) = entry.target else { continue };
            let bytes = self.read_blob(cluster, blob)?;
            let value = String::from_utf8_lossy(&bytes);
            return Ok(value
                .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
                .find(|code| !code.is_empty())
                .map(str::to_lowercase));
        }
        Ok(None)
    }

    /// Open the Xapian database stored as the content of directory entry
    /// `idx` in place: the in-file glass database is opened directly at the
    /// blob's file offset, with no temp file and no copy into RAM. Any
    /// failure is a clean error naming the index via `what` ("full-text",
    /// "title"): the blob is not raw in-file bytes (compressed cluster), it
    /// spans chunk files in a chunked archive, the archive is truncated
    /// before the blob's end, or Xapian rejects the database. Never yields
    /// `Ok(None)` once the entry exists.
    fn open_index_entry(&self, idx: u32, what: &str) -> io::Result<Option<XapianDatabase>> {
        let entry = self.get_entry(idx)?;
        let Target::Cluster(cluster, blob) = entry.target else {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("{what} index entry has no content"),
            ));
        };
        let loc = self.locate_blob(cluster, blob)?;
        let Some(voff) = loc.file_offset else {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("{what} index is not stored uncompressed; cannot open in place"),
            ));
        };
        let Some((path, off_in_file, file_len)) = self.store.file_location(voff) else {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("{what} index spans chunk files; cannot open in place"),
            ));
        };
        if off_in_file + loc.length > file_len {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("{what} index is truncated past the end of the archive"),
            ));
        }
        xapian2::Database::open_at(path, off_in_file, xapian2::DbFlags::NONE)
            .map(Some)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("embedded {what} index: {e}")))
    }

    /// Open the archive's full-text Xapian index.
    /// Returns `Ok(None)` when the archive carries no full-text index.
    pub fn open_fulltext_xapian(&self) -> io::Result<Option<XapianDatabase>> {
        match self.fulltext_index_entry()? {
            Some(idx) => self.open_index_entry(idx, "full-text"),
            None => Ok(None),
        }
    }

    /// Open the archive's title Xapian index (`X/title/xapian`), the second
    /// index openZIM archives embed: one document per article whose terms
    /// are the title's lowercased Porter2 stems (unprefixed), whose data is
    /// the article path and whose value slot 0 is the title. Both indexes
    /// share one docid space. `Ok(None)` when the archive carries none.
    pub fn open_title_xapian(&self) -> io::Result<Option<XapianDatabase>> {
        match self.title_index_entry()? {
            Some(idx) => self.open_index_entry(idx, "title"),
            None => Ok(None),
        }
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

/// The Xapian search databases of one archive, opened together: the
/// full-text index plus, when the archive carries one, the title index.
pub(crate) struct XapianHandles {
    /// The full-text index (`X/fulltext/xapian`); every searchable archive
    /// has one.
    pub(crate) fulltext: XapianDatabase,
    /// The title index (`X/title/xapian`), optional: one document per
    /// article, terms = the title's words (the same unprefixed stems the
    /// full-text index uses), document data = the article path, value slot
    /// 0 = the article title. `None` for archives without a title index.
    pub(crate) title: Option<XapianDatabase>,
}

/// A ZIM archive plus a pool of Xapian search-database handle sets.
pub struct Archive {
    /// Archive name relative to the ZIM directory (e.g. "wikipedia.zim").
    pub name: String,
    pub zim: Zim,
    /// The archive's Language metadata, read once on first use (it decides
    /// the stemmer of every search against this archive).
    language: OnceLock<Option<String>>,
    /// Idle Xapian handle sets. Handles are moved in and out of the pool,
    /// never shared: `XapianDatabase` is `Send` (not `Sync`) because Xapian
    /// does not support concurrent calls on one database object.
    xapian_pool: Mutex<Vec<XapianHandles>>,
}

impl Archive {
    pub fn new(name: String, zim: Zim) -> Self {
        Self { name, zim, language: OnceLock::new(), xapian_pool: Mutex::new(Vec::new()) }
    }

    /// The archive's Language metadata (first code, lowercased - see
    /// [`Zim::language_metadata`]), read once and cached. `None` when the
    /// archive carries no Language metadata; callers default to English
    /// stemming, which matches the indexes of those archives.
    pub fn language(&self) -> Option<String> {
        self.language
            .get_or_init(|| match self.zim.language_metadata() {
                Ok(language) => {
                    language
                }
                Err(e) => {
                    eprintln!(
                        "warning: reading the Language metadata of {} failed ({e}); queries assume english stems",
                        self.name
                    );
                    None
                }
            })
            .clone()
    }

    pub fn article_count(&self) -> u32 {
        self.zim.entry_count()
    }

    /// Whether this archive carries a full-text Xapian index - a cheap
    /// directory-existence probe; no database is opened.
    pub fn searchable(&self) -> bool {
        self.zim.fulltext_index_entry().map(|idx| idx.is_some()).unwrap_or(false)
    }

    /// Run `f` with handles on this archive's Xapian search indexes (full
    /// text, plus the title index when the archive embeds one).
    ///
    /// Handle sets are pooled and moved in and out of the pool under the
    /// lock, so no handle is ever reachable from two threads at once; `f`
    /// runs outside the lock. This amortizes the ~0.15 s open cost per
    /// archive and index without ever sharing a Xapian object between
    /// searches (Xapian's documented thread-safety contract forbids
    /// concurrent calls on one database object).
    ///
    /// `Ok(None)` when the archive carries no full-text index. A title
    /// index that fails to open only downgrades to no title band - it never
    /// fails the search.
    pub fn with_xapian<T, E>(
        &self,
        f: impl FnOnce(&XapianHandles) -> Result<T, E>,
    ) -> Result<Option<T>, E>
    where
        E: From<io::Error>,
    {
        // `XapianDatabase` is `Send`: moving handles in/out of the pool is sound.
        let mut pool = self.xapian_pool.lock().unwrap();
        let handles = match pool.pop() {
            Some(handles) => handles,
            None => {
                let Some(fulltext) = self.zim.open_fulltext_xapian()? else {
                    return Ok(None);
                };
                // A title index that exists but cannot be opened is a
                // downgrade (warn + no title band), never a failed search:
                // the full-text band works without it.
                let title = match self.zim.open_title_xapian() {
                    Ok(title) => title,
                    Err(e) => {
                        eprintln!(
                            "warning: opening the title index of {} failed ({e}); ranking by full text only",
                            self.name
                        );
                        None
                    }
                };
                XapianHandles { fulltext, title }
            }
        };
        let result = f(&handles);
        pool.push(handles);
        result.map(Some)
    }

    /// Exact-match `query` against the ZIM directory itself (not the search
    /// index): the query is interpreted as an article URL (spaces become
    /// underscores, plus case variants) or as an explicit path
    /// ("C/Chemistry"). Returns the entry's full path and its directory
    /// title (often empty in modern openZIM archives - callers apply
    /// fallbacks). Redirects are deliberately NOT followed: a redirect is a
    /// legitimate exact match, and redirect entries exist only in the
    /// directory, never in the search index.
    pub fn lookup_exact(&self, query: &str) -> io::Result<Option<(String, String)>> {
        let base = query.trim().replace(' ', "_");
        if base.is_empty() {
            return Ok(None);
        }
        let lower = base.to_lowercase();
        // Candidates in probe order: the query as typed first (an exact URL
        // or path match must not be shadowed by case variants), then the
        // case shapes an article URL might use.
        let mut candidates: Vec<String> = Vec::new();
        for candidate in [
            base.clone(),
            upper_first(&base),
            upper_first(&lower),
            title_case(&lower),
            base.to_uppercase(),
        ] {
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
        for candidate in &candidates {
            if let Some(idx) = self.zim.resolve_path(candidate)? {
                let entry = self.zim.get_entry(idx)?;
                let path = full_path(&entry);
                // `X/` entries are internal (embedded search indexes), not
                // articles; a query that spells one out must not surface it.
                if path.starts_with("X/") {
                    continue;
                }
                return Ok(Some((path, entry.title)));
            }
        }
        Ok(None)
    }

    /// Follow `path`'s redirect chain to its terminal entry (bounded, see
    /// `MAX_REDIRECT_HOPS`). Returns the terminal entry's full path and raw
    /// directory title (often empty - callers apply fallbacks); `None` when
    /// the path resolves to no entry. Search candidates resolve through
    /// this before dedupe so that a redirect reports the article it names
    /// - and dedupes against it.
    pub fn resolve_terminal(&self, path: &str) -> io::Result<Option<(String, String)>> {
        let Some(idx) = self.zim.resolve_path(path)? else {
            return Ok(None);
        };
        let entry = self.resolve_entry(idx)?;
        Ok(Some((full_path(&entry), entry.title)))
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

/// How the library was opened: one ZIM file, or a directory scanned for
/// ZIM files. The MCP tool set is shaped by it: single mode drops the
/// `zim` argument from zim_get/zim_get_section (there is nothing to name)
/// and has no zim_list; directory mode requires the argument and adds
/// zim_list plus an optional zim filter on zim_search.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Opened with a single ZIM file.
    Single,
    /// Opened with a directory, scanned recursively.
    Directory,
}

/// A set of ZIM archives: everything under a directory (`scan`), or a
/// single file (`single`).
pub struct ZimLibrary {
    pub root: PathBuf,
    pub archives: Vec<Arc<Archive>>,
    /// How the library was opened - the MCP tool set is shaped by it.
    pub mode: Mode,
}

impl ZimLibrary {
    /// Recursively scan `root` for ZIM files and open them. Files that fail
    /// to open are skipped with a warning; a directory holding none scans to
    /// an empty library - the pipelines report their own per-call errors
    /// (search's "no ZIM files with a Xapian full-text index"), so an empty
    /// directory is not fatal at scan time.
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
        Ok(ZimLibrary { root: root.to_path_buf(), archives, mode: Mode::Directory })
    }

    /// Open the single ZIM archive at `file` (`*.zim`, or the first
    /// `*.zimaa` chunk of a chunked archive) as a one-archive library. Errors
    /// are fatal (unlike `scan`, no skip-and-continue). The archive keeps its
    /// file name and `root` is its parent directory, so search results and
    /// the `zim` argument of zim_get/zim_get_section behave exactly as in a
    /// scanned directory.
    pub fn single(file: &Path) -> io::Result<ZimLibrary> {
        let name = file.file_name().and_then(zim_archive_name).ok_or_else(|| {
            io::Error::new(ErrorKind::InvalidInput, format!("not a ZIM file: {}", file.display()))
        })?;
        // "x.zimaa" normalizes to "x.zim", whose (nonexistent) path makes
        // Store::open pick up the chunked form x.zimaa, x.zimab, ...
        let zim = Zim::open(file.with_file_name(&name))?;
        let root = file.parent().unwrap_or(Path::new(".")).to_path_buf();
        Ok(ZimLibrary {
            root,
            archives: vec![Arc::new(Archive::new(name, zim))],
            mode: Mode::Single,
        })
    }

    /// The loaded archive whose name (relative to the ZIM directory) is `name`
    /// - the form search results report and zim_list will list. The name must
    /// stay inside the directory: a `..` component, an absolute path, or an
    /// empty string yields `None` (traversal outside the ZIM directory is
    /// refused), while symlinks are fine because the scan records the
    /// directory-entry name the symlink shows. `./` prefixes are trimmed.
    pub fn archive(&self, name: &str) -> Option<&Arc<Archive>> {
        let wanted = name.trim().trim_start_matches("./");
        if wanted.is_empty()
            || std::path::Path::new(wanted).is_absolute()
            || wanted.split('/').any(|part| part == "..")
        {
            return None;
        }
        self.archives.iter().find(|a| a.name == wanted)
    }

    /// The one archive of a single-file library; `None` for a scanned
    /// directory (its tools take the archive by name instead).
    pub fn single_archive(&self) -> Option<&Arc<Archive>> {
        if self.mode == Mode::Single {
            self.archives.first()
        } else {
            None
        }
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

    /// An `M/Language` metadata entry carrying `code` (the dirent layout a
    /// real archive uses), for pairing with a hand-built index to simulate
    /// any language. Append it to `content`; it is not an article.
    pub fn language_metadata_entry(code: &'static str) -> TestEntry {
        TestEntry { namespace: b'M', url: "Language", title: "", mime: 0, body: code.as_bytes() }
    }

    /// Build a minimal archive whose only cluster stores `blobs` under
    /// `compression` (4 = lzma, 5 = zstd), for compressed-read tests.
    fn build_cluster_archive(compression: u8, blobs: &[&[u8]]) -> Vec<u8> {
        // Cluster body: blob-offset table (32-bit offsets; the first entry's
        // value is the table's own byte size) + blob data.
        let mut data = Vec::new();
        let mut off = (blobs.len() + 1) as u32 * 4;
        for b in blobs {
            data.extend_from_slice(&off.to_le_bytes());
            off += b.len() as u32;
        }
        data.extend_from_slice(&off.to_le_bytes()); // end sentinel
        for b in blobs {
            data.extend_from_slice(b);
        }
        let mut cluster = vec![compression]; // info byte
        match compression {
            CLUSTER_LZMA => {
                std::io::Read::read_to_end(
                    &mut xz2::read::XzEncoder::new(data.as_slice(), 6),
                    &mut cluster,
                )
                .unwrap();
            }
            CLUSTER_ZSTD => cluster.extend_from_slice(&zstd::encode_all(data.as_slice(), 0).unwrap()),
            _ => cluster.extend_from_slice(&data),
        }
        // Header for one cluster and no directory entries; `read_blob_prefix`
        // only needs the cluster pointer list.
        let ptr_pos = 81u64; // header (80) + empty mime list
        let cluster_pos = ptr_pos + 8;
        let mut out = vec![0u8; 80];
        out[0..4].copy_from_slice(&ZIM_MAGIC.to_le_bytes());
        out[4..6].copy_from_slice(&6u16.to_le_bytes()); // major
        out[6..8].copy_from_slice(&1u16.to_le_bytes()); // minor
        out[28..32].copy_from_slice(&1u32.to_le_bytes()); // cluster count
        out[32..40].copy_from_slice(&ptr_pos.to_le_bytes()); // url ptr (empty)
        out[40..48].copy_from_slice(&ptr_pos.to_le_bytes()); // title ptr (empty)
        out[48..56].copy_from_slice(&ptr_pos.to_le_bytes()); // cluster ptr
        out[56..64].copy_from_slice(&80u64.to_le_bytes()); // mime list
        out[72..80].copy_from_slice(&(cluster_pos + cluster.len() as u64).to_le_bytes());
        out.push(0); // empty mime list
        out.extend_from_slice(&cluster_pos.to_le_bytes());
        out.extend_from_slice(&cluster);
        out.extend_from_slice(&[0u8; 16]);
        out
    }

    /// Assemble a complete in-memory ZIM with an optional full-text index;
    /// see `build_archive_indexes` for the general shape.
    pub fn build_archive(
        mime_types: &[&str],
        content: &[TestEntry],
        redirects: &[TestRedirect],
        main_page_content: usize,
        index: Option<&[u8]>,
    ) -> Vec<u8> {
        build_archive_indexes(mime_types, content, redirects, main_page_content, index, None)
    }

    /// Assemble a complete in-memory ZIM by writing it with the crate's own
    /// ZIM writer ([`crate::zimwrite::ZimCreator`]) — the tests therefore
    /// exercise reader and writer against each other instead of maintaining
    /// a second hand-rolled byte assembler. `fulltext_index`/`title_index`,
    /// when given, are stored as uncompressed `X/fulltext/xapian` /
    /// `X/title/xapian` content entries, like a real archive's embedded
    /// Xapian databases.
    ///
    /// Differences from a fully explicit byte assembly (both are libzim's
    /// own creator semantics, not accidents):
    /// - the archive gains the writer's standard `M/Counter` and
    ///   `X/listing/titleOrdered/v1` entries, and its MIME list is sorted;
    /// - a dirent title equal to the path is omitted from the bytes
    ///   (libzim's tiny-string packing), so such entries read back with an
    ///   empty title and callers apply their usual title fallbacks;
    /// - `main_page_content` is accepted for helper API compatibility but
    ///   unused: the synthetic fixtures never read the header's mainPage
    ///   field (the writer only sets it via a `W/mainPage` redirect, which
    ///   would change the dirent set the convert fixtures assert on).
    pub fn build_archive_indexes(
        mime_types: &[&str],
        content: &[TestEntry],
        redirects: &[TestRedirect],
        main_page_content: usize,
        fulltext_index: Option<&[u8]>,
        title_index: Option<&[u8]>,
    ) -> Vec<u8> {
        let _ = main_page_content;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("fixture.zim");
        let mut zc = crate::zimwrite::ZimCreator::new(&out).unwrap();
        for e in content {
            zc.add_item_in_namespace(
                e.namespace,
                e.url,
                e.title,
                mime_types[e.mime as usize],
                // Store fixture bodies uncompressed, like the hand-rolled
                // builder did; the reader handles both cluster kinds.
                false,
                e.namespace == b'C',
                e.body.to_vec(),
            )
            .unwrap();
        }
        for r in redirects {
            // Test redirects always point at C-namespace articles: that is
            // the only shape `ZimCreator::add_redirection` writes (libzim's
            // user-facing redirections are C-only too).
            assert_eq!(r.namespace, b'C', "test redirects are C-namespace");
            zc.add_redirection(r.url, r.title, content[r.target_content].url, true)
                .unwrap();
        }
        if let Some(ix) = fulltext_index {
            zc.add_xapian_index_bytes("fulltext/xapian", ix).unwrap();
        }
        if let Some(ix) = title_index {
            zc.add_xapian_index_bytes("title/xapian", ix).unwrap();
        }
        zc.finish().unwrap();
        std::fs::read(&out).unwrap()
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
    fn get_article_content() {
        let (a, _f) = sample_archive();
        let art = a.get_article("Banana").unwrap();
        assert_eq!(art.bytes, b"<html><body><h2>Bananas</h2><p>A delicious fruit.</p></body></html>");
        // The dirent title "Banana" equals the path, so the writer's
        // tiny-string packing omits it; `get_article` falls back to the
        // full path (same value a modern openZIM archive yields).
        assert_eq!(art.title, "C/Banana");
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
        // Terminal article C/Apple: its dirent title ("Apple") equals the
        // path and is omitted, so the full path is the reported title.
        assert_eq!(art.title, "C/Apple");
        assert_eq!(art.full_path, "C/Apple");
    }
    #[test]
    fn path_resolution_decorations() {
        let (a, _f) = sample_archive();
        // In-page href style with a leading "./" resolves.
        assert!(a.get_article("./Apple").is_ok());
        // A namespace-only path must not silently resolve to an article
        // that happens to be named like the namespace.
        assert!(a.get_article("C/").is_err());
        // A trailing slash is a different (nonexistent) path, not the
        // article without it.
        assert!(a.get_article("Apple/").is_err());
        assert!(a.get_article("C/Apple/").is_err());
    }

    #[test]
    fn language_metadata_first_code_lowercased_or_none() {
        // With metadata: the first code, trimmed and lowercased (multi-code
        // values occur; separators split them).
        let content = [
            TestEntry {
                namespace: b'C',
                url: "Apple",
                title: "Apple",
                mime: 0,
                body: b"<html><body><h1>Apple</h1><p>An apple a day.</p></body></html>",
            },
            language_metadata_entry("FRA, eng"),
        ];
        let (z, _f) = open_bytes(&build_archive(&["text/html"], &content, &[], 0, None));
        let a = Archive::new("fr.zim".to_string(), z);
        assert_eq!(a.language().as_deref(), Some("fra"));
        // Without metadata (the sample archive carries none): None - search
        // defaults such archives to English stemming.
        let (a, _f) = sample_archive();
        assert_eq!(a.language(), None);
    }

    #[test]
    fn preview_reads_only_prefix() {
        let (a, _f) = sample_archive();
        let (title, mime, bytes) = a.article_preview("Apple", 64).unwrap().unwrap();
        // `article_preview` reports the RAW dirent title; the dirent title
        // "Apple" equals the path and is omitted by the writer.
        assert_eq!(title, "");
        assert_eq!(mime.as_deref(), Some("text/html"));
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn compressed_cluster_blob_prefix_reads() {
        let blobs: &[&[u8]] = &[
            b"alpha-first-blob",
            b"middle blob payload, long enough to make offsets interesting",
            b"zeta-last-blob",
        ];
        for compression in [CLUSTER_LZMA, CLUSTER_ZSTD] {
            let (z, _f) = open_bytes(&build_cluster_archive(compression, blobs));
            let (len1, len2) = (blobs[1].len() as u64, blobs[2].len() as u64);
            // max < len: truncated prefix of the first blob.
            assert_eq!(z.read_blob_prefix(0, 0, 5).unwrap(), b"alpha");
            // max = len, max > len, and a middle blob.
            assert_eq!(z.read_blob_prefix(0, 1, len1).unwrap(), blobs[1]);
            assert_eq!(z.read_blob_prefix(0, 2, len2 + 10).unwrap(), blobs[2]);
            assert_eq!(z.read_blob_prefix(0, 1, 7).unwrap(), b"middle ");
            // Unbounded reads return the whole blob.
            assert_eq!(z.read_blob(0, 0).unwrap(), blobs[0]);
            // A blob index past the table is rejected.
            assert_eq!(
                z.read_blob_prefix(0, 3, 4).unwrap_err().to_string(),
                "blob index out of bounds"
            );
        }
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

    #[test]
    fn single_file_library() {
        let dir = tempfile::tempdir().unwrap();
        let content = [TestEntry {
            namespace: b'C',
            url: "Apple",
            title: "Apple",
            mime: 0,
            body: b"<html><body><h1>Apple</h1><p>An apple a day keeps the doctor away.</p></body></html>",
        }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, None);
        let file = dir.path().join("sample.zim");
        std::fs::write(&file, &bytes).unwrap();
        let lib = ZimLibrary::single(&file).unwrap();
        assert_eq!(lib.archives.len(), 1);
        assert_eq!(lib.archives[0].name, "sample.zim");
        assert_eq!(lib.root, dir.path());
        let art = lib.archives[0].get_article("C/Apple").unwrap();
        // Dirent title "Apple" == path (omitted): the full path is reported.
        assert_eq!(art.title, "C/Apple");
    }

    #[test]
    fn single_file_accepts_chunked_first_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let content = [TestEntry {
            namespace: b'C',
            url: "Apple",
            title: "Apple",
            mime: 0,
            body: b"<html><body><p>An apple a day.</p></body></html>",
        }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, None);
        let file = dir.path().join("chunk.zimaa");
        std::fs::write(&file, &bytes).unwrap();
        let lib = ZimLibrary::single(&file).unwrap();
        // The chunk argument is normalized to the archive's `*.zim` name.
        assert_eq!(lib.archives[0].name, "chunk.zim");
    }

    #[test]
    fn single_file_rejects_non_zim() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.txt");
        std::fs::write(&file, b"definitely not a ZIM archive").unwrap();
        assert!(ZimLibrary::single(&file).is_err());
    }

    #[test]
    fn archive_name_resolution() {
        let dir = tempfile::tempdir().unwrap();
        let content = [TestEntry {
            namespace: b'C',
            url: "Apple",
            title: "Apple",
            mime: 0,
            body: b"<html><body><h1>Apple</h1><p>An apple a day.</p></body></html>",
        }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, None);
        std::fs::write(dir.path().join("a.zim"), &bytes).unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/b.zim"), &bytes).unwrap();
        let lib = ZimLibrary::scan(dir.path()).unwrap();
        assert_eq!(lib.archives.len(), 2);

        // Plain and nested relative names resolve to the scanned archive.
        assert!(Arc::ptr_eq(lib.archive("a.zim").unwrap(), &lib.archives[0]));
        assert!(Arc::ptr_eq(lib.archive("sub/b.zim").unwrap(), &lib.archives[1]));
        // A "./" prefix is trimmed.
        assert!(Arc::ptr_eq(lib.archive("./a.zim").unwrap(), &lib.archives[0]));
        // Names leaving the ZIM directory are refused, unknown ones too.
        assert!(lib.archive("../a.zim").is_none());
        assert!(lib.archive("/etc/a.zim").is_none());
        assert!(lib.archive("").is_none());
        assert!(lib.archive("nope.zim").is_none());
    }

    #[test]
    fn scan_of_empty_directory_is_ok_and_empty() {
        // A scanned directory may hold zero ZIM files: the scan succeeds
        // with an empty library, and the pipelines report their own per-call
        // errors for it (only `single` stays strict - it names one file).
        let dir = tempfile::tempdir().unwrap();
        let lib = ZimLibrary::scan(dir.path()).unwrap();
        assert!(lib.archives.is_empty());
        assert_eq!(lib.mode, Mode::Directory);
        assert!(lib.single_archive().is_none());
    }

}
