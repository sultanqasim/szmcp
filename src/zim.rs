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
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

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
            5 => Box::new(zstd::stream::read::Decoder::with_buffer(body)?),
            // liblzma's auto decoder transparently handles the LZMA
            // streams ZIM clusters store.
            4 => Box::new(xz2::read::XzDecoder::new(body)),
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
        let sz = if (info & 0x10) != 0 { 8 } else { 4 }; // 64-bit blob offsets
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
    /// `idx`, preferring the zero-copy path (opening the in-file glass
    /// database at the blob's offset), falling back to a temp-file copy.
    /// `what` names the index in errors ("full-text", "title"). Never yields
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
                    tracing::warn!(
                        "reading the Language metadata of {} failed ({e}); queries assume english stems",
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
                        tracing::warn!(
                            "opening the title index of {} failed ({e}); ranking by full text only",
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
            4 => {
                std::io::Read::read_to_end(
                    &mut xz2::read::XzEncoder::new(data.as_slice(), 6),
                    &mut cluster,
                )
                .unwrap();
            }
            5 => cluster.extend_from_slice(&zstd::encode_all(data.as_slice(), 0).unwrap()),
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

    /// Assemble a complete in-memory ZIM. `fulltext_index`/`title_index`,
    /// when given, are stored as uncompressed `X/fulltext/xapian` /
    /// `X/title/xapian` content entries (single-blob clusters), like a real
    /// archive's embedded Xapian databases.
    pub fn build_archive_indexes(
        mime_types: &[&str],
        content: &[TestEntry],
        redirects: &[TestRedirect],
        main_page_content: usize,
        fulltext_index: Option<&[u8]>,
        title_index: Option<&[u8]>,
    ) -> Vec<u8> {
        let has_index = fulltext_index.is_some();
        let has_title_index = title_index.is_some();
        let entry_count = (content.len() + redirects.len() + has_index as usize + has_title_index as usize) as u32;
        let cluster_count = (content.len() + has_index as usize + has_title_index as usize) as u32;

        let mut mime_blob = Vec::new();
        for m in mime_types {
            push_zstring(&mut mime_blob, m);
        }
        mime_blob.push(0);

        enum Logical<'a> {
            Content { e: &'a TestEntry, cluster: u32 },
            Redirect { r: &'a TestRedirect },
            Index { url: &'static str, cluster: u32 },
        }
        let mut logical: Vec<(u8, &str, Logical)> = Vec::new();
        for (ci, e) in content.iter().enumerate() {
            logical.push((e.namespace, e.url, Logical::Content { e, cluster: ci as u32 }));
        }
        for r in redirects {
            logical.push((r.namespace, r.url, Logical::Redirect { r }));
        }
        let mut next_cluster = content.len() as u32;
        if let Some(_ix) = fulltext_index {
            logical.push((b'X', "fulltext/xapian", Logical::Index { url: "fulltext/xapian", cluster: next_cluster }));
            next_cluster += 1;
        }
        if let Some(_ix) = title_index {
            logical.push((b'X', "title/xapian", Logical::Index { url: "title/xapian", cluster: next_cluster }));
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
                Logical::Index { url, cluster } => {
                    b.extend_from_slice(&0u16.to_le_bytes());
                    b.push(0);
                    b.push(*ns);
                    b.extend_from_slice(&0u32.to_le_bytes());
                    b.extend_from_slice(&cluster.to_le_bytes());
                    b.extend_from_slice(&0u32.to_le_bytes());
                    push_zstring(&mut b, url);
                    push_zstring(&mut b, "Xapian index");
                }
            }
            entry_bodies.push(b);
        }

        let mut clusters: Vec<Vec<u8>> = content.iter().map(|e| build_cluster(e.body)).collect();
        if let Some(ix) = fulltext_index {
            clusters.push(build_cluster(ix));
        }
        if let Some(ix) = title_index {
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
    fn fulltext_index_absent_reports_none() {
        let (a, _f) = sample_archive();
        assert!(a.zim.open_fulltext_xapian().unwrap().is_none());
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
        assert_eq!(title, "Apple");
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
        for compression in [4u8, 5] {
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
        assert_eq!(art.title, "Apple");
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

    #[test]
    fn library_mode_matches_how_it_was_opened() {
        // The mode records the launch shape (one file vs. a scanned
        // directory), and single_archive hands out the one archive only in
        // single mode - the MCP tool set is built on both facts.
        let dir = tempfile::tempdir().unwrap();
        let content = [TestEntry {
            namespace: b'C',
            url: "Apple",
            title: "Apple",
            mime: 0,
            body: b"<html><body><h1>Apple</h1><p>An apple a day.</p></body></html>",
        }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, None);

        let file = dir.path().join("one.zim");
        std::fs::write(&file, &bytes).unwrap();
        let single = ZimLibrary::single(&file).unwrap();
        assert_eq!(single.mode, Mode::Single);
        assert!(Arc::ptr_eq(single.single_archive().unwrap(), &single.archives[0]));

        std::fs::write(dir.path().join("two.zim"), &bytes).unwrap();
        let scanned = ZimLibrary::scan(dir.path()).unwrap();
        assert_eq!(scanned.mode, Mode::Directory);
        assert!(scanned.single_archive().is_none());
    }
}
