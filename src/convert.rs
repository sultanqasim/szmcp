//! `szmcp convert` — a Rust port of `wikizim_parser/zim2zim.py`: turn a
//! Kiwix HTML ZIM into a ZIM of Markdown articles with fresh fulltext and
//! title Xapian indexes, built exactly as libzim 9.8.2 builds them (see
//! `mcp_stuff/convert_notes.md`).
//!
//! Shape: a parallel classification pre-pass over the dirents in path order
//! (headers only; HTML articles also record their path/title and blob
//! coordinates), then a parallel conversion walk in ascending cluster order
//! — dirents are path-sorted but clusters were appended in completion
//! order, so cluster order decodes each compressed cluster exactly once.
//! Per-entry 12-byte records carry the walk's outcomes; membership is
//! arithmetic over their flags, and a single-threaded finalize streams the
//! member dirents in source order and builds the title-ordered listing.
//! Every entry is converted (text/html → markdown), recreated (redirect
//! resolving to a converted article) or skipped and tallied by MIME.
//! Indexing runs on a dedicated thread fed by a bounded document queue, so
//! document order inside the Xapian databases is nondeterministic (FIFO of
//! worker completion — the parallel adds race like libzim's workers); both
//! sides are equivalent as sets keyed by document data.

use crate::zim::{Target, Zim};
use crate::zimcommon::MIME_REDIRECT;
use crate::zimwrite::{BlobRef, DirentOut, ZimCreator};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;
use xapian2::{
    Document, Stem, StemStrategy, TermGenerator, WritableDatabase,
    tg_flags, wdb_flags,
};

/// The longest title libzim's title indexer can safely handle: a title of
/// at most 238 UTF-8 bytes can never trip the title-indexer abort (which
/// fires when >= 239 title bytes end up unindexed; the loss can never exceed
/// the title's own byte length).
const TITLE_MAX_BYTES: usize = 238;

/// libzim's title-index anchor term (constants.h ANCHOR_TERM).
const ANCHOR_TERM: &str = "0posanchor ";

/// The M/ metadata keys copied verbatim from the source archive.
const METADATA_KEYS: [&str; 7] = [
    "Name",
    "Title",
    "Language",
    "Description",
    "Date",
    "Publisher",
    "Creator",
];

// ---------------------------------------------------------------------------
// Ported helpers (zim2zim.py)
// ---------------------------------------------------------------------------

/// `zim2zim._trim_title`: cut `title` to at most [`TITLE_MAX_BYTES`] UTF-8
/// bytes, dropping a trailing partial character (python's
/// `encode()[:238].decode("utf-8", "ignore")`).
fn trim_title(title: &str) -> String {
    if title.len() <= TITLE_MAX_BYTES {
        return title.to_string();
    }
    let mut end = TITLE_MAX_BYTES;
    while end > 0 && !title.is_char_boundary(end) {
        end -= 1;
    }
    title[..end].to_string()
}

/// The effective title of a source entry: the dirent title, else the URL
/// with underscores as spaces (zim2zim's fallback).
fn entry_title(dirent_title: &str, item_path: &str) -> String {
    let title = if dirent_title.is_empty() {
        item_path.replace('_', " ")
    } else {
        dirent_title.to_string()
    };
    if title.len() > TITLE_MAX_BYTES {
        trim_title(&title)
    } else {
        title
    }
}

/// `zim2zim._is_hatnote_para`: a standalone paragraph entirely wrapped in
/// single asterisks (html2md renders hatnotes exactly like that, right
/// after the title line or another hatnote). Bullets ("* item") and bold
/// ("**x**") don't match.
fn is_hatnote_para(para: &str) -> bool {
    let t = para.trim();
    let chars: Vec<char> = t.chars().collect();
    chars.len() > 2
        && chars[0] == '*'
        && chars[chars.len() - 1] == '*'
        // t[1] not in " *"
        && chars[1] != ' '
        && chars[1] != '*'
        && !chars[1..chars.len() - 1].contains(&'*')
}

/// `zim2zim._intro_for_index`: the text to full-text-index for an article -
/// the leading title line (always kept) plus the intro (everything up to the
/// first "## " heading) as paragraphs, with the leading run of hatnote
/// paragraphs removed. Without a "## " heading the whole body after the
/// title line is the intro.
fn intro_for_index(md: &str) -> String {
    let mut lines = md.split('\n');
    let head = lines.next().unwrap_or("");
    let mut body: Vec<&str> = lines.collect();
    for (i, line) in body.iter().enumerate() {
        if line.starts_with("## ") {
            body.truncate(i);
            break;
        }
    }
    let mut paras: Vec<String> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for line in body {
        if !line.trim().is_empty() {
            cur.push(line);
        } else if !cur.is_empty() {
            paras.push(cur.join("\n"));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        paras.push(cur.join("\n"));
    }
    while paras.first().map(|p| is_hatnote_para(p)).unwrap_or(false) {
        paras.remove(0);
    }
    if paras.is_empty() {
        head.to_string()
    } else {
        format!("{head}\n\n{}", paras.join("\n\n"))
    }
}

/// `zim2zim._indexing_language`: one Xapian-indexing language code from a raw
/// metadata value - the first token of the `[,;\s]`-split value, BCP-47
/// subtags stripped, lowercased ("fr-FR" -> "fr"); "eng" when nothing is
/// left. libzim picks its stemmer by exactly this code.
fn indexing_language(raw: Option<&str>) -> String {
    const DEFAULT: &str = "eng";
    let Some(raw) = raw else { return DEFAULT.to_string() };
    let tok = raw
        .trim()
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .find(|t| !t.is_empty())
        .unwrap_or("");
    let tok = tok.split('-').next().unwrap_or("").trim().to_lowercase();
    if tok.is_empty() {
        DEFAULT.to_string()
    } else {
        tok
    }
}

// ---------------------------------------------------------------------------
// Source reading helpers
// ---------------------------------------------------------------------------

/// The raw `M/Language` metadata of the source (verbatim, e.g. "fra" or a
/// comma list), or `None` when the archive carries none.
fn raw_language_metadata(z: &Zim) -> Option<String> {
    let idx = z.find_entry(b'M', "Language").ok()??;
    let entry = z.get_entry(idx).ok()?;
    let Target::Cluster(cluster, blob) = entry.target else {
        return None;
    };
    let bytes = z.read_blob(cluster, blob).ok()?;
    let value = String::from_utf8(bytes).ok()?;
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

/// The source's main entry path, replicating
/// `arch.main_entry.get_item().path`: the header's `mainPage` field
/// (`zim::ZimHeader.main_page`) followed through its redirect chain to the
/// terminal entry.
fn main_entry_path(z: &Zim) -> Option<String> {
    let main_page = z.header.main_page;
    if main_page == u32::MAX {
        return None;
    }
    let mut entry = z.get_entry(main_page).ok()?;
    for _ in 0..64 {
        match entry.target {
            Target::Redirect(next) => entry = z.get_entry(next).ok()?,
            _ => return Some(entry.url),
        }
    }
    None
}

/// The display name for a skipped entry's MIME id: the MIME string from the
/// archive's mime list, or "unknown/<id>".
fn skip_mime_name(z: &Zim, mime: u16) -> String {
    match z.mime_type(mime) {
        Some(s) => s.to_string(),
        None => format!("unknown/{mime}"),
    }
}

// ---------------------------------------------------------------------------
// Xapian index documents (libzim's xapianWorker.cpp / xapianIndexer.cpp)
// ---------------------------------------------------------------------------

/// The Xapian indexing language of a conversion: the RAW normalized code as
/// configured (stored as the index's `language` metadata, like libzim stores
/// the raw code) plus the ICU-primary language Xapian actually stems with
/// (`None` when `Stem::new` fails, mirroring libzim's try/catch - no
/// stemming for unsupported codes).
struct IndexLang {
    raw: String,
    stemmer: Option<String>,
}

impl IndexLang {
    fn new(raw: String) -> Self {
        let mapped = crate::lang_map::primary_language(&raw).to_string();
        let stemmer = Stem::new(&mapped).ok().map(|_| mapped);
        Self { raw, stemmer }
    }

    /// Set the stemmer + stemming strategy on `indexer` the way libzim does
    /// for `mode` ("fulltext" = STEM_ALL, "title" = STEM_SOME): both inside
    /// one try block, so a failed stemmer leaves the strategy untouched
    /// (Xapian's own STEM_SOME default, a no-op without a stemmer).
    fn set_stemmer(&self, indexer: &mut TermGenerator, all: bool) -> Result<(), String> {
        if let Some(lang2) = &self.stemmer {
            indexer
                .set_stemmer(lang2)
                .map_err(|e| format!("stemmer {lang2}: {e}"))?;
            let strategy = if all { StemStrategy::All } else { StemStrategy::Some };
            indexer.set_stemming_strategy(strategy).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// One fulltext-index document, the way libzim's IndexTask::run builds it
/// from custom IndexData: data = "C/"+path, value 0 = folded title, value 1
/// = wordcount string, terms = STEM_ALL over the folded (accent-folding is
/// our job; libzim indexes custom IndexData verbatim) content without
/// positions, then the folded title again at
/// `boost = folded_content.len()/500 + 1` (getTitleBoostFactor).
///
/// The document builds WITHOUT the database: the TermGenerator/Document FFI
/// never touches the WDB pointer (exactly libzim's own worker design, which
/// indexes into Document objects on worker threads), so pass-1 workers build
/// outside any lock and ship the finished documents to the indexer thread.
fn build_fulltext_document(
    lang: &IndexLang,
    path: &str,
    folded_title: &str,
    folded_content: &str,
) -> Result<Document, String> {
    let mut indexer = TermGenerator::new().map_err(|e| e.to_string())?;
    indexer.set_flags(tg_flags::FLAG_NGRAMS).map_err(|e| e.to_string())?;
    lang.set_stemmer(&mut indexer, true)?;
    // No stopper: zim2zim's 3-letter language codes never load a stopword
    // resource in libzim either ("stopwords/<raw>" misses) - matching
    // behavior; a 2-letter code would (documented divergence).

    let mut doc = Document::new().map_err(|e| e.to_string())?;
    indexer.set_document(&doc).map_err(|e| e.to_string())?;
    doc.set_data(format!("C/{path}")).map_err(|e| e.to_string())?;
    doc.set_value(0, folded_title).map_err(|e| e.to_string())?;
    let wordcount = folded_content.split_whitespace().count();
    doc.set_value(1, wordcount.to_string()).map_err(|e| e.to_string())?;

    if !folded_content.is_empty() {
        indexer
            .index_text_without_positions(folded_content)
            .map_err(|e| e.to_string())?;
    }
    if !folded_title.is_empty() {
        let boost = folded_content.len() / 500 + 1;
        indexer
            .index_text_without_positions_with_wdf(folded_title, boost as u32)
            .map_err(|e| e.to_string())?;
    }
    Ok(doc)
}

/// One title-index document, the way libzim's XapianIndexer::indexTitle does:
/// data = "C/"+path, value 0 = the RAW title (NOT folded), value 1 = the
/// redirect target path (or the article's own path); the accent-folded title
/// indexed WITH positions behind libzim's anchor term, STEM_SOME,
/// FLAG_NGRAMS, a 240-character word cap. A title made solely of non-word
/// characters leaves only the anchor term: it is removed and the whole title
/// added as one term when it fits (libzim's collapse). Built WITHOUT the
/// database, like [`build_fulltext_document`].
fn build_title_document(
    lang: &IndexLang,
    path: &str,
    title: &str,
    target_path: Option<&str>,
) -> Result<Document, String> {
    let mut indexer = TermGenerator::new().map_err(|e| e.to_string())?;
    indexer.set_max_word_length(240).map_err(|e| e.to_string())?;
    indexer.set_flags(tg_flags::FLAG_NGRAMS).map_err(|e| e.to_string())?;
    lang.set_stemmer(&mut indexer, false)?;

    let mut doc = Document::new().map_err(|e| e.to_string())?;
    indexer.set_document(&doc).map_err(|e| e.to_string())?;
    doc.set_data(format!("C/{path}")).map_err(|e| e.to_string())?;
    doc.set_value(0, title).map_err(|e| e.to_string())?;
    match target_path {
        Some(target) => doc.set_value(1, target).map_err(|e| e.to_string())?,
        None => doc.set_value(1, path).map_err(|e| e.to_string())?,
    }

    let unaccented_title = crate::search::fold_accents(title);
    if !unaccented_title.is_empty() {
        let anchored_title = format!("{ANCHOR_TERM}{unaccented_title}");
        indexer.index_text(&anchored_title, 1).map_err(|e| e.to_string())?;
        // libzim aborts the whole build when >= 239 title bytes end up
        // unindexed (TitleIndexingError). zim2zim pre-trims titles to 238
        // bytes, which can never fire this - surface it as a hard error if
        // it ever would.
        let indexed = doc.indexed_text_size().map_err(|e| e.to_string())? as usize;
        if anchored_title.len() >= indexed + 240 {
            return Err(format!(
                "title indexing would lose too much data: anchored title \
                 {anchored_title:?} ({} bytes), {indexed} indexed bytes",
                anchored_title.len()
            ));
        }
        if doc.termlist_count() == 1 {
            // Only the anchor term was added: the title carries no word
            // characters, so index the entire title as a single term. libzim
            // removes *termlist_begin() - the STORED anchor term, "0posanchor"
            // without the trailing space of the ANCHOR_TERM literal (a space
            // is a word separator, so it never reaches a term).
            let first_term = doc.first_term().map_err(|e| e.to_string())?;
            doc.remove_term(&first_term).map_err(|e| e.to_string())?;
            if unaccented_title.len() <= 240 {
                // libzim's add_term default: wdf increment 1.
                doc.add_term(&unaccented_title, 1).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(doc)
}

/// Create one throwaway Xapian database with libzim's indexing prelude
/// (xapianIndexer.cpp): DB_CREATE_OR_OVERWRITE | DB_NO_TERMLIST plus the
/// metadata pairs libzim's indexer records. `kind`/`valuesmap` are the only
/// parts that differ between the fulltext and the title database.
fn create_index_wdb(
    path: &Path,
    kind: &str,
    valuesmap: &str,
    language: &str,
) -> Result<WritableDatabase, String> {
    let mut wdb = WritableDatabase::create_with_flags(
        path,
        wdb_flags::DB_CREATE_OR_OVERWRITE | wdb_flags::DB_NO_TERMLIST,
    )
    .map_err(|e| format!("{kind} index: {e}"))?;
    for (key, value) in [
        ("valuesmap", valuesmap),
        ("kind", kind),
        ("data", "fullPath"),
        ("language", language),
        ("stopwords", ""),
    ] {
        wdb.set_metadata(key, value)
            .map_err(|e| format!("{kind} index metadata {key}: {e}"))?;
    }
    Ok(wdb)
}

// ---------------------------------------------------------------------------
// Source asset copying
// ---------------------------------------------------------------------------

/// Copy the core string metadata verbatim (`zim2zim._copy_metadata`): each
/// present, non-empty key of `METADATA_KEYS` becomes an
/// `M/<name>` metadata entry with libzim/python's default
/// "text/plain;charset=UTF-8" mimetype. The `Language` metadata is copied
/// VERBATIM; only when the source declares none is the normalized indexing
/// language written (the ZIM spec requires the key to exist). Returns the
/// effective language (the copied value or the default).
fn copy_metadata(
    z: &Zim,
    creator: &mut ZimCreator,
    default_language: &str,
) -> Result<String, String> {
    let mut language = default_language.to_string();
    let mut copied_language = false;
    for key in METADATA_KEYS {
        let Some(idx) = (match z.find_entry(b'M', key) {
            Ok(idx) => idx,
            Err(_) => continue,
        }) else {
            continue;
        };
        let entry = match z.get_entry(idx) {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let Target::Cluster(cluster, blob) = entry.target else {
            continue;
        };
        let Ok(bytes) = z.read_blob(cluster, blob) else { continue };
        let Ok(value) = String::from_utf8(bytes) else { continue };
        // Skip missing/whitespace-only values; the value itself is copied
        // unmodified (python checks `val.strip()` and writes `val`).
        if value.trim().is_empty() {
            continue;
        }
        if key == "Language" {
            language = value.clone();
            copied_language = true;
        }
        creator
            .add_metadata(key, value.as_bytes(), "text/plain;charset=UTF-8")
            .map_err(|e| format!("metadata {key:?}: {e}"))?;
    }
    if !copied_language {
        creator
            .add_metadata("Language", default_language.as_bytes(), "text/plain;charset=UTF-8")
            .map_err(|e| format!("metadata 'Language': {e}"))?;
    }
    Ok(language)
}

/// Copy the source's 48x48 illustration (`M/Illustration_48x48@1`) as the
/// output's illustration; a warning when the source has none.
fn copy_illustration(z: &Zim, creator: &mut ZimCreator) -> Result<(), String> {
    let idx = z.find_entry(b'M', "Illustration_48x48@1").ok().flatten();
    let Some(idx) = idx else {
        eprintln!("warning: illustration: the source archive has no 48x48 illustration");
        return Ok(());
    };
    let entry = z.get_entry(idx).map_err(|e| format!("illustration: {e}"))?;
    let Target::Cluster(cluster, blob) = entry.target else {
        eprintln!("warning: illustration: the source's 48x48 illustration has no content");
        return Ok(());
    };
    let bytes = z.read_blob(cluster, blob).map_err(|e| format!("illustration: {e}"))?;
    creator.add_illustration(48, &bytes).map_err(|e| format!("illustration: {e}"))
}

// ---------------------------------------------------------------------------
// Parallel-pass infrastructure
// ---------------------------------------------------------------------------

/// Worker-pulled chunk size of the classification pre-pass: fixed ranges
/// keep each worker's work contiguous while the shared atomic counter
/// hands chunks out dynamically, so heterogeneous cores stay busy.
const CHUNK_ENTRIES: u64 = 512;

/// WDB commit pacing: glass buffers every uncommitted change in RAM and
/// only writes on commit, so this bounds the uncommitted glass buffers
/// (tens of KB per document) while keeping the number of flush/merge
/// passes — and their write amplification — low.
const COMMIT_EVERY: u64 = 250_000;

/// Progress-line interval (carriage-return overwrite, zim2zim style).
const STATUS_EVERY: u64 = 1_000;

/// Record flags: bit 0 = the article was converted and its blob entered an
/// output cluster; bit 1 = that blob lives in a compressed cluster; bit 2 =
/// a redirect whose resolved terminal is in `a`; bit 3 = that terminal's
/// MIME is text/html; bit 4 = a no-content article (title-only, or title
/// plus one bare wikilink line), indexed like a redirect.
const REC_ARTICLE: u8 = 1;
const REC_COMPRESS: u8 = 2;
const REC_REDIRECT: u8 = 4;
const REC_TARGET_HTML: u8 = 8;
/// A no-content article: indexed like a redirect (see the flags above).
const REC_NOCONTENT: u8 = 16;

/// One per-entry record (12 bytes), a union discriminated by the flags byte
/// — an entry is either an article or a redirect, never both: an article
/// stores its blob's cluster GENERATION in `a` and the blob index in `b`; a
/// redirect stores its terminal source index in `a` (u32::MAX when the
/// chain died or cycled) and never touches `b`. Skipped entries stay
/// all-zero; the flags gate every read after the join, so no sentinels.
/// Atomics because the vector is shared; each slot is written once by the
/// worker that owns it (disjoint slots) and read after the join.
#[derive(Default)]
struct RecSlot {
    a: AtomicU32,
    b: AtomicU32,
    flags: AtomicU8,
}

/// One classified HTML article: blob coordinates plus the (offset, len)
/// of its path and title inside the pre-pass string arena. Sorting by
/// (cluster, blob, idx) groups one task per cluster in file order.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct ArticleCoord {
    cluster: u32,
    blob: u32,
    idx: u32,
    path: (u32, u32),
    title: (u32, u32),
}

/// The pre-pass's collector: the [`ArticleCoord`]s plus the string arena
/// their path/title slices point into (raw bytes, lengths recorded), one
/// mutex over both so coordinates and strings stay consistent. The arena
/// (~0.5-1 GB at 7M articles) buys the walk zero dirent reads: the
/// cluster-order walk would otherwise re-read each article's dirent from
/// its scattered position across the dirent area, page-faulting 16-64 KB
/// per article under memory pressure.
struct PrePass {
    arena: Vec<u8>,
    coords: Vec<ArticleCoord>,
}

/// The NUL-terminated title at `off` in the finalize listing's bump arena
/// (the terminator separates rows; titles derive from dirent titles and
/// paths, which cannot contain NUL).
fn arena_title(arena: &[u8], off: usize) -> &[u8] {
    let end = arena[off..]
        .iter()
        .position(|&b| b == 0)
        .expect("row title terminator")
        + off;
    &arena[off..end]
}

/// Output entry index of a member source index: cumulative popcounts per
/// 64-bit word (~3 MB at 50M entries) plus a popcount tail.
struct Rank<'a> {
    words: &'a [u64],
    cum: Vec<u32>,
}

impl<'a> Rank<'a> {
    fn new(words: &'a [u64]) -> Self {
        let mut cum = Vec::with_capacity(words.len());
        let mut total = 0u32;
        for w in words {
            cum.push(total);
            total += w.count_ones();
        }
        Rank { words, cum }
    }

    fn rank(&self, i: usize) -> u32 {
        self.cum[i / 64] + (self.words[i / 64] & ((1u64 << (i % 64)) - 1)).count_ones()
    }
}

/// Counters and tallies shared by the worker threads (atomic because every
/// worker fetch-adds; the skip tally is indexed by MIME id, so workers
/// never contend on one lock). `next` hands out
/// the work (512-entry chunks in the classification pre-pass, one cluster
/// task per fetch-add in the conversion walk; reset between the phases),
/// `processed` counts classified entries, `redirects` counts the redirect
/// records stored by the pre-pass, and `articles_done`/`articles_total` is
/// the walk's per-article progress.
/// `ft_docs` IS the converted-article count: every successful conversion
/// adds exactly one fulltext document except no-content pages (they are
/// indexed like redirects), and only the indexer thread
/// increments `ft_docs`/`ti_docs`.
#[derive(Default)]
struct Shared {
    next: AtomicU64,
    processed: AtomicU64,
    redirects: AtomicU64,
    articles_done: AtomicU64,
    articles_total: u64,
    failed: AtomicU64,
    md_bytes: AtomicU64,
    // Arc so the dedicated indexer thread (plain-spawned, not scoped) can
    // own clones by value; it is the only writer of both counters.
    ft_docs: Arc<AtomicU64>,
    ti_docs: Arc<AtomicU64>,
    /// Skipped-entry counts indexed by MIME id, sized to the archive's MIME
    /// list; `skipped_unknown` counts entries whose id is past the list.
    skipped: Vec<AtomicU64>,
    skipped_unknown: AtomicU64,
}

/// Run `worker` on `threads` scoped threads; each pulls fixed
/// [`CHUNK_ENTRIES`] chunks from `next` until the range `end` is
/// exhausted (the classification pre-pass's shape). A worker's first error
/// stops it and surfaces to the caller.
fn run_chunk_workers(
    threads: usize,
    next: &AtomicU64,
    end: u32,
    worker: impl Fn(u32, u32) -> Result<(), String> + Sync,
) -> Result<(), String> {
    let error: Mutex<Option<String>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c * CHUNK_ENTRIES >= end as u64 {
                    return;
                }
                let start = (c * CHUNK_ENTRIES) as u32;
                let stop = ((c + 1) * CHUNK_ENTRIES).min(end as u64) as u32;
                if let Err(e) = worker(start, stop) {
                    let mut g = error.lock().unwrap();
                    if g.is_none() {
                        *g = Some(e);
                    }
                    return;
                }
            });
        }
    });
    match error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// [`run_chunk_workers`]'s task-based sibling for the conversion walk:
/// each worker pulls ONE task from `next` until `tasks` is exhausted. A
/// worker's first error stops it and surfaces to the caller.
fn run_task_workers<T: Sync>(
    threads: usize,
    next: &AtomicU64,
    tasks: &[T],
    worker: impl Fn(&T) -> Result<(), String> + Sync,
) -> Result<(), String> {
    let error: Mutex<Option<String>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c as usize >= tasks.len() {
                    return;
                }
                if let Err(e) = worker(&tasks[c as usize]) {
                    let mut g = error.lock().unwrap();
                    if g.is_none() {
                        *g = Some(e);
                    }
                    return;
                }
            });
        }
    });
    match error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Document-queue cap: at most this many built articles' documents wait for
/// the indexer thread, bounding the queue's RAM (backpressure) while the
/// walk stays ahead of the index adds.
const DOC_QUEUE_CAP: usize = 2000;

/// State under [`DocQueue`]'s mutex.
struct QueueState {
    /// Built (fulltext, title) document pairs with the article path (for
    /// error messages), in worker-completion order.
    docs: VecDeque<(Document, Document, String)>,
    /// No more documents will arrive (the walk joined, or the indexer
    /// stopped on its first error).
    closed: bool,
    /// The indexer's first add/commit error (the queue closes with it).
    error: Option<String>,
}

/// The bounded channel between the walk workers and the dedicated indexer
/// thread (same Mutex+Condvar shape as zimwrite's compression pool): workers
/// block while full (backpressure), the indexer blocks while empty, `close`
/// releases a drained queue, and the indexer's first error closes it so
/// every waiting worker aborts instead of hanging.
struct DocQueue {
    state: Mutex<QueueState>,
    wake: Condvar,
}

impl DocQueue {
    fn new() -> Self {
        DocQueue {
            state: Mutex::new(QueueState {
                docs: VecDeque::new(),
                closed: false,
                error: None,
            }),
            wake: Condvar::new(),
        }
    }

    /// Lock, recovering from a poisoned mutex: a panic must not turn the
    /// error path into a hang.
    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Queue one article's documents. Blocks while the queue is full; fails
    /// fast with the indexer's error once it has stopped.
    fn push(&self, ft_doc: Document, ti_doc: Document, path: String) -> Result<(), String> {
        let mut st = self.lock();
        loop {
            if let Some(e) = &st.error {
                return Err(e.clone());
            }
            if st.closed {
                return Err("indexer stopped unexpectedly".to_string());
            }
            if st.docs.len() < DOC_QUEUE_CAP {
                break;
            }
            st = self.wake.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.docs.push_back((ft_doc, ti_doc, path));
        drop(st);
        self.wake.notify_all();
        Ok(())
    }

    /// Take the next document pair; `None` once the queue is closed and
    /// drained (the indexer's exit condition).
    fn pop(&self) -> Option<(Document, Document, String)> {
        let mut st = self.lock();
        loop {
            if let Some(doc) = st.docs.pop_front() {
                drop(st);
                self.wake.notify_all(); // a queue slot freed (backpressure waiters)
                return Some(doc);
            }
            if st.closed {
                return None;
            }
            st = self.wake.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// No more documents will arrive (the walk's join is done). Idempotent.
    fn close(&self) {
        self.lock().closed = true;
        self.wake.notify_all();
    }

    /// Record the indexer's first error and stop the queue: every waiting
    /// worker wakes and aborts with it.
    fn fail(&self, error: String) {
        let mut st = self.lock();
        if st.error.is_none() {
            st.error = Some(error);
        }
        st.closed = true;
        drop(st);
        self.wake.notify_all();
    }
}

/// The dedicated indexer thread's body: owns both WritableDatabases for the
/// whole walk, drains the document queue (FIFO of worker completion),
/// increments the document counters and paces commits every
/// [`COMMIT_EVERY`] documents exactly like the old per-worker lock scopes,
/// then performs the final commits of both databases and hands them back to
/// the caller. The first add/commit error stops it (the queue closes with
/// the error, so workers abort); on error the final commits are skipped.
fn run_indexer(
    mut ft: WritableDatabase,
    mut ti: WritableDatabase,
    queue: &DocQueue,
    ft_docs: &AtomicU64,
    ti_docs: &AtomicU64,
) -> (WritableDatabase, WritableDatabase, Result<(), String>) {
    let mut error: Option<String> = None;
    while let Some((ft_doc, ti_doc, path)) = queue.pop() {
        if let Err(e) = ft.add_document(&ft_doc) {
            error = Some(format!("indexing {path:?}: {e}"));
            break;
        }
        let d = ft_docs.fetch_add(1, Ordering::Relaxed) + 1;
        if d % COMMIT_EVERY == 0 {
            if let Err(e) = ft.commit() {
                error = Some(format!("fulltext index commit: {e}"));
                break;
            }
        }
        if let Err(e) = ti.add_document(&ti_doc) {
            error = Some(format!("title indexing {path:?}: {e}"));
            break;
        }
        let d = ti_docs.fetch_add(1, Ordering::Relaxed) + 1;
        if d % COMMIT_EVERY == 0 {
            if let Err(e) = ti.commit() {
                error = Some(format!("title index commit: {e}"));
                break;
            }
        }
    }
    if let Some(e) = error {
        queue.fail(e.clone());
        return (ft, ti, Err(e));
    }
    if let Err(e) = ft.commit() {
        return (ft, ti, Err(format!("fulltext index commit: {e}")));
    }
    if let Err(e) = ti.commit() {
        return (ft, ti, Err(format!("title index commit: {e}")));
    }
    (ft, ti, Ok(()))
}

/// Convert one article's resolved HTML bytes to markdown. `Ok(None)` is a
/// failed conversion - counted as a conversion failure with the reference's
/// warning, never fatal.
fn convert_article_html(
    html: &[u8],
    path: &str,
    title: &str,
    lang: &str,
) -> Result<Option<String>, String> {
    match std::str::from_utf8(html) {
        Ok(html) => Ok(Some(crate::html2md::html_to_md(html, Some(title), Some(lang)))),
        Err(_) => {
            eprintln!("warning: failed to convert {path:?}: content is not valid UTF-8");
            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// The conversion pass (zim2zim.convert)
// ---------------------------------------------------------------------------

/// Convert the HTML ZIM at `zimfile` into a Markdown ZIM at `outfile` with
/// freshly built fulltext + title Xapian indexes, mirroring
/// `zim2zim.py convert()`. `limit` < 0 converts everything; otherwise only
/// the first `limit` entries are processed (every entry counts one), which
/// can leave the output with dangling redirects (dropped at `finish`).
///
/// Accent folding is always applied to the indexed text (the search side
/// folds queries the same way); `index_intro_only` fulltext-indexes each
/// article's intro instead of the whole markdown; `index_redirect_titles`
/// gives recreated redirects and no-content pages (title-only or title plus
/// one bare wikilink — never fulltext-indexed) their title-index documents
/// (the default excludes them, zim2zim's --no-redirect-titles behavior).
pub fn convert(
    zimfile: &Path,
    outfile: &Path,
    limit: i64,
    index_intro_only: bool,
    index_redirect_titles: bool,
) -> Result<(), String> {
    let t0 = Instant::now();
    eprintln!("Opening source ZIM: {}", zimfile.display());
    let z = Zim::open(zimfile)
        .map_err(|e| format!("failed to open {}: {e}", zimfile.display()))?;
    let entry_count = z.entry_count();
    // The MIME list runs from id 0 up to the first unknown id.
    let mut mime_count = 0u16;
    while z.mime_type(mime_count).is_some() {
        mime_count += 1;
    }
    eprintln!("Enumerating source ZIM...");
    eprintln!("  {entry_count} entries ({mime_count} MIME types)");

    let main_entry_path = main_entry_path(&z);

    let src_lang_raw = raw_language_metadata(&z);
    let conv_lang = indexing_language(src_lang_raw.as_deref());
    eprintln!(
        "  language: {conv_lang} (source metadata: {})",
        src_lang_raw.as_deref().unwrap_or("none")
    );
    let lang = IndexLang::new(conv_lang.clone());

    let limit_entries: u32 = if limit < 0 {
        entry_count
    } else {
        (limit as u64).min(entry_count as u64) as u32
    };

    // The two Xapian databases are built under a unique temp dir NEXT TO THE
    // OUTPUT (huge archives need the output filesystem's space for the
    // throwaway databases), compacted to single files and streamed into the
    // archive. The counter keeps concurrent conversions (test threads)
    // apart.
    static TMP_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = outfile
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            ".szmcp-convert-{}-{n}-{}",
            std::process::id(),
            outfile.file_name().and_then(|n| n.to_str()).unwrap_or("zim")
        ));
    if tmp.exists() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    std::fs::create_dir_all(&tmp)
        .map_err(|e| format!("cannot create temp dir {}: {e}", tmp.display()))?;

    // The inner closure exists only so the temp dir is removed on every
    // error path.
    let outcome = (|| -> Result<(), String> {
        // The walk range: the LIMIT (not the full entry count) — --limit
        // runs never allocate proportional to the archive.
        let end = limit_entries;
        let mut creator = ZimCreator::new(outfile)
            .map_err(|e| format!("cannot create {}: {e}", outfile.display()))?;

        // Creator preamble (zim2zim's `with creator:` block): metadata and
        // the illustration are copied before the walk; the article MIME is
        // registered once here instead of per article (the writer sorts the
        // MIME list at begin_write, so registration order is irrelevant).
        let metadata_language = copy_metadata(&z, &mut creator, &conv_lang)?;
        copy_illustration(&z, &mut creator)?;
        creator.register_mime("text/markdown").map_err(|e| format!("mime: {e}"))?;
        // The preamble is done; from here the workers share the creator
        // behind its mutex (short blob adds only).
        let creator = Mutex::new(creator);

        // Both databases are compacted to the single files libzim embeds
        // (xapianIndexer.cpp indexingPrelude).
        let ft_path = tmp.join("fulltext.idx");
        let ti_path = tmp.join("title.idx");
        let ft_wdb = create_index_wdb(
            &tmp.join("fulltext.idx.tmp"),
            "fulltext",
            "title:0;wordcount:1;geo.position:2",
            &lang.raw,
        )?;
        let ti_wdb = create_index_wdb(
            &tmp.join("title.idx.tmp"),
            "title",
            "title:0;targetPath:1",
            &lang.raw,
        )?;

        // Thread pool: N workers pulling work from a shared atomic counter
        // - fixed 512-entry chunks in the classification pre-pass, one
        // cluster task per fetch-add in the conversion walk - dynamic, so
        // heterogeneous cores stay busy. Blob adds still go through short
        // mutex-locked adds on the shared creator; index documents cross to
        // the indexer thread through a bounded queue (see the walk below).
        let threads = std::env::var("SZMCP_CONVERT_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        eprintln!("  {threads} conversion threads");

        let mut shared = Shared::default();
        // One skip-tally slot per id in the archive's MIME list: skips
        // stay a single relaxed fetch-add.
        shared.skipped = (0..mime_count as usize).map(|_| AtomicU64::new(0)).collect();
        // Per-entry records, preallocated exactly once for the LIMIT (not
        // the full entry count): 12 bytes per entry is the dominant
        // constant-RAM structure (600 MB at 50M entries).
        let recs: Vec<RecSlot> = (0..end).map(|_| RecSlot::default()).collect();

        // ---- pass 1, phase A: a lean parallel classification walk in path
        // order — dirents only, NO blob IO. Every entry stores its 12-byte
        // record (redirects resolve their terminal now, non-HTML mimes are
        // tallied); every HTML article also has its path and raw title
        // copied into the collector's string arena below, so the conversion
        // walk performs zero dirent reads.
        let collector = Mutex::new(PrePass {
            arena: Vec::new(),
            coords: Vec::new(),
        });
        let t_classify = Instant::now();
        eprintln!("Classifying {end} entries (dirent headers only)...");
        run_chunk_workers(threads, &shared.next, end, |start, stop| {
            // Chunk-local arena and coordinates: the offsets in `found` are
            // relative to this buffer, rebased onto the shared arena under
            // the collector's lock below.
            let mut arena: Vec<u8> = Vec::new();
            let mut found: Vec<ArticleCoord> = Vec::new();
            for idx in start..stop {
                shared.processed.fetch_add(1, Ordering::Relaxed);
                let (mime, target) = z
                    .entry_head(idx)
                    .map_err(|e| format!("reading entry {idx}: {e}"))?;
                if mime == MIME_REDIRECT {
                    shared.redirects.fetch_add(1, Ordering::Relaxed);
                    // A redirect: resolve its terminal and the terminal's mime
                    // NOW (dirent headers only, no blobs) and store both in the
                    // record; liveness is decided after the join, when the
                    // terminal's own conversion outcome is final.
                    let slot = &recs[idx as usize];
                    match z.redirect_terminal(idx) {
                        Some(t) => {
                            let (tmime, _) = z
                                .entry_head(t)
                                .map_err(|e| format!("reading redirect target of entry {idx}: {e}"))?;
                            slot.a.store(t, Ordering::Relaxed);
                            slot.flags.store(
                                REC_REDIRECT
                                    | if z
                                        .mime_type(tmime)
                                        .is_some_and(|m| m.starts_with("text/html"))
                                    {
                                        REC_TARGET_HTML
                                    } else {
                                        0
                                    },
                                Ordering::Relaxed,
                            );
                        }
                        None => slot.flags.store(REC_REDIRECT, Ordering::Relaxed),
                    }
                    continue;
                }
                if !z.mime_type(mime).is_some_and(|m| m.starts_with("text/html")) {
                    // Media/metadata/whatever: skipped, tallied by MIME.
                    if let Some(c) = shared.skipped.get(mime as usize) {
                        c.fetch_add(1, Ordering::Relaxed);
                    } else {
                        shared.skipped_unknown.fetch_add(1, Ordering::Relaxed);
                    }
                    continue;
                }
                // An HTML article: capture path and title NOW (the pre-pass
                // still reads dirents sequentially) plus the blob
                // coordinates for the cluster-order walk. Raw bytes only,
                // no fallback: the walk applies entry_title().
                let Target::Cluster(cluster, blob) = target else {
                    continue;
                };
                let entry = z.get_entry(idx).map_err(|e| format!("reading entry {idx}: {e}"))?;
                let path_off = arena.len() as u32;
                arena.extend_from_slice(entry.url.as_bytes());
                let title_off = arena.len() as u32;
                arena.extend_from_slice(entry.title.as_bytes());
                found.push(ArticleCoord {
                    cluster,
                    blob,
                    idx,
                    path: (path_off, entry.url.len() as u32),
                    title: (title_off, entry.title.len() as u32),
                });
            }
            // One lock acquisition per 512-entry chunk, not per entry:
            // rebase the chunk-local offsets onto the shared arena, then
            // extend both under the same lock so coordinates and strings
            // stay consistent.
            let mut pre = collector.lock().unwrap();
            let base = pre.arena.len() as u32;
            for c in found.iter_mut() {
                c.path.0 += base;
                c.title.0 += base;
            }
            pre.arena.extend_from_slice(&arena);
            pre.coords.append(&mut found);
            drop(pre);
            Ok(())
        })?;

        let dt = t_classify.elapsed().as_secs_f64();
        let articles = collector.lock().unwrap().coords.len();
        let redirects = shared.redirects.load(Ordering::Relaxed);
        // The skip tally as (label, count): one slot per archive MIME id,
        // plus one combined line for ids past the list.
        let mut skipped: Vec<(String, u64)> = shared
            .skipped
            .iter()
            .enumerate()
            .map(|(id, n)| (skip_mime_name(&z, id as u16), n.load(Ordering::Relaxed)))
            .filter(|(_, n)| *n > 0)
            .collect();
        let unknown = shared.skipped_unknown.load(Ordering::Relaxed);
        if unknown > 0 {
            skipped.push(("unknown".to_string(), unknown));
        }
        skipped.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        let skipped_total: u64 = skipped.iter().map(|&(_, n)| n).sum();
        eprintln!(
            "  classified {end} entries in {dt:.1}s: {articles} articles, {redirects} redirects, {skipped_total} skipped"
        );

        // Between the phases: the collected coordinates sort by (cluster,
        // blob, idx) — ascending cluster order is what makes the source
        // reads sequential — and group into one task per cluster (the
        // consecutive items sharing one). For a 6.7M-article ZIM the
        // coordinates are ~130 MB of RAM on top of the collector's string
        // arena (see PrePass).
        let PrePass { arena, mut coords } = collector.into_inner().unwrap();
        coords.sort_unstable();
        shared.articles_total = coords.len() as u64;
        let mut tasks: Vec<(u32, Vec<ArticleCoord>)> = Vec::new();
        for c in coords {
            match tasks.last_mut() {
                Some((cl, items)) if *cl == c.cluster => items.push(c),
                _ => tasks.push((c.cluster, vec![c])),
            }
        }
        // The shared counter hands out 512-entry chunks in the pre-pass and
        // cluster tasks in the walk: reset it between the two phases.
        shared.next.store(0, Ordering::Relaxed);

        // ---- pass 1, phase B: the conversion walk, one task per cluster in
        // ascending cluster order — that order reads and decodes each
        // compressed cluster exactly once and keeps the source reads
        // sequential. ZERO dirent reads: every path and title rides in the
        // pre-pass's string arena (borrowed below).
        //
        // Indexing runs on ONE dedicated thread that owns both databases for
        // the whole walk: workers build each article's documents outside any
        // lock and ship the pair over a bounded queue — blocking while full,
        // so neither add_document nor the every-[`COMMIT_EVERY`] commits
        // ever stall the workers.
        let queue = Arc::new(DocQueue::new());
        let indexer = {
            let queue = Arc::clone(&queue);
            let ft_docs = Arc::clone(&shared.ft_docs);
            let ti_docs = Arc::clone(&shared.ti_docs);
            std::thread::spawn(move || run_indexer(ft_wdb, ti_wdb, &queue, &ft_docs, &ti_docs))
        };
        let walked = if !tasks.is_empty() {
            run_task_workers(threads, &shared.next, &tasks, |(cluster, items)| {
                let compressed = z
                    .cluster_is_compressed(*cluster)
                    .map_err(|e| format!("reading cluster {cluster}: {e}"))?;
                // A compressed cluster (the overwhelmingly common case for
                // text/html) decodes exactly once per task; an uncompressed
                // cluster is NOT copied into RAM (it can hold media) — each
                // wanted blob is read per item below.
                let data = if compressed {
                    Some(
                        z.decompress_cluster_with_spans(*cluster)
                            .map_err(|e| format!("decoding cluster {cluster}: {e}"))?,
                    )
                } else {
                    None
                };
                for c in items {
                    let n = shared.articles_done.fetch_add(1, Ordering::Relaxed);
                    if n % STATUS_EVERY == STATUS_EVERY - 1 {
                        // \r, not \n: the next status overwrites this one; the
                        // epilogue after the join submits the final newline.
                        eprint!(
                            "[{}/{}] articles: {} converted ({:.1} MB written)\r",
                            n + 1,
                            shared.articles_total,
                            shared.ft_docs.load(Ordering::Relaxed),
                            shared.md_bytes.load(Ordering::Relaxed) as f64 / 1e6
                        );
                    }
                    // Path and title from the pre-pass arena (offset, len);
                    // the bytes came from get_entry's parsed Strings, so a
                    // UTF-8 failure cannot happen in practice — map it to a
                    // clear fatal error rather than panic.
                    let item_path = std::str::from_utf8(
                        &arena[c.path.0 as usize..c.path.0 as usize + c.path.1 as usize],
                    )
                    .map_err(|e| format!("path of entry {}: {e}", c.idx))?;
                    let title = entry_title(
                        std::str::from_utf8(
                            &arena[c.title.0 as usize..c.title.0 as usize + c.title.1 as usize],
                        )
                        .map_err(|e| format!("title of entry {}: {e}", c.idx))?,
                        item_path,
                    );
                    let html = match &data {
                        Some(d) => d.span(c.blob).map(|(s, e)| {
                            std::borrow::Cow::Borrowed(&d.data[s as usize..e as usize])
                        }),
                        None => z.read_blob(*cluster, c.blob).map(std::borrow::Cow::Owned),
                    };
                    let html = match html {
                        Ok(html) => html,
                        Err(e) => {
                            eprintln!("warning: failed to convert {item_path:?}: {e}");
                            shared.failed.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };
                    let Some(md) = convert_article_html(&html, &item_path, &title, &conv_lang)?
                    else {
                        shared.failed.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    // No-content page (title-only, or title plus exactly one
                    // bare wikilink line — mwoffliner's meta-refresh redirect
                    // stubs): indexed like a redirect — no fulltext doc, and
                    // a title doc only via the redirect-titles post-pass
                    // below. The blob is still written.
                    let body = md.split_once('\n').map(|x| x.1.trim()).unwrap_or("");
                    let no_content = body.is_empty()
                        || (body.starts_with("[[") && body.ends_with("]]") && !body.contains('\n'));
                    if !no_content {
                        // Both documents build OUTSIDE any lock — the TermGenerator/Document
                        // FFI never touches the WDB pointer (exactly libzim's own
                        // worker design, which indexes into Document objects on
                        // worker threads) — then cross to the indexer thread
                        // through the bounded queue, which blocks while full (its
                        // backpressure) and aborts with the indexer's error once
                        // indexing has stopped.
                        let folded_title = crate::search::fold_accents(&title);
                        let folded_content = if index_intro_only {
                            crate::search::fold_accents(&intro_for_index(&md))
                        } else {
                            crate::search::fold_accents(&md)
                        };
                        let ft_doc = build_fulltext_document(
                            &lang,
                            &item_path,
                            &folded_title,
                            &folded_content,
                        )
                        .map_err(|e| format!("indexing {item_path:?}: {e}"))?;
                        let ti_doc = build_title_document(&lang, &item_path, &title, None)
                            .map_err(|e| format!("title indexing {item_path:?}: {e}"))?;
                        queue.push(ft_doc, ti_doc, item_path.to_string())?;
                    }
                    // Writer add in its own lock scope (no nesting): the
                    // record carries the blob's ref.
                    {
                        let mut zc = creator.lock().unwrap();
                        let b = zc
                            .add_blob(true, md.as_bytes())
                            .map_err(|e| format!("item {item_path:?}: {e}"))?;
                        let slot = &recs[c.idx as usize];
                        slot.a.store(b.generation, Ordering::Relaxed);
                        slot.b.store(b.blob, Ordering::Relaxed);
                        slot.flags.store(
                            REC_ARTICLE
                                | if no_content { REC_NOCONTENT } else { 0 }
                                | if b.compress { REC_COMPRESS } else { 0 },
                            Ordering::Relaxed,
                        );
                    }
                    shared.md_bytes.fetch_add(md.len() as u64, Ordering::Relaxed);
                }
                Ok(())
            })
        } else {
            Ok(())
        };
        // The walk's join is done: close the queue and wait for the indexer
        // to drain the remaining documents, do the final commits of both
        // databases and hand them back. ft_docs keeps rising until then, so
        // the final status print below must follow this join.
        queue.close();
        let (ft, mut ti, index_result) =
            indexer.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
        walked?;
        index_result?;
        if !tasks.is_empty() {
            // Final status: workers and indexer are done, so the counters
            // are final — the last periodic line can be up to STATUS_EVERY-1
            // items stale. Always emit the completed state. The cursor is at
            // column 0 here (every earlier status ends in \r or \n), so a
            // plain eprintln overwrites it.
            eprintln!(
                "[{}/{}] articles: {} converted ({:.1} MB written)",
                shared.articles_done.load(Ordering::Relaxed),
                shared.articles_total,
                shared.ft_docs.load(Ordering::Relaxed),
                shared.md_bytes.load(Ordering::Relaxed) as f64 / 1e6
            );
        }

        // ---- post-join: the indexer handed both databases back with their
        // final commits done; the serial rest owns them outright.

        // Membership is pure arithmetic over the record flags, no IO: an
        // article is a member iff it was written; a redirect iff its chain
        // resolved to an HTML terminal that was written. The --limit cutoff is
        // encoded in the ARTICLE flag itself (only entries below the limit ever
        // got it) and the check runs AFTER the join, so a redirect whose target
        // converts later in path order is still live.
        let mut member = vec![0u64; end.div_ceil(64) as usize];
        for idx in 0..end {
            let s = &recs[idx as usize];
            let f = s.flags.load(Ordering::Relaxed);
            let live = if f & REC_ARTICLE != 0 {
                true
            } else if f & (REC_REDIRECT | REC_TARGET_HTML) == REC_REDIRECT | REC_TARGET_HTML {
                let t = s.a.load(Ordering::Relaxed);
                t != u32::MAX
                    && t < end
                    && recs[t as usize].flags.load(Ordering::Relaxed) & REC_ARTICLE != 0
            } else {
                false
            };
            if live {
                member[idx as usize / 64] |= 1 << (idx % 64);
            }
        }
        let rank = Rank::new(&member);

        // Opt-in redirect title documents (the default build has none): value 1
        // = the terminal article's path, like the walk's article adds —
        // except no-content articles, which get no target: their `a` holds
        // a blob generation, not an index, so the doc targets their own
        // path like a regular article add. Serial; commits paced every
        // [`COMMIT_EVERY`] documents, before the title database's compaction.
        if index_redirect_titles {
            let mut added = 0u64;
            for idx in 0..end {
                let flags = recs[idx as usize].flags.load(Ordering::Relaxed);
                if member[idx as usize / 64] & (1 << (idx % 64)) == 0
                    || flags & (REC_REDIRECT | REC_NOCONTENT) == 0
                {
                    continue;
                }
                let entry = z.get_entry(idx)
                    .map_err(|e| format!("reading entry {idx}: {e}"))?;
                let title = entry_title(&entry.title, &entry.url);
                if title.is_empty() {
                    continue;
                }
                let doc = if flags & REC_NOCONTENT != 0 {
                    build_title_document(&lang, &entry.url, &title, None)
                } else {
                    let t = recs[idx as usize].a.load(Ordering::Relaxed);
                    let t_entry = z.get_entry(t)
                        .map_err(|e| format!("reading entry {t}: {e}"))?;
                    build_title_document(&lang, &entry.url, &title, Some(&t_entry.url))
                }
                .map_err(|e| format!("title indexing {:?}: {e}", entry.url))?;
                ti.add_document(&doc)
                    .map_err(|e| format!("title indexing {:?}: {e}", entry.url))?;
                added += 1;
                if added % COMMIT_EVERY == 0 {
                    ti.commit().map_err(|e| format!("title index commit: {e}"))?;
                }
            }
            shared.ti_docs.fetch_add(added, Ordering::Relaxed);
            ti.commit().map_err(|e| format!("title index commit: {e}"))?;
        }

        // Compaction postlude: single-file databases (DBCOMPACT_SINGLE_FILE |
        // FULL); an index with no documents is not embedded at all.
        let converted = shared.ft_docs.load(Ordering::Relaxed);
        let ti_docs = shared.ti_docs.load(Ordering::Relaxed);
        let mut ft_file = None;
        let mut ti_file = None;
        if converted > 0 {
            ft.compact_to_path(&ft_path)
                .map_err(|e| format!("fulltext index compact: {e}"))?;
            ft_file = Some(&ft_path);
        }
        if ti_docs > 0 {
            ti.compact_to_path(&ti_path)
                .map_err(|e| format!("title index compact: {e}"))?;
            ti_file = Some(&ti_path);
        }

        // ---- finalize (single-threaded): ONE dirent-header walk over the
        // members — emit the dirents in SOURCE order and build the title-
        // ordered listing rows. Listing sort key: the derived title in a
        // bump arena, tie-broken by the member's OUTPUT index, which IS
        // (title, path) order — the source dirents are path-sorted and all
        // members are C-namespace.
        eprintln!("Finalizing (this may take a while)...");
        // The source index of the main-page PATH (its terminal article), for the
        // resolved W/mainPage target below.
        let main_src_idx = main_entry_path
            .as_deref()
            .and_then(|mp| z.find_entry(b'C', mp).ok().flatten());

        // Embedded index blobs first (the content order libzim's finish
        // produces: fulltext, title, then counter and listing at finish_write).
        if let Some(path) = ft_file {
            creator
                .lock()
                .unwrap()
                .add_xapian_index("fulltext/xapian", path)
                .map_err(|e| format!("fulltext index: {e}"))?;
        }
        if let Some(path) = ti_file {
            creator
                .lock()
                .unwrap()
                .add_xapian_index("title/xapian", path)
                .map_err(|e| format!("title index: {e}"))?;
        }
        creator.lock().unwrap().begin_write().map_err(|e| format!("finalizing: {e}"))?;

        // Member dirents in SOURCE order (the archive's entry order); the
        // terminal of a member redirect rides in its record — no re-resolution,
        // no error path. The listing row is built while the dirent is already
        // parsed; the title bytes go into a NUL-terminated bump arena (titles
        // derive from dirent titles and paths, which cannot contain NUL).
        let mut first_member_article = None;
        let mut rows: Vec<u64> = Vec::new();
        let mut arena: Vec<u8> = Vec::new();
        for idx in 0..end {
            if member[idx as usize / 64] & (1 << (idx % 64)) == 0 {
                continue;
            }
            let rec = &recs[idx as usize];
            let entry = z.get_entry(idx)
                .map_err(|e| format!("reading entry {idx}: {e}"))?;
            let title = entry_title(&entry.title, &entry.url);
            let out_idx = rank.rank(idx as usize);
            let article = rec.flags.load(Ordering::Relaxed) & REC_ARTICLE != 0;
            if article || index_redirect_titles {
                if arena.len() + title.len() + 1 > u32::MAX as usize {
                    return Err("listing title arena exceeds 4 GiB".to_string());
                }
                let off = arena.len() as u64;
                arena.extend_from_slice(title.as_bytes());
                arena.push(0);
                rows.push((off << 32) | out_idx as u64);
            }
            if article {
                creator.lock().unwrap().emit_dirent(DirentOut::Item {
                    ns: b'C',
                    path: entry.url.clone(),
                    title,
                    mime: "text/markdown".to_string(),
                    blob: BlobRef {
                        compress: rec.flags.load(Ordering::Relaxed) & REC_COMPRESS != 0,
                        generation: rec.a.load(Ordering::Relaxed),
                        blob: rec.b.load(Ordering::Relaxed),
                    },
                })
                .map_err(|e| format!("writing {:?}: {e}", entry.url))?;
                if first_member_article.is_none() {
                    first_member_article = Some(out_idx);
                }
            } else {
                creator.lock().unwrap().emit_dirent(DirentOut::Redirect {
                    ns: b'C',
                    path: entry.url.clone(),
                    title,
                    target_idx: rank.rank(rec.a.load(Ordering::Relaxed) as usize),
                })
                .map_err(|e| format!("writing {:?}: {e}", entry.url))?;
            }
        }

        // Sort the rows by (title, output index) and build the u32-LE listing
        // blob of member output indexes; the arena slices are NUL-terminated, so
        // a slice comparison compares exactly the title bytes.
        rows.sort_unstable_by(|&a, &b| {
            let ta = arena_title(&arena, (a >> 32) as usize);
            let tb = arena_title(&arena, (b >> 32) as usize);
            ta.cmp(tb).then_with(|| (a & 0xFFFF_FFFF).cmp(&(b & 0xFFFF_FFFF)))
        });
        let mut listing_blob = Vec::with_capacity(rows.len() * 4);
        for &r in &rows {
            listing_blob.extend_from_slice(&((r & 0xFFFF_FFFF) as u32).to_le_bytes());
        }
        drop(rows);
        drop(arena);
        creator.lock().unwrap().set_listing_bytes(listing_blob);

        // Main path: the source main page if its article was written, else the
        // first written article - as a RESOLVED target entry index.
        let main_target = main_src_idx
            .filter(|i| {
                *i < end && recs[*i as usize].flags.load(Ordering::Relaxed) & REC_ARTICLE != 0
            })
            .map(|i| rank.rank(i as usize))
            .or(first_member_article);
        creator
            .lock()
            .unwrap()
            .set_main_page_target(main_target);
        creator
            .lock()
            .unwrap()
            .finish_write()
            .map_err(|e| format!("finalizing {}: {e}", outfile.display()))?;

        // Summary (stderr, like all human output of this subcommand).
        let out_size = std::fs::metadata(outfile).map(|m| m.len()).unwrap_or(0);
        let input_size = std::fs::metadata(zimfile).map(|m| m.len()).unwrap_or(0);
        let elapsed = t0.elapsed().as_secs_f64();
        let processed = shared.processed.load(Ordering::Relaxed);
        let failed = shared.failed.load(Ordering::Relaxed);
        eprintln!();
        eprintln!("{}", "=".repeat(60));
        eprintln!("SUMMARY");
        eprintln!("  entries processed  : {processed} of {entry_count}");
        eprintln!("  articles converted : {converted}");
        eprintln!("  language           : {} (metadata: {})", conv_lang, metadata_language);
        eprintln!("  conversion failures: {failed}");
        eprintln!("  skipped entries by MIME:");
        for (mime, count) in skipped {
            eprintln!("    {mime:<30} {count}");
        }
        eprintln!("  input size : {:.1} MB", input_size as f64 / 1e6);
        eprintln!("  output size: {:.1} MB", out_size as f64 / 1e6);
        eprintln!("  elapsed    : {elapsed:.1} s");
        eprintln!("{}", "=".repeat(60));
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    outcome
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_title_cuts_to_238_utf8_bytes() {
        assert_eq!(trim_title("short"), "short");
        // 238 ASCII bytes pass through unchanged.
        let ok = "x".repeat(238);
        assert_eq!(trim_title(&ok), ok);
        // 239 bytes get cut to 238.
        let long = format!("{}x", "x".repeat(238));
        assert_eq!(trim_title(&long), ok);
        // A multibyte character straddling the cut is dropped whole.
        let straddled = format!("{}é{}", "x".repeat(237), "y".repeat(5));
        assert_eq!(trim_title(&straddled), "x".repeat(237));
    }

    #[test]
    fn entry_title_falls_back_to_the_url() {
        assert_eq!(entry_title("Dirent Title", "Some_Path"), "Dirent Title");
        assert_eq!(entry_title("", "Some_Path"), "Some Path");
    }

    #[test]
    fn hatnote_paragraphs() {
        assert!(is_hatnote_para("*allons enfants*"));
        assert!(is_hatnote_para("  *Not to be confused with X*  "));
        // Bullets, bold, spaced asterisks, and inner asterisks don't match
        // (python: t[0]=='*' and t[-1]=='*' and t[1] not in " *" and no '*'
        // inside).
        assert!(!is_hatnote_para("* item"));
        assert!(!is_hatnote_para("**bold**"));
        assert!(!is_hatnote_para("* allons enfants *"));
        assert!(!is_hatnote_para("* spaced *"));
        assert!(!is_hatnote_para("* inner *asterisk* *"));
        assert!(!is_hatnote_para("**"));
    }

    #[test]
    fn intro_selection() {
        let md = "# Titre\n\n*hatnote note*\n\nPara one.\n\nPara two.\n\n## History\n\nBody.\n";
        assert_eq!(intro_for_index(md), "# Titre\n\nPara one.\n\nPara two.");
        // No heading: the whole body after the title line is the intro.
        assert_eq!(intro_for_index("# T\n\nOnly.\n"), "# T\n\nOnly.");
        // A leading hatnote run is dropped, the next paragraph kept.
        let two_hats = "# T\n\n*one*\n\n*two*\n\nReal.\n\n## H\n\nB.\n";
        assert_eq!(intro_for_index(two_hats), "# T\n\nReal.");
        // Empty content degrades to an empty intro.
        assert_eq!(intro_for_index(""), "");
    }

    #[test]
    fn indexing_language_normalization() {
        // First token wins; BCP-47 subtags are stripped; case is lowered.
        assert_eq!(indexing_language(Some("fra")), "fra");
        assert_eq!(indexing_language(Some("fr-FR, en")), "fr");
        assert_eq!(indexing_language(Some("  deu;eng  ")), "deu");
        assert_eq!(indexing_language(Some("zh-Hant-TW")), "zh");
        // Empty and absent values fall back to English.
        assert_eq!(indexing_language(Some("")), "eng");
        assert_eq!(indexing_language(None), "eng");
        assert_eq!(indexing_language(Some("   ")), "eng");
    }
}

/// A second test module: end-to-end conversions over a synthetic source ZIM
/// (the zim.rs testutil builds the archive; see search.rs's tests for the
/// pattern).
#[cfg(test)]
mod e2e {
    use super::*;
    use crate::zim::testutil::{build_archive, TestEntry, TestRedirect};

    const APPLE_HTML: &str = "<html><head><title>Apple</title></head><body><h1>Apple</h1>\
<p>An <b>apple</b> is the fruit of &lt;rosaceae&gt; trees.</p>\
<h2 id=\"History\">History</h2>\
<p>Apples have been cultivated for 10,000 years.</p>\
<h3>Domestication</h3><p>Wild apples grew in Kazakhstan.</p></body></html>";

    const BANANA_HTML: &str = "<html><body><h1>Banana</h1><p>A banana is an elongated fruit.</p></body></html>";

    const REV_HTML: &str = "<html><body><h1>Révolution</h1>\
<p>Une révolution est un changement politique majeur.</p></body></html>";

    const CSS: &[u8] = b"body{color:red}";
    const FAKE_PNG: &[u8] = b"\x89PNG\r\n\x1a\n-fake-48x48-illustration-";

    /// The synthetic source: three HTML articles (one accented), one CSS
    /// asset, two redirects (one to Apple, one to Banana), core metadata
    /// with Language=fra, and a 48x48 illustration. The main page is Apple.
    fn fixture() -> Vec<u8> {
        let content = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Révolution", title: "Révolution française", mime: 0, body: REV_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "style.css", title: "style.css", mime: 1, body: CSS },
            TestEntry { namespace: b'M', url: "Name", title: "", mime: 2, body: b"wikipedia_en_mini" },
            TestEntry { namespace: b'M', url: "Title", title: "", mime: 2, body: "Chemistry \u{2014} Mini".as_bytes() },
            TestEntry { namespace: b'M', url: "Language", title: "", mime: 2, body: b"fra" },
            TestEntry { namespace: b'M', url: "Description", title: "", mime: 2, body: b"A miniature encyclopedia" },
            TestEntry { namespace: b'M', url: "Illustration_48x48@1", title: "", mime: 3, body: FAKE_PNG },
        ];
        let redirects = [
            TestRedirect { namespace: b'C', url: "Apple_fruit", title: "", target_content: 0 },
            TestRedirect { namespace: b'C', url: "Alt_Banana", title: "Alternate Banana", target_content: 1 },
        ];
        build_archive(
            &["text/html", "text/css", "text/plain;charset=UTF-8", "image/png"],
            &content,
            &redirects,
            0, // main page: Apple
            None,
        )
    }

    struct Converted {
        _dir: tempfile::TempDir,
        zim: Zim,
    }

    fn convert_fixture(limit: i64, intro_only: bool, redirect_titles: bool) -> Converted {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.zim");
        std::fs::write(&src, fixture()).unwrap();
        let out = dir.path().join("out.zim");
        convert(&src, &out, limit, intro_only, redirect_titles).unwrap();
        let zim = Zim::open(&out).unwrap();
        Converted { _dir: dir, zim }
    }

    /// The blob of a directory entry, following its Target::Cluster.
    fn blob_of(z: &Zim, namespace: u8, url: &str) -> Option<Vec<u8>> {
        let idx = z.find_entry(namespace, url).ok()??;
        let entry = z.get_entry(idx).ok()?;
        match entry.target {
            Target::Cluster(c, b) => z.read_blob(c, b).ok(),
            _ => None,
        }
    }

    /// Resolve `path` to its terminal entry index (following redirects).
    fn terminal(z: &Zim, namespace: u8, url: &str) -> Option<(u32, crate::zim::Entry)> {
        let mut idx = z.find_entry(namespace, url).ok()??;
        for _ in 0..16 {
            let entry = z.get_entry(idx).ok()?;
            match entry.target {
                Target::Redirect(next) => idx = next,
                _ => return Some((idx, entry)),
            }
        }
        None
    }

    #[test]
    fn many_chunks_convert_deterministically_across_workers() {
        // A source bigger than one 512-entry chunk: the classification
        // pre-pass really distributes its 512-entry chunks across workers.
        // The fixtures store every entry UNCOMPRESSED in a single cluster
        // (see build_archive_indexes), so the conversion walk is one task
        // turned by one worker — the streaming emission must still
        // reproduce the source order deterministically.
        let n = 1200usize;
        let html = "<html><body><h1>T</h1><p>An article body.</p></body></html>";
        let mut content: Vec<TestEntry> = Vec::new();
        let mut redirects: Vec<TestRedirect> = Vec::new();
        for i in 0..n {
            content.push(TestEntry {
                namespace: b'C',
                url: Box::leak(format!("Page_{i:05}").into_boxed_str()),
                title: "",
                mime: 0,
                body: html.as_bytes(),
            });
            if i % 3 == 0 {
                // A media entry (skipped) and a redirect onto Page_i every
                // third article; content index of Page_i is i + 2*(i/3).
                content.push(TestEntry {
                    namespace: b'C',
                    url: Box::leak(format!("Pic_{i:05}").into_boxed_str()),
                    title: "",
                    mime: 1,
                    body: b"png",
                });
                redirects.push(TestRedirect {
                    namespace: b'C',
                    url: Box::leak(format!("Alias_{i:05}").into_boxed_str()),
                    title: "",
                    target_content: i + i / 3,
                });
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.zim");
        std::fs::write(
            &src,
            build_archive(&["text/html", "image/png"], &content, &redirects, 0, None),
        )
        .unwrap();
        let out = dir.path().join("out.zim");
        convert(&src, &out, -1, false, false).unwrap();
        let z = Zim::open(&out).unwrap();

        // Every article is present as text/markdown, in SOURCE member
        // order: the source holds all 1200 articles first (media skipped
        // entirely), then the 400 redirects, so the output C entries are
        // the pages at 0..n and the aliases at n..n+n/3.
        for e in &content {
            if e.url.starts_with("Page_") {
                let i: usize = e.url.strip_prefix("Page_").unwrap().parse().unwrap();
                // Sorted dirent order: the 400 "Alias_" entries sort
                // before every "Page_" entry (the archive is path-sorted;
                // this IS the source member order, the source being sorted).
                let idx = z.find_entry(b'C', &e.url).unwrap().unwrap();
                assert_eq!(idx as usize, redirects.len() + i, "order for {}", e.url);
                let entry = z.get_entry(idx).unwrap();
                assert_eq!(z.mime_type(entry.mime), Some("text/markdown"));
            } else {
                assert!(z.find_entry(b'C', &e.url).unwrap().is_none(), "{} skipped", e.url);
            }
        }
        for r in &redirects {
            let idx = z.find_entry(b'C', r.url).unwrap().unwrap();
            assert_eq!(
                idx as usize,
                redirects.iter().position(|x| std::ptr::eq(x, r)).unwrap(),
                "aliases sort among themselves in numeric order"
            );
            assert_eq!(z.get_entry(idx).unwrap().mime, crate::zimcommon::MIME_REDIRECT);
        }

        // The counter and both indexes know exactly the article count.
        let counter = blob_of(&z, b'M', "Counter").unwrap();
        assert_eq!(counter, format!("text/markdown={n}").into_bytes());
        let ft = z.open_fulltext_xapian().unwrap().unwrap();
        assert_eq!(ft.doc_count(), n as u32);
        let ti = z.open_title_xapian().unwrap().unwrap();
        assert_eq!(ti.doc_count(), n as u32);
        // The listing lists every article once (redirect titles excluded).
        let listing = blob_of(&z, b'X', "listing/titleOrdered/v1").unwrap();
        assert_eq!(listing.len() / 4, n);

        // The document data set resolves to the article paths (docid order
        // races with the worker count).
        let mut paths = std::collections::BTreeSet::new();
        for d in 1..=ft.doc_count() {
            paths.insert(ft.get_document(d).unwrap().data_str().unwrap());
        }
        for i in 0..n {
            assert!(paths.contains(&format!("C/Page_{i:05}")), "doc {i} missing");
        }
    }

    #[test]
    fn converts_articles_recreates_redirects_and_copies_assets() {
        let c = convert_fixture(-1, false, false);
        let z = &c.zim;

        // The article is markdown at the SAME path, with the dirent title.
        let idx = z.resolve_path("C/Apple").unwrap().unwrap();
        let entry = z.get_entry(idx).unwrap();
        assert_eq!(z.mime_type(entry.mime), Some("text/markdown"));
        let md = match entry.target {
            Target::Cluster(cluster, blob) => z.read_blob(cluster, blob).unwrap(),
            _ => panic!("article has no content"),
        };
        let expected = crate::html2md::html_to_md(APPLE_HTML, Some("Apple"), Some("fra"));
        assert_eq!(md, expected.into_bytes());
        assert!(String::from_utf8_lossy(&md).contains("rosaceae"));

        // The redirect was recreated against the new path space, its empty
        // dirent title replaced by the URL-derived fallback.
        let (target_idx, target) = terminal(z, b'C', "Apple_fruit").unwrap();
        assert_eq!(target.url, "Apple");
        let redirect_idx = z.find_entry(b'C', "Apple_fruit").unwrap().unwrap();
        let redirect = z.get_entry(redirect_idx).unwrap();
        assert_eq!(redirect.title, "Apple fruit");
        let _ = target_idx;

        // Non-HTML assets are skipped entirely.
        assert!(z.resolve_path("C/style.css").unwrap().is_none());

        // Core metadata is copied verbatim (Language stays "fra").
        assert_eq!(blob_of(z, b'M', "Title"), Some("Chemistry \u{2014} Mini".into()));
        assert_eq!(blob_of(z, b'M', "Language"), Some(b"fra".to_vec()));
        assert_eq!(blob_of(z, b'M', "Name"), Some(b"wikipedia_en_mini".to_vec()));
        assert_eq!(blob_of(z, b'M', "Description"), Some(b"A miniature encyclopedia".to_vec()));

        // The illustration, the counter and the listing exist.
        assert_eq!(blob_of(z, b'M', "Illustration_48x48@1"), Some(FAKE_PNG.to_vec()));
        let counter = blob_of(z, b'M', "Counter").unwrap();
        assert!(String::from_utf8_lossy(&counter).contains("text/markdown=3"), "{counter:?}");
        let listing = blob_of(z, b'X', "listing/titleOrdered/v1").unwrap();
        // Three front articles (redirect titles excluded by default).
        assert_eq!(listing.len() / 4, 3);

        // The main page header points at the main article (Apple).
        assert_eq!(main_entry_path(z), Some("Apple".to_string()));
    }

    #[test]
    fn limit_drops_dangling_redirects() {
        // Entries in source order: C/Alt_Banana (redirect), C/Apple,
        // C/Apple_fruit (redirect), C/Banana, ... A limit of 3 stops after
        // Apple_fruit: Alt_Banana's target (Banana) was never converted, so
        // that redirect dangles and is dropped by finish().
        let c = convert_fixture(3, false, false);
        let z = &c.zim;
        assert!(z.resolve_path("C/Apple").unwrap().is_some());
        assert!(z.resolve_path("C/Apple_fruit").unwrap().is_some());
        assert!(z.resolve_path("C/Banana").unwrap().is_none());
        assert!(z.resolve_path("C/Alt_Banana").unwrap().is_none());

        // Only the one converted article made it into both indexes.
        let ft = z.open_fulltext_xapian().unwrap().unwrap();
        assert_eq!(ft.doc_count(), 1);
        let title = z.open_title_xapian().unwrap().unwrap();
        assert_eq!(title.doc_count(), 1);
    }

    #[test]
    fn title_index_excludes_redirect_titles_by_default() {
        let c = convert_fixture(-1, false, false);
        let z = &c.zim;
        let title = z.open_title_xapian().unwrap().unwrap();
        // One doc per article, sorted by path: Apple, Banana, Révolution.
        assert_eq!(title.doc_count(), 3);
        assert_eq!(title.get_metadata("kind").unwrap(), "title");
        assert_eq!(title.get_metadata("data").unwrap(), "fullPath");
        // The RAW indexing language code is stored (not the ICU mapping).
        assert_eq!(title.get_metadata("language").unwrap(), "fra");
        assert_eq!(title.get_metadata("valuesmap").unwrap(), "title:0;targetPath:1");
        assert_eq!(title.get_metadata("stopwords").unwrap(), "");

        let mut doc = title.get_document(1).unwrap();
        assert_eq!(doc.data_str().unwrap(), "C/Apple");
        assert_eq!(doc.value(0).unwrap(), b"Apple");
        assert_eq!(doc.value(1).unwrap(), b"Apple");
        let mut doc = title.get_document(3).unwrap();
        assert_eq!(doc.data_str().unwrap(), "C/Révolution");
        // Value 0 is the RAW title (NOT folded); value 1 the article's own
        // path (no targetPath for non-redirects).
        assert_eq!(doc.value(0).unwrap(), "Révolution française".as_bytes());
        assert_eq!(doc.value(1).unwrap(), "Révolution".as_bytes().to_vec());

        // Surface forms are stored (STEM_SOME over the folded title), the
        // Z-prefixed stem alongside (computed with the archive's stemmer).
        assert!(title.termfreq("revolution") > 0);
        assert!(title.termfreq("apple") > 0);
        let mut fr = xapian2::Stem::new("fr").unwrap();
        let stem = fr.apply("revolution").unwrap();
        assert!(title.termfreq(&format!("Z{stem}")) > 0, "Z{stem}");
    }

    #[test]
    fn title_index_includes_redirect_titles_opt_in() {
        let c = convert_fixture(-1, false, true);
        let z = &c.zim;
        let title = z.open_title_xapian().unwrap().unwrap();
        // Articles + both recreated redirects (both targets were converted),
        // as DATA: Alt_Banana, Apple, Apple_fruit, Banana, Révolution. The
        // document ids follow worker completion order (articles during the
        // walk, redirect titles after the join), so the value checks key by
        // data, like every other racing docid comparison.
        assert_eq!(title.doc_count(), 5);
        let doc_by_data = |data: &str| -> (Vec<u8>, Vec<u8>) {
            for d in 1..=title.doc_count() {
                let mut doc = title.get_document(d).unwrap();
                if doc.data_str().unwrap() == data {
                    return (doc.value(0).unwrap(), doc.value(1).unwrap());
                }
            }
            panic!("title doc {data} missing");
        };
        let (v0, v1) = doc_by_data("C/Alt_Banana");
        assert_eq!(v0, b"Alternate Banana".to_vec());
        assert_eq!(v1, b"Banana".to_vec());
        let (v0, v1) = doc_by_data("C/Apple_fruit");
        assert_eq!(v0, b"Apple fruit".to_vec());
        assert_eq!(v1, b"Apple".to_vec());

        // Redirect titles joined the listing too.
        let listing = blob_of(z, b'X', "listing/titleOrdered/v1").unwrap();
        assert_eq!(listing.len() / 4, 5);
    }

    #[test]
    fn fulltext_index_is_folded_and_french_stemmed() {
        let c = convert_fixture(-1, false, false);
        let z = &c.zim;
        let ft = z.open_fulltext_xapian().unwrap().unwrap();
        // One doc per converted article, in conversion order.
        assert_eq!(ft.doc_count(), 3);
        assert_eq!(ft.get_metadata("kind").unwrap(), "fulltext");
        assert_eq!(ft.get_metadata("language").unwrap(), "fra");
        assert_eq!(
            ft.get_metadata("valuesmap").unwrap(),
            "title:0;wordcount:1;geo.position:2"
        );
        assert_eq!(ft.get_metadata("stopwords").unwrap(), "");

        let mut doc = ft.get_document(3).unwrap();
        assert_eq!(doc.data_str().unwrap(), "C/Révolution");
        // Value 0 is the FOLDED title, value 1 the wordcount of the folded
        // indexed content.
        assert_eq!(doc.value(0).unwrap(), b"revolution francaise");
        let md = crate::html2md::html_to_md(REV_HTML, Some("Révolution française"), Some("fra"));
        let wordcount = crate::search::fold_accents(&md).split_whitespace().count();
        assert_eq!(doc.value(1).unwrap(), wordcount.to_string().into_bytes());

        // Accents were folded away; the French stem (libzim's STEM_ALL)
        // carries the matches.
        assert_eq!(ft.termfreq("révolution"), 0);
        assert!(ft.termfreq("revolu") > 0);
        // STEM_ALL stores no surface forms in the fulltext index.
        assert_eq!(ft.termfreq("revolution"), 0);
        // Content words are searchable through the same stemmer the index
        // was built with (the stem's shape is the stemmer's business).
        let mut fr = xapian2::Stem::new("fr").unwrap();
        let stem = fr.apply("changement").unwrap();
        assert!(ft.termfreq(&stem) > 0, "stem of 'changement' = {stem:?}");
    }

    #[test]
    fn language_metadata_verbatim_and_fallback() {
        // The Language metadata is copied VERBATIM even when it is a
        // multi-code value; the indexing language is its first code.
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.zim");
        let mut content = vec![
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'M', url: "Language", title: "", mime: 2, body: b"fra,eng" },
        ];
        let redirects: Vec<TestRedirect> = vec![];
        let bytes = build_archive(
            &["text/html", "text/css", "text/plain;charset=UTF-8", "image/png"],
            &content,
            &redirects,
            0,
            None,
        );
        std::fs::write(&src, bytes).unwrap();
        let out = dir.path().join("out.zim");
        convert(&src, &out, -1, false, false).unwrap();
        let z = Zim::open(&out).unwrap();
        assert_eq!(blob_of(&z, b'M', "Language"), Some(b"fra,eng".to_vec()));
        let ft = z.open_fulltext_xapian().unwrap().unwrap();
        // The raw (first) code, not the ICU-mapped one, is stored.
        assert_eq!(ft.get_metadata("language").unwrap(), "fra");

        // Without any Language metadata the normalized default is written.
        content.pop();
        let bytes = build_archive(
            &["text/html", "text/css", "text/plain;charset=UTF-8", "image/png"],
            &content,
            &redirects,
            0,
            None,
        );
        std::fs::write(&src, bytes).unwrap();
        let out = dir.path().join("out2.zim");
        convert(&src, &out, -1, false, false).unwrap();
        let z = Zim::open(&out).unwrap();
        assert_eq!(blob_of(&z, b'M', "Language"), Some(b"eng".to_vec()));
        let ft = z.open_fulltext_xapian().unwrap().unwrap();
        assert_eq!(ft.get_metadata("language").unwrap(), "eng");
    }

    #[test]
    fn intro_only_indexes_only_the_intro() {
        // Full content: the History section's "Kazakhstan" is indexed
        // (termfreq is a document frequency; the stem is computed with the
        // archive's own stemmer - the fixture is English text in a fra
        // archive, so surface forms don't survive STEM_ALL).
        let mut fr = xapian2::Stem::new("fr").unwrap();
        let kaz = fr.apply("kazakhstan").unwrap();
        let full = convert_fixture(-1, false, false);
        let ft = full.zim.open_fulltext_xapian().unwrap().unwrap();
        assert_eq!(ft.termfreq(&kaz), 1);
        assert!(ft.termfreq(&fr.apply("apple").unwrap()) >= 1);

        // Intro-only: the lead section only - title line kept, everything
        // after the first "## " heading gone.
        let intro = convert_fixture(-1, true, false);
        let ft = intro.zim.open_fulltext_xapian().unwrap().unwrap();
        assert_eq!(ft.termfreq(&kaz), 0);
        // The title line keeps the article searchable in the lead.
        assert!(ft.termfreq(&fr.apply("apple").unwrap()) >= 1);

        // The wordcount shrank with the indexed text.
        let mut doc = ft.get_document(1).unwrap();
        let wc_intro = String::from_utf8(doc.value(1).unwrap()).unwrap();
        let mut doc = full.zim.open_fulltext_xapian().unwrap().unwrap().get_document(1).unwrap();
        let wc_full = String::from_utf8(doc.value(1).unwrap()).unwrap();
        assert!(
            wc_intro.parse::<usize>().unwrap() < wc_full.parse::<usize>().unwrap(),
            "intro wordcount {wc_intro} vs full {wc_full}"
        );
    }

    const STUB_HTML: &str = "<html><body><a href=\"./Apple#f\">label</a></body></html>";

    /// One normal article (a `<p>` body) plus one stub-shaped article (a
    /// bare internal `<a>` as the only body child — mwoffliner's
    /// meta-refresh redirect stub).
    fn stub_fixture() -> Vec<u8> {
        let content = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Stub", title: "Stub", mime: 0, body: STUB_HTML.as_bytes() },
            TestEntry { namespace: b'M', url: "Language", title: "", mime: 2, body: b"eng" },
        ];
        build_archive(
            &["text/html", "text/css", "text/plain;charset=UTF-8", "image/png"],
            &content,
            &[],
            0,
            None,
        )
    }

    fn convert_stub_fixture(redirect_titles: bool) -> Converted {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.zim");
        std::fs::write(&src, stub_fixture()).unwrap();
        let out = dir.path().join("out.zim");
        convert(&src, &out, -1, false, redirect_titles).unwrap();
        let zim = Zim::open(&out).unwrap();
        Converted { _dir: dir, zim }
    }

    #[test]
    fn no_content_stub_is_not_indexed_by_default() {
        let c = convert_stub_fixture(false);
        let z = &c.zim;

        // The stub produces no fulltext and no title document: only the
        // normal article is indexed in both databases.
        let ft = z.open_fulltext_xapian().unwrap().unwrap();
        assert_eq!(ft.doc_count(), 1);
        assert_eq!(ft.get_document(1).unwrap().data_str().unwrap(), "C/Apple");
        let ti = z.open_title_xapian().unwrap().unwrap();
        assert_eq!(ti.doc_count(), 1);
        assert_eq!(ti.get_document(1).unwrap().data_str().unwrap(), "C/Apple");

        // The stub page still exists as text/markdown carrying its wikilink
        // line, and still resolves to that content.
        let idx = z.resolve_path("C/Stub").unwrap().unwrap();
        let entry = z.get_entry(idx).unwrap();
        assert_eq!(z.mime_type(entry.mime), Some("text/markdown"));
        let md = String::from_utf8(blob_of(z, b'C', "Stub").unwrap()).unwrap();
        assert_eq!(md, "# Stub\n\n[[Apple#f|label]]\n");
    }

    #[test]
    fn no_content_stub_title_doc_only_opt_in() {
        // Under --index-redirect-titles the stub gets a title document with
        // NO target path (its record's `a` holds a blob generation, not an
        // index): value 1 is its own path, like a regular article.
        let c = convert_stub_fixture(true);
        let z = &c.zim;
        assert_eq!(z.open_fulltext_xapian().unwrap().unwrap().doc_count(), 1);
        let ti = z.open_title_xapian().unwrap().unwrap();
        assert_eq!(ti.doc_count(), 2);
        let doc_by_data = |data: &str| -> (Vec<u8>, Vec<u8>) {
            for d in 1..=ti.doc_count() {
                let mut doc = ti.get_document(d).unwrap();
                if doc.data_str().unwrap() == data {
                    return (doc.value(0).unwrap(), doc.value(1).unwrap());
                }
            }
            panic!("title doc {data} missing");
        };
        assert_eq!(doc_by_data("C/Stub"), (b"Stub".to_vec(), b"Stub".to_vec()));
        assert_eq!(doc_by_data("C/Apple"), (b"Apple".to_vec(), b"Apple".to_vec()));
    }

    #[test]
    fn title_index_collapses_wordless_titles_like_libzim() {
        // A title made solely of non-word characters ("!=" - a real French
        // Wikipedia article) indexes to the anchor term alone, which libzim
        // swaps for the entire title as one term: it removes the FIRST
        // TERMLIST TERM (the stored "0posanchor" - the ANCHOR_TERM literal's
        // trailing space never survives tokenization) and adds the whole
        // title with the default wdf increment of 1. Removing the literal
        // "0posanchor " instead throws InvalidArgumentError and aborted real
        // conversions (found on wikipedia_fr_top_nopic).
        let dir = tempfile::tempdir().unwrap();
        let mut wdb = WritableDatabase::create_with_flags(
            dir.path().join("t.idx.tmp"),
            wdb_flags::DB_CREATE_OR_OVERWRITE | wdb_flags::DB_NO_TERMLIST,
        )
        .unwrap();
        let lang = IndexLang::new("eng".to_string());
        let doc = build_title_document(&lang, "!=", "!=", None).unwrap();
        wdb.add_document(&doc).unwrap();
        let doc = build_title_document(&lang, "Apple_Pie", "Apple Pie", None).unwrap();
        wdb.add_document(&doc).unwrap();
        wdb.commit().unwrap();
        wdb.compact_to_path(dir.path().join("t.idx")).unwrap();

        let db = xapian2::Database::open(&dir.path().join("t.idx")).unwrap();
        assert_eq!(db.doc_count(), 2);
        // The wordless title: collapsed to a single raw term, no anchor.
        assert_eq!(db.termfreq("!="), 1);
        // The re-added whole-title term carries libzim's add_term default
        // wdf of 1 (a 0 wdf would zero its BM25 contribution).
        assert_eq!(db.wdf(1, "!="), 1);
        // The anchor term survives only for the wordy title.
        assert_eq!(db.termfreq("0posanchor"), 1);
        // STEM_SOME keeps the surface form of wordy titles.
        assert_eq!(db.termfreq("apple"), 1);
        // Values: 0 = the raw title, 1 = the path (no redirect target).
        let mut doc = db.get_document(1).unwrap();
        assert_eq!(doc.data_str().unwrap(), "C/!=");
        assert_eq!(String::from_utf8(doc.value(0).unwrap()).unwrap(), "!=");
        assert_eq!(String::from_utf8(doc.value(1).unwrap()).unwrap(), "!=");
    }
}
