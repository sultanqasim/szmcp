//! `szmcp convert` — a faithful Rust port of `wikizim_parser/zim2zim.py`:
//! turn a Kiwix HTML ZIM into a ZIM of Markdown articles with fresh search
//! indexes.
//!
//! The conversion is one streaming pass over the source directory (path
//! order): every entry is either converted (text/html article → markdown
//! item), recreated (redirect whose chain resolves to a source HTML article)
//! or skipped and tallied by MIME. Fulltext and title Xapian indexes are
//! built as libzim 9.8.2 would build them (see `mcp_stuff/convert_notes.md`),
//! with one deliberate divergence: the fulltext documents are added in
//! deterministic conversion order, not libzim's worker-race order — the two
//! are equivalent as sets keyed by document data.

use crate::zim::{Target, Zim};
use crate::zimcommon::MIME_REDIRECT;
use crate::zimwrite::ZimCreator;
use std::collections::{HashMap, HashSet};
use std::path::Path;
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

/// Follow the redirect chain starting at `idx` in the SOURCE to the first
/// non-redirect entry (transitively, cycle-safe). `None` when the chain
/// leaves the entry range or cycles - a dangling redirect, like
/// `zim2zim.redirect_target`.
fn redirect_terminal(z: &Zim, idx: u32) -> Option<u32> {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut cur = idx;
    loop {
        if cur >= z.entry_count() || !seen.insert(cur) {
            return None;
        }
        let entry = z.get_entry(cur).ok()?;
        match entry.target {
            Target::Redirect(next) => cur = next,
            _ => return Some(cur),
        }
    }
}

/// The mime tally key for a skipped entry: the MIME string from the archive's
/// mime list, or "unknown/<id>".
fn skip_mime_key(z: &Zim, mime: u16) -> String {
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
fn add_fulltext_document(
    wdb: &mut WritableDatabase,
    lang: &IndexLang,
    path: &str,
    folded_title: &str,
    folded_content: &str,
) -> Result<(), String> {
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
    wdb.add_document(&doc).map_err(|e| e.to_string())?;
    Ok(())
}

/// One title-index document, the way libzim's XapianIndexer::indexTitle does:
/// data = "C/"+path, value 0 = the RAW title (NOT folded), value 1 = the
/// redirect target path (or the article's own path); the accent-folded title
/// indexed WITH positions behind libzim's anchor term, STEM_SOME,
/// FLAG_NGRAMS, a 240-character word cap. A title made solely of non-word
/// characters leaves only the anchor term: it is removed and the whole title
/// added as one term when it fits (libzim's collapse).
fn add_title_document(
    wdb: &mut WritableDatabase,
    lang: &IndexLang,
    path: &str,
    title: &str,
    target_path: Option<&str>,
) -> Result<(), String> {
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
    wdb.add_document(&doc).map_err(|e| e.to_string())?;
    Ok(())
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
/// gives recreated redirects the FRONT_ARTICLE hint so their titles enter
/// the title index (the default excludes them, zim2zim's
/// --no-redirect-titles behavior).
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

    // The two Xapian databases are built under a unique temp dir, compacted
    // to single files and streamed into the archive (no RAM concern). The
    // counter keeps concurrent conversions (test threads) apart.
    static TMP_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!(
        "szmcp-convert-{}-{n}-{}",
        std::process::id(),
        outfile.file_name().and_then(|n| n.to_str()).unwrap_or("zim")
    ));
    if tmp.exists() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    std::fs::create_dir_all(&tmp)
        .map_err(|e| format!("cannot create temp dir {}: {e}", tmp.display()))?;

    let outcome = run_pass(Pass {
        z: &z,
        zimfile,
        outfile,
        entry_count,
        limit_entries,
        index_intro_only,
        index_redirect_titles,
        conv_lang: &conv_lang,
        lang: &lang,
        main_entry_path: &main_entry_path,
        tmp: &tmp,
        t0,
    });
    let _ = std::fs::remove_dir_all(&tmp);
    outcome
}

/// Everything one conversion run needs, bundled to keep `run_pass`'s
/// signature flat.
struct Pass<'a> {
    z: &'a Zim,
    zimfile: &'a Path,
    outfile: &'a Path,
    entry_count: u32,
    limit_entries: u32,
    index_intro_only: bool,
    index_redirect_titles: bool,
    conv_lang: &'a str,
    lang: &'a IndexLang,
    main_entry_path: &'a Option<String>,
    tmp: &'a Path,
    t0: Instant,
}

/// One conversion pass: creator setup, the streaming scan, the two index
/// postludes and the finalization.
fn run_pass(p: Pass) -> Result<(), String> {
    let mut creator =
        ZimCreator::new(p.outfile).map_err(|e| format!("cannot create {}: {e}", p.outfile.display()))?;

    // Creator preamble (zim2zim's `with creator:` block): metadata and the
    // illustration are copied before the scan.
    let metadata_language = copy_metadata(p.z, &mut creator, p.conv_lang)?;
    copy_illustration(p.z, &mut creator)?;

    // Fulltext database prelude (xapianIndexer.cpp indexingPrelude, FULL
    // mode): libzim opens its throwaway database at <path>.tmp with
    // DB_CREATE_OR_OVERWRITE | DB_NO_TERMLIST and records the metadata.
    let ft_tmp = p.tmp.join("fulltext.idx.tmp");
    let ft_path = p.tmp.join("fulltext.idx");
    let mut ft_wdb = WritableDatabase::create_with_flags(
        &ft_tmp,
        wdb_flags::DB_CREATE_OR_OVERWRITE | wdb_flags::DB_NO_TERMLIST,
    )
    .map_err(|e| format!("fulltext index: {e}"))?;
    for (key, value) in [
        ("valuesmap", "title:0;wordcount:1;geo.position:2"),
        ("kind", "fulltext"),
        ("data", "fullPath"),
        ("language", p.lang.raw.as_str()),
        ("stopwords", ""),
    ] {
        ft_wdb
            .set_metadata(key, value)
            .map_err(|e| format!("fulltext index metadata {key}: {e}"))?;
    }

    eprintln!("Converting entries...");
    let mut converted = 0usize;
    let mut failed = 0usize;
    let mut written = 0usize;
    let mut processed = 0u32;
    let mut skipped_mimes: HashMap<String, usize> = HashMap::new();
    // Paths written as articles: a recreated redirect survives the creator's
    // dangling-redirect cleanup only when its terminal target is one of
    // these, and only surviving front articles make it into the title index.
    let mut written_paths: HashSet<String> = HashSet::new();
    // Title-index records (path, title, redirect target path): articles
    // always, redirects only with the FRONT_ARTICLE hint. Filtered to
    // survivors and built in sorted path order after the scan - libzim
    // iterates the sorted dirent set, so docids follow that order.
    let mut title_records: Vec<(String, String, Option<String>)> = Vec::new();
    let mut main_converted = false;
    let mut first_converted_path: Option<String> = None;

    let mut idx = 0u32;
    while idx < p.entry_count {
        let entry = p
            .z
            .get_entry(idx)
            .map_err(|e| format!("reading entry {idx}: {e}"))?;
        let item_path = entry.url.clone();
        let dirent_title = entry.title;
        let mime = entry.mime;
        if mime == MIME_REDIRECT {
            let title = entry_title(&dirent_title, &item_path);
            // Resolve the ultimate target through the redirect chain in the
            // SOURCE (transitively, cycle-safe): libzim resolves
            // redirect->redirect transitively, so the recreated redirect
            // must point at a real article of the new ZIM too. A redirect
            // that resolves to a non-HTML entry is silently skipped.
            let Target::Redirect(target_idx) = entry.target else {
                unreachable!("mime == MIME_REDIRECT implies a redirect target");
            };
            if let Some(t_idx) = redirect_terminal(p.z, target_idx) {
                let t_entry = p
                    .z
                    .get_entry(t_idx)
                    .map_err(|e| format!("reading redirect target of {item_path:?}: {e}"))?;
                if p.z.mime_type(t_entry.mime).is_some_and(|m| m.starts_with("text/html")) {
                    // Without FRONT_ARTICLE the redirect still resolves but
                    // its title stays out of the title index.
                    if let Err(e) = creator.add_redirection(
                        &item_path,
                        &title,
                        &t_entry.url,
                        p.index_redirect_titles,
                    ) {
                        eprintln!("warning: redirect {item_path:?}: {e}");
                    } else if p.index_redirect_titles {
                        title_records.push((item_path.clone(), title, Some(t_entry.url.clone())));
                    }
                }
            }
        } else if p.z.mime_type(mime).is_some_and(|m| m.starts_with("text/html")) {
            // An HTML article: read, convert, add, fulltext-index.
            let title = entry_title(&dirent_title, &item_path);
            let html = match entry.target {
                Target::Cluster(cluster, blob) => p.z.read_blob(cluster, blob),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "entry has no content",
                )),
            };
            match html.ok().and_then(|bytes| String::from_utf8(bytes).ok()) {
                Some(html) => {
                    let md = crate::html2md::html_to_md(&html, Some(&title), Some(p.conv_lang));
                    creator
                        .add_item(
                            &item_path,
                            &title,
                            "text/markdown",
                            true,
                            true,
                            md.clone().into_bytes(),
                        )
                        .map_err(|e| format!("item {item_path:?}: {e}"))?;
                    let indexed = if p.index_intro_only {
                        intro_for_index(&md)
                    } else {
                        md.clone()
                    };
                    let folded_title = crate::search::fold_accents(&title);
                    let folded_content = crate::search::fold_accents(&indexed);
                    add_fulltext_document(
                        &mut ft_wdb,
                        p.lang,
                        &item_path,
                        &folded_title,
                        &folded_content,
                    )
                    .map_err(|e| format!("indexing {item_path:?}: {e}"))?;
                    converted += 1;
                    written += md.len();
                    written_paths.insert(item_path.clone());
                    if first_converted_path.is_none() {
                        first_converted_path = Some(item_path.clone());
                    }
                    if p.main_entry_path.as_deref() == Some(item_path.as_str()) {
                        main_converted = true;
                    }
                    title_records.push((item_path, title, None));
                }
                None => {
                    failed += 1;
                    eprintln!(
                        "warning: failed to convert {item_path:?}: content is not valid UTF-8"
                    );
                }
            }
        } else {
            // Media/metadata/whatever: skipped, tallied by MIME.
            *skipped_mimes.entry(skip_mime_key(p.z, mime)).or_insert(0) += 1;
        }
        processed += 1;
        if processed % 100 == 0 {
            eprintln!(
                "[{}/{}] entries: {} articles converted ({:.1} MB written)",
                processed,
                p.limit_entries,
                converted,
                written as f64 / 1e6
            );
        }
        if processed >= p.limit_entries {
            break;
        }
        idx += 1;
    }

    // Main path, set late (after all items, before finalization): the main
    // entry's own path if its article was written, else the first written
    // article's path.
    let main_path = if main_converted {
        p.main_entry_path.clone()
    } else {
        None
    }
    .or_else(|| first_converted_path.clone());
    if let Some(mp) = &main_path {
        creator.set_main_path(mp);
    }

    eprintln!();
    eprintln!("Writing Xapian indexes and finalizing (this may take a while)...");

    // Fulltext postlude: commit + compact to the single-file database libzim
    // embeds (DBCOMPACT_SINGLE_FILE | FULL); an index with no documents is
    // not embedded at all.
    if converted > 0 {
        ft_wdb.commit().map_err(|e| format!("fulltext index commit: {e}"))?;
        ft_wdb
            .compact_to_path(&ft_path)
            .map_err(|e| format!("fulltext index compact: {e}"))?;
        creator
            .add_xapian_index("fulltext/xapian", &ft_path)
            .map_err(|e| format!("fulltext index: {e}"))?;
    }

    // Title index postlude: only the front-article dirents that survived the
    // dangling-redirect cleanup (live targets), in sorted path order.
    let mut records: Vec<(String, String, Option<String>)> = title_records
        .into_iter()
        .filter(|(_, _, target)| match target {
            Some(target) => written_paths.contains(target),
            None => true,
        })
        .collect();
    records.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    if !records.is_empty() {
        let ti_tmp = p.tmp.join("title.idx.tmp");
        let ti_path = p.tmp.join("title.idx");
        let mut ti_wdb = WritableDatabase::create_with_flags(
            &ti_tmp,
            wdb_flags::DB_CREATE_OR_OVERWRITE | wdb_flags::DB_NO_TERMLIST,
        )
        .map_err(|e| format!("title index: {e}"))?;
        for (key, value) in [
            ("valuesmap", "title:0;targetPath:1"),
            ("kind", "title"),
            ("data", "fullPath"),
            ("language", p.lang.raw.as_str()),
            ("stopwords", ""),
        ] {
            ti_wdb
                .set_metadata(key, value)
                .map_err(|e| format!("title index metadata {key}: {e}"))?;
        }
        for (path, title, target) in &records {
            add_title_document(&mut ti_wdb, p.lang, path, title, target.as_deref())
                .map_err(|e| format!("title indexing {path:?}: {e}"))?;
        }
        ti_wdb.commit().map_err(|e| format!("title index commit: {e}"))?;
        ti_wdb
            .compact_to_path(&ti_path)
            .map_err(|e| format!("title index compact: {e}"))?;
        creator
            .add_xapian_index("title/xapian", &ti_path)
            .map_err(|e| format!("title index: {e}"))?;
    }

    creator
        .finish()
        .map_err(|e| format!("finalizing {}: {e}", p.outfile.display()))?;

    // Summary (stderr, like all human output of this subcommand).
    let out_size = std::fs::metadata(p.outfile).map(|m| m.len()).unwrap_or(0);
    let input_size = std::fs::metadata(p.zimfile).map(|m| m.len()).unwrap_or(0);
    let elapsed = p.t0.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("{}", "=".repeat(60));
    eprintln!("SUMMARY");
    eprintln!("  entries processed  : {processed} of {}", p.entry_count);
    eprintln!("  articles converted : {converted}");
    eprintln!("  language           : {} (metadata: {})", p.conv_lang, metadata_language);
    eprintln!("  conversion failures: {failed}");
    eprintln!("  skipped entries by MIME:");
    let mut skipped: Vec<(&String, &usize)> = skipped_mimes.iter().collect();
    skipped.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (mime, count) in skipped {
        eprintln!("    {mime:<30} {count}");
    }
    eprintln!("  input size : {:.1} MB", input_size as f64 / 1e6);
    eprintln!("  output size: {:.1} MB", out_size as f64 / 1e6);
    eprintln!("  elapsed    : {elapsed:.1} s");
    eprintln!("{}", "=".repeat(60));
    Ok(())
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
        // in sorted path order: Alt_Banana, Apple, Apple_fruit, Banana,
        // Révolution.
        assert_eq!(title.doc_count(), 5);
        let mut doc = title.get_document(1).unwrap();
        assert_eq!(doc.data_str().unwrap(), "C/Alt_Banana");
        assert_eq!(doc.value(0).unwrap(), b"Alternate Banana");
        assert_eq!(doc.value(1).unwrap(), b"Banana");
        let mut doc = title.get_document(3).unwrap();
        assert_eq!(doc.data_str().unwrap(), "C/Apple_fruit");
        assert_eq!(doc.value(0).unwrap(), b"Apple fruit");
        assert_eq!(doc.value(1).unwrap(), b"Apple");

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
        add_title_document(&mut wdb, &lang, "!=", "!=", None).unwrap();
        add_title_document(&mut wdb, &lang, "Apple_Pie", "Apple Pie", None).unwrap();
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
