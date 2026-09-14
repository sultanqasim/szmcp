//! The search pipeline behind the `zim_search` tool: query parsing and
//! ranking against the ZIM full-text indexes, cross-archive merging, and
//! the per-hit preview/section reporting.

use crate::html;
use crate::markdown;
use crate::tools::ToolError;
use crate::zim::{Archive, ZimLibrary};
use schemars::JsonSchema;
use serde::Serialize;
use std::sync::Arc;
use xapian2::{Enquire, Operator, Query, QueryParser, Stem, StemStrategy};

/// Number of results `zim_search` returns in total (across all archives).
const SEARCH_LIMIT: u32 = 20;
/// Maximum characters of the `preview` reported per search hit.
const INTRO_CHARS: usize = 300;
/// How many raw bytes of an article are read to locate the query's matches
/// in it (region and paragraph level). Matching needs the whole article, not
/// just the lead the old 64 KiB intro preview covered; for compressed
/// clusters the whole cluster decompresses anyway.
const HIT_READ_BYTES: u64 = 1024 * 1024;
/// Cap on one paragraph's characters while scanning it for matches: long
/// paragraphs keep matching far into their text. A paragraph chosen for
/// reporting is truncated to `INTRO_CHARS` separately.
const PARA_MATCH_CHARS: usize = 2000;

/// Common English words dropped when collecting an all-terms query boost.
/// They are stopped at index time (libzim's TermGenerator), so they are
/// absent from the index terms and an AND over them would match nothing.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in",
    "is", "it", "its", "not", "of", "on", "or", "that", "the", "these", "this", "those", "to",
    "was", "were", "which", "with",
];

/// One search result.
#[derive(Serialize, JsonSchema, Debug)]
pub struct SearchHit {
    /// ZIM file name, relative to the ZIM directory
    pub zim: String,
    /// Path of the article inside the ZIM file
    pub path: String,
    /// Page/article title
    pub title: String,
    /// Preview of the article: the first paragraph when the query matches
    /// the title or that paragraph, otherwise the sentence with the most
    /// query matches, followed by its paragraph's next sentences up to the
    /// length cap
    pub preview: String,
    /// Names of the regions holding query matches - the intro listed as
    /// `_intro` first when it matched, then the sections in document order;
    /// absent when the query matches the title or the first intro paragraph
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sections: Option<Vec<String>>,
}

/// The search result set (best matches first).
#[derive(Serialize, JsonSchema)]
pub struct SearchResults {
    /// The search results
    pub results: Vec<SearchHit>,
}

/// The query's non-stopword terms, stemmed the way the ZIM full-text
/// indexes were built (unprefixed Porter2 stems, lowercased): split on
/// whitespace, keep alphanumeric characters only per word, lowercase, drop
/// stopwords, stem, dedupe (preserving first-occurrence order).
fn query_terms(query: &str, stem: &mut Stemmer) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    for word in query.split_whitespace() {
        // Punctuation never appears in index terms either, so strip it.
        let word: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        let word = word.to_lowercase();
        if word.is_empty() || STOPWORDS.contains(&word.as_str()) {
            continue;
        }
        // The index has no spelling data, so an unknown word just stems to
        // something that matches nothing; it cannot break anything here.
        let stemmed = stem.stem(&word).to_string();
        if !terms.contains(&stemmed) {
            terms.push(stemmed);
        }
    }
    terms
}

/// A stemmer that remembers the stem of every word it has seen. Natural
/// text repeats its words heavily, and every uncached stem crosses the
/// Xapian FFI; one instance serves a whole search (the query terms and all
/// hits' paragraph matchers), so repeats dominate after the first hit.
struct Stemmer {
    stem: Stem,
    cache: std::collections::HashMap<String, String>,
}

impl Stemmer {
    fn new(language: &str) -> xapian2::Result<Self> {
        Ok(Self {
            stem: Stem::new(language)?,
            cache: std::collections::HashMap::new(),
        })
    }

    /// The word's stem, cased and stemmed the way `query_terms` and the ZIM
    /// full-text indexes were built (lowercased Porter2 stems).
    fn stem(&mut self, word: &str) -> &str {
        if !self.cache.contains_key(word) {
            let lowered = word.to_lowercase();
            let stemmed = self.stem.apply(&lowered).unwrap_or(lowered);
            self.cache.insert(word.to_string(), stemmed);
        }
        self.cache[word].as_str()
    }
}

/// Search all articles in all ZIM files of the library - the pipeline behind
/// the `zim_search` tool: ranked hits, best first.
pub fn search(library: &ZimLibrary, query: &str) -> Result<SearchResults, ToolError> {
    if query.trim().is_empty() {
        return Err(ToolError::InvalidArgument("query must not be empty".into()));
    }

    let mut qp = QueryParser::new()?;
    // openZIM's full-text indexes contain unprefixed Porter2/English stems
    // (libzim indexes with STEM_ALL), so queries must be stemmed the same
    // way - Xapian's default strategy would turn lowercase terms into
    // "Z"-prefixed stem terms that never match. Default combining op is OR.
    qp.set_stemmer("english")?;
    qp.set_stemming_strategy(StemStrategy::All)?;
    qp.set_default_op(Operator::Or)?;
    let xquery = qp
        .parse_query(query)
        .map_err(|e| ToolError::InvalidArgument(format!("failed to parse query: {e}")))?;

    // Multi-word queries: rank articles containing ALL the words above
    // articles containing only some of them, by ORing the parsed query with
    // an AND over the stemmed terms (the classic Xapian all-terms boost).
    // BM25 sums contributions across OR branches, so an all-words article
    // scores higher than one with a subset, while partial matches still
    // appear. Positional operators cannot approximate exact matching here -
    // the ZIM indexes carry no positional data, so OP_PHRASE/OP_NEAR match
    // nothing - and the indexes have no spelling data, so a misspelled word
    // simply matches nothing in the AND branch while the parsed OR branch
    // still retrieves results for the good words.
    // The query's terms drive two things: the all-terms boost below and the
    // paragraph matching when the hits are built - one stemmer serves both.
    let mut stemmer = Stemmer::new("english")?;
    let terms = query_terms(query, &mut stemmer);
    let xquery = if terms.len() >= 2 {
        // Fold the AND left to right; Xapian flattens the tree itself.
        let mut all_terms = Query::term(&terms[0])?;
        for term in &terms[1..] {
            all_terms = Query::combine(Operator::And, &all_terms, &Query::term(term)?)?;
        }
        Query::combine(Operator::Or, &all_terms, &xquery)?
    } else {
        xquery
    };

    // Exact title/URL matches, found in the ZIM directory itself: redirects
    // are not in the full-text index, and a query that names an article
    // exactly must rank first no matter what BM25 produces. One probe per
    // archive; a failed probe simply contributes nothing. The title falls
    // back to the query (spaces restored) because modern openZIM archives
    // leave directory-entry titles empty.
    let mut merged: Vec<(&Arc<Archive>, String, String, bool)> = Vec::new();
    for arc in &library.archives {
        if let Some((path, title)) = arc.lookup_exact(query)? {
            let title = if title.is_empty() { query.replace('_', " ") } else { title };
            merged.push((arc, path, title, true));
        }
    }

    // Per-archive ranked hit lists: (weight, path, title from the index).
    // libzim stores the article title in Xapian value slot 0.
    let mut per_archive: Vec<(&Arc<Archive>, Vec<(f64, String, String)>)> = Vec::new();
    for arc in &library.archives {
        let Some(db) = arc.xapian_db()? else {
            continue;
        };
        let mut enquire = Enquire::new(&db)?;
        enquire.set_query(&xquery)?;
        enquire.set_sort_by_relevance();
        let mset = enquire.get_mset(0, SEARCH_LIMIT, 0)?;
        let mut list = Vec::with_capacity(mset.size() as usize);
        for (j, m) in mset.iter().enumerate() {
            let mut doc = mset.document(j as u32)?;
            let path = doc.data_str()?;
            if path.is_empty() {
                continue;
            }
            let title = String::from_utf8_lossy(&doc.value(0)?).into_owned();
            list.push((m.weight, path, title));
        }
        per_archive.push((arc, list));
    }

    if per_archive.is_empty() {
        return Err(ToolError::Internal(format!(
            "no ZIM files with a Xapian full-text index were found in {}",
            library.root.display()
        )));
    }

    // Xapian weights are computed from per-database statistics and are not
    // comparable across archives, so merge the archives' ranked lists by
    // rotation instead of by weight: every archive contributes its best
    // match before any archive contributes its second best. Exact matches
    // were already placed first in `merged`.
    let mut rank = 0usize;
    loop {
        let mut picked = false;
        for (arc, list) in &per_archive {
            if let Some((_, path, title)) = list.get(rank) {
                merged.push((arc, path.clone(), title.clone(), false));
                picked = true;
            }
        }
        if !picked {
            break;
        }
        rank += 1;
    }

    // The same article is often present in several archives (e.g. an HTML
    // and a Markdown edition of the same ZIM): dedupe by normalized title
    // so each article is reported once, from the archive ranked first.
    // Exact matches come first, so duplicates of them drop out here.
    let mut seen = std::collections::HashSet::new();
    merged.retain(|(_, path, title, _)| {
        let key = if title.is_empty() { path.as_str() } else { title.as_str() };
        seen.insert(html::normalize(key))
    });
    merged.truncate(SEARCH_LIMIT as usize);

    let mut hits = Vec::with_capacity(merged.len());
    for (arc, path, idx_title, exact) in &merged {
        let (entry_title, mime, bytes) = match arc.article_preview(path, HIT_READ_BYTES) {
            Ok(Some((entry_title, mime, bytes))) => (entry_title, mime, bytes),
            _ => (String::new(), None, Vec::new()),
        };
        // An exact match's title is already final - the redirect's own
        // title, not the target's (which `entry_title` is). For the rest,
        // prefer the entry's own title; many openZIM archives leave the
        // directory-entry title empty and only carry the title in the
        // index (which we already read as `idx_title`).
        let title = if *exact {
            idx_title.clone()
        } else if !entry_title.is_empty() {
            entry_title
        } else if !idx_title.is_empty() {
            idx_title.clone()
        } else {
            path.clone()
        };
        let article = String::from_utf8_lossy(&bytes);
        // Markdown editions carry plain Markdown, not HTML: pick the matching
        // splitter so the paragraphs and section names are free of markup.
        let is_markdown = mime.as_deref().is_some_and(|m| m.contains("markdown"));
        let (preview, sections) = hit_preview(&article, &terms, *exact, &mut stemmer, is_markdown);
        hits.push(SearchHit {
            zim: arc.name.clone(),
            path: path.clone(),
            title,
            preview,
            sections,
        });
    }
    Ok(SearchResults { results: hits })
}

/// How many of `text`'s word occurrences are query terms - the paragraph's
/// match count, stemmed the same way the index and the query are.
fn para_matches(text: &str, terms: &[String], stem: &mut Stemmer) -> usize {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !STOPWORDS.iter().any(|s| s.eq_ignore_ascii_case(w)))
        .filter(|w| terms.iter().any(|t| t == stem.stem(w)))
        .count()
}

/// Whether every query term occurs (stemmed) somewhere in `text`'s words.
fn covers_all_terms(text: &str, terms: &[String], stem: &mut Stemmer) -> bool {
    let mut covered = vec![false; terms.len()];
    let mut left = terms.len();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() || STOPWORDS.iter().any(|s| s.eq_ignore_ascii_case(word)) {
            continue;
        }
        let stemmed = stem.stem(word);
        for (k, term) in terms.iter().enumerate() {
            if !covered[k] && term.as_str() == stemmed {
                covered[k] = true;
                left -= 1;
                if left == 0 {
                    return true;
                }
            }
        }
    }
    left == 0
}

/// Split a paragraph into sentences: a sentence ends after `.`, `!`, or `?`
/// followed by whitespace or the paragraph's end; a paragraph without any
/// terminator is a single sentence. (Abbreviations like "U.S." over-split,
/// which is acceptable for a preview.)
fn sentences(paragraph: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in paragraph.char_indices() {
        if matches!(c, '.' | '!' | '?') {
            let after = i + c.len_utf8();
            if after == paragraph.len() || paragraph[after..].starts_with(char::is_whitespace) {
                out.push(paragraph[start..after].trim());
                start = after;
            }
        }
    }
    if start < paragraph.len() {
        out.push(paragraph[start..].trim());
    }
    out.into_iter().filter(|s| !s.is_empty()).collect()
}

/// The `preview`/`sections` pair of one search hit, from the article's raw
/// text (`is_markdown` picks the Markdown or the HTML splitter):
///
/// - a hit whose title matched exactly is a title match by definition: its
///   preview is the first intro paragraph, no sections;
/// - otherwise, when every query term occurs in the first intro paragraph,
///   same: the lead already covers the query;
/// - otherwise every region is scanned - the intro first (under its
///   `_intro` name, `html::INTRO_SECTION`), then the body sections in
///   document order. The regions holding at least one query match are
///   reported as `sections`, and the preview is the best-matching sentence
///   (most matching word occurrences across all regions; ties keep the
///   earliest, so an intro sentence beats a body one), followed by its
///   paragraph's next sentences while the length stays under `INTRO_CHARS`
///   and truncated to `INTRO_CHARS` - the matched sentence sits at the
///   front, so the truncation cannot hide the words that matched;
/// - when no region matches either (only the title in the index matched
///   the query), the preview falls back to the first intro paragraph. Regions
///   without paragraphs degrade to an empty preview, never a panic.
///
/// The intro's paragraphs are extracted first and alone: the two title/lead
/// cases above - the common ones - never need the full article split.
fn hit_preview(
    article: &str,
    terms: &[String],
    exact: bool,
    stem: &mut Stemmer,
    is_markdown: bool,
) -> (String, Option<Vec<String>>) {
    let intro = if is_markdown {
        markdown::intro_paragraphs(article, PARA_MATCH_CHARS)
    } else {
        html::intro_paragraphs(article, PARA_MATCH_CHARS)
    };
    let lead = || {
        intro.first()
            .map(|p| p.chars().take(INTRO_CHARS).collect())
            .unwrap_or_default()
    };
    if exact || intro.first().is_some_and(|p| covers_all_terms(p, terms, stem)) {
        return (lead(), None);
    }
    let secs = if is_markdown {
        markdown::sections(article, PARA_MATCH_CHARS)
    } else {
        html::sections(article, PARA_MATCH_CHARS)
    };
    // Paragraphs are scored whole only for the `sections` reporting; the
    // preview picks the best-matching SENTENCE, so that a match in the
    // middle of a long paragraph is still visible in the preview.
    let mut best_count = 0usize;
    // The best-matching sentence, as (its paragraph, its index within it).
    let mut best_sent: Option<(&String, usize)> = None;
    let mut names: Vec<String> = Vec::new();
    for (name, paras) in &secs {
        let mut matched = false;
        for para in paras {
            if para_matches(para, terms, stem) > 0 {
                matched = true;
                for (i, s) in sentences(para).into_iter().enumerate() {
                    let n = para_matches(s, terms, stem);
                    if n > best_count {
                        // Ties keep the earlier sentence: only a strictly
                        // better count replaces the incumbent.
                        best_count = n;
                        best_sent = Some((para, i));
                    }
                }
            }
        }
        if matched {
            names.push(name.clone());
        }
    }
    match best_sent {
        Some((para, first)) => {
            let sent = sentences(para);
            let mut preview = String::new();
            for s in &sent[first..] {
                if preview.chars().count() >= INTRO_CHARS {
                    break;
                }
                if !preview.is_empty() {
                    preview.push(' ');
                }
                preview.push_str(s);
            }
            (preview.chars().take(INTRO_CHARS).collect(), Some(names))
        }
        None => (lead(), None),
    }
}

// ---------------------------------------------------------------------------
// Tests: end-to-end over a synthetic archive carrying a real Xapian index
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tools::{
        ZimGetParams, ZimGetSectionParams, ZimGetSectionTool, ZimGetTool, ZimMcpServer,
        ZimSearchParams, ZimSearchTool,
    };
    use crate::zim::testutil::{build_archive, TestEntry, TestRedirect};
    use rmcp::handler::server::router::tool::SyncTool;
    use xapian2::{Document, WritableDatabase};

    const APPLE_HTML: &str = "<html><head><title>Apple</title></head><body>\
        <h1>Apple</h1>\
        <p>An <b>apple</b> is the fruit of &lt;rosaceae&gt; trees.</p>\
        <h2 id=\"History\">History</h2>\
        <p>Apples have been cultivated for 10,000 years.</p>\
        <h3>Domestication</h3><p>Wild apples grew in Kazakhstan.</p>\
        <h2 id=\"Computers\">Computers</h2>\
        <p>Computing devices also go by that name.</p>\
        </body></html>";

    const BANANA_HTML: &str = "<html><body><h1>Banana</h1>\
        <h2 id=\"Growth\">Growth</h2>\
        <p>Banana trees are actually tall herbaceous plants.</p>\
        </body></html>";

    const CHERRY_HTML: &str = "<html><body><h1>Cherry</h1>\
        <p>A cherry is the fruit of trees of the genus <i>Prunus</i>.</p>\
        </body></html>";

    const NITROGEN_HTML: &str = "<html><body><h1>Nitrogen</h1>\
        <p>Nitrogen is a colorless, odorless gas.</p>\
        </body></html>";

    const ATMOSPHERE_HTML: &str = "<html><body><h1>Atmosphere</h1>\
        <p>The atmosphere is mostly nitrogen and oxygen.</p>\
        </body></html>";

    const AERONAUTICS_HTML: &str = "<html><body><h1>Aeronautics</h1>\
        <p>Aeronautics is the science of flight.</p>\
        </body></html>";

    /// An article whose intro has two paragraphs: a query can match the
    /// first paragraph (the lead fast path), a later one (the intro
    /// reported as the region `_intro`), or the intro and a body section.
    const SALT_HTML: &str = "<html><body><h1>Salt</h1>\
        <p>Salt is a mineral composed primarily of sodium chloride.</p>\
        <p>The Himalaya range holds vast deposits of rock salt.</p>\
        <h2>Formation</h2>\
        <p>Salt beds form when seas evaporate.</p>\
        <h2>Uses</h2>\
        <p>People season their food with it.</p>\
        </body></html>";

    /// Articles whose best-matching paragraph holds several sentences, with
    /// the query matching a mid-paragraph sentence: the preview must START
    /// with that sentence, which the old paragraph preview did not (the
    /// matched words sat mid-paragraph). The Volcano lead does not cover
    /// its query, so the lead fallback does not fire.
    const VOLCANO_HTML: &str = "<html><body><h1>Volcano</h1>\
        <p>Volcanoes are openings in the crust.</p>\
        <p>Molten rock rises from chambers below. Eruptions reshape the \
        land. Ash clouds can ground aircraft. Farmers fear the fallout.</p>\
        </body></html>";

    const GLACIER_MD: &str = "\
# Glacier

A glacier is a body of dense ice.

## Movement

Glaciers move under their own weight. The flow is slower than a river. \
Meltwater streams out of the ice.
";

    fn search(server: &ZimMcpServer, query: &str) -> Vec<SearchHit> {
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": query }),
        )
        .unwrap();
        ZimSearchTool::invoke(server, params).unwrap().results
    }

    /// Build a single-file glass Xapian index, the way openZIM does: the
    /// document data is the article's full path inside the archive, the
    /// title sits in value slot 0, and the terms are unprefixed Porter2
    /// stems, exactly as libzim indexes with STEM_ALL ("appl" is the stem
    /// of "apple", "comput" of "computing").
    fn make_index(docs: &[(&str, &str, &str)]) -> Vec<u8> {
        let dir = tempfile::tempdir().unwrap();
        let db_dir = dir.path().join("db");
        {
            let mut wdb = WritableDatabase::create(&db_dir).unwrap();
            for (path, terms, title) in docs {
                let mut doc = Document::new().unwrap();
                doc.set_data(*path).unwrap();
                if !title.is_empty() {
                    doc.set_value(0, *title).unwrap();
                }
                for t in terms.split_whitespace() {
                    doc.add_term(t, 1).unwrap();
                }
                wdb.add_document(&doc).unwrap();
            }
            wdb.commit().unwrap();
        }
        let single = dir.path().join("single.xdb");
        let db = xapian2::Database::open(&db_dir).unwrap();
        db.compact_single_file(&single).unwrap();
        std::fs::read(&single).unwrap()
    }

    pub(crate) fn test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Apple", "appl histori 10 000 year domest wild kazakhstan comput devic nam", "Apple"),
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search() {
        let (server, _keep) = test_server();

        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "apple" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert!(!hits.is_empty(), "search must return hits");
        let first = &hits[0];
        assert_eq!(first.zim, "test.zim");
        assert_eq!(first.path, "C/Apple");
        assert_eq!(first.title, "Apple");
        // An exact title match reports the lead paragraph as its preview.
        assert_eq!(first.preview, "An apple is the fruit of <rosaceae> trees.");
        assert_eq!(first.sections, None);

        // Stemmed query ("computing" -> "comput").
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "computing" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "C/Apple");

        // Unrelated term: no hits.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "zzzzz" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert!(hits.is_empty());

        // OR semantics: two terms from different articles.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "banana apple" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn e2e_search_empty_query_is_invalid() {
        let (server, _keep) = test_server();
        let params = ZimSearchParams { query: "   ".into() };
        assert!(matches!(
            ZimSearchTool::invoke(&server, params),
            Err(ToolError::InvalidArgument(_))
        ));
    }

    #[test]
    fn e2e_search_title_falls_back_to_index_title() {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Nitrogen", "nitrogen gas inert", "Nitrogen"),
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
        ]);
        let content = [
            // Empty directory-entry title, as in modern openZIM archives.
            TestEntry { namespace: b'C', url: "Nitrogen", title: "", mime: 0, body: NITROGEN_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        let server = ZimMcpServer::new(library);

        // Not an exact title/URL match ("nitrogen gas" is nobody's title),
        // so the hit comes from the full-text index: with the directory
        // title empty, the title falls back to the index title (value slot
        // 0). (The plain query "nitrogen" now resolves as an exact match.)
        let hits = search(&server, "nitrogen gas");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Nitrogen", "{hits:?}");
    }

    /// An archive where BM25 alone ranks the wrong article first: the
    /// "Atmosphere" document repeats the term "nitrogen" seven times, so it
    /// out-scores the "Nitrogen" article for the query "nitrogen". "NACA"
    /// and "Usa" are redirects onto the Aeronautics article; redirect
    /// entries live only in the directory, not in the search index.
    fn exact_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        // Index terms are the stems the query parser produces ("atmosphere"
        // -> "atmospher"), unprefixed, as libzim indexes with STEM_ALL.
        let index = make_index(&[
            ("C/Nitrogen", "nitrogen colorless odorless gas", "Nitrogen"),
            ("C/Atmosphere", "nitrogen nitrogen nitrogen nitrogen nitrogen nitrogen nitrogen naca naca atmospher", "Atmosphere"),
            ("C/Aeronautics", "aeronautics naca aviation wind tunnel flight", "Aeronautics"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Nitrogen", title: "Nitrogen", mime: 0, body: NITROGEN_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Atmosphere", title: "Atmosphere", mime: 0, body: ATMOSPHERE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Aeronautics", title: "Aeronautics", mime: 0, body: AERONAUTICS_HTML.as_bytes() },
        ];
        let redirects = [
            TestRedirect { namespace: b'C', url: "NACA", title: "NACA", target_content: 2 },
            // Empty directory title, as in modern openZIM archives.
            TestRedirect { namespace: b'C', url: "Usa", title: "", target_content: 2 },
        ];
        let bytes = build_archive(&["text/html"], &content, &redirects, 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_exact_title_ranks_first() {
        let (server, _keep) = exact_test_server();

        // "nitrogen" is exactly the title/URL of C/Nitrogen, yet BM25 ranks
        // the Atmosphere document first (it repeats the term seven times):
        // the exact match must come out on top.
        let hits = search(&server, "nitrogen");
        assert_eq!(hits[0].path, "C/Nitrogen", "{hits:?}");
        assert_eq!(hits[0].title, "Nitrogen");
        assert_eq!(hits[0].zim, "test.zim");
        // An exact match is a title match: the lead paragraph, no sections.
        assert_eq!(hits[0].preview, "Nitrogen is a colorless, odorless gas.");
        assert_eq!(hits[0].sections, None);
        assert!(!serde_json::to_string(&hits[0]).unwrap().contains("sections"));
        // The BM25 runner-up is still reported, behind the exact match.
        assert_eq!(hits[1].path, "C/Atmosphere", "{hits:?}");
    }

    #[test]
    fn e2e_search_exact_redirect_title_ranks_first() {
        let (server, _keep) = exact_test_server();

        // "NACA" is a redirect (directory title "NACA") onto the Aeronautics
        // article. Redirects are not in the full-text index, so without the
        // directory lookup this query would report Atmosphere first (it
        // mentions "naca" twice).
        let hits = search(&server, "NACA");
        assert_eq!(hits[0].path, "C/NACA", "{hits:?}");
        assert_eq!(hits[0].title, "NACA");
        // The preview is built from the redirect target's content.
        assert_eq!(hits[0].preview, "Aeronautics is the science of flight.");
        assert_eq!(hits[0].sections, None);
        // Fulltext hits follow in BM25 order.
        assert_eq!(hits[1].path, "C/Atmosphere", "{hits:?}");
        assert_eq!(hits[2].path, "C/Aeronautics", "{hits:?}");

        // A redirect with an empty directory title: the query becomes the
        // title, and the all-lowercase query still finds the redirect via
        // the case variants of its URL.
        let hits = search(&server, "usa");
        assert_eq!(hits[0].path, "C/Usa", "{hits:?}");
        assert_eq!(hits[0].title, "usa");
    }

    #[test]
    fn e2e_search_query_without_exact_match_keeps_ranking() {
        let (server, _keep) = exact_test_server();

        // "nitrogen atmosphere" is nobody's title or URL, so the ranking is
        // the unchanged BM25 order: the document containing both terms
        // first, the one containing only "nitrogen" second.
        let hits = search(&server, "nitrogen atmosphere");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].path, "C/Atmosphere", "{hits:?}");
        assert_eq!(hits[1].path, "C/Nitrogen", "{hits:?}");
        // The Atmosphere lead covers the whole query ("The atmosphere is
        // mostly nitrogen and oxygen.") in its first paragraph: an intro
        // match, so no sections. The Nitrogen lead only covers "nitrogen"
        // - a partial intro match is reported like any other, as the
        // region _intro.
        assert_eq!(hits[0].sections, None, "{:?}", hits[0]);
        assert_eq!(hits[0].preview, "The atmosphere is mostly nitrogen and oxygen.");
        assert_eq!(hits[1].sections, Some(vec!["_intro".to_string()]), "{:?}", hits[1]);
        assert_eq!(hits[1].preview, "Nitrogen is a colorless, odorless gas.");
    }

    #[test]
    fn e2e_search_intro_match_reports_lead_without_sections() {
        // Both query terms occur in the lead paragraph ("An apple is the
        // fruit of <rosaceae> trees."): an intro match on a full-text hit
        // (nobody's title is "apple fruit"), so the preview is the lead and
        // the serialized JSON carries no "sections" field at all.
        let (server, _keep) = test_server();
        let hits = search(&server, "apple fruit");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Apple");
        assert_eq!(hits[0].preview, "An apple is the fruit of <rosaceae> trees.");
        assert_eq!(hits[0].sections, None);
        let json = serde_json::to_string(&hits[0]).unwrap();
        assert!(!json.contains("sections"), "{json}");
    }

    #[test]
    fn e2e_search_section_match_reports_sections_and_best_paragraph() {
        // The query term appears only in a later section of the article
        // ("Wild apples grew in Kazakhstan." under History): the intro
        // cannot cover it, so the hit reports the matched sections and the
        // best-matching paragraph, not the lead.
        let (server, _keep) = test_server();
        let hits = search(&server, "kazakhstan");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Apple");
        // The paragraph sits under History, whose range includes the nested
        // Domestication heading: both sections report the match.
        assert_eq!(
            hit.sections,
            Some(vec!["History".to_string(), "Domestication".to_string()]),
            "{hit:?}"
        );
        assert_eq!(hit.preview, "Wild apples grew in Kazakhstan.");
        // The serialized JSON carries the section names.
        let json = serde_json::to_string(hit).unwrap();
        assert!(json.contains(r#""sections":["History","Domestication"]"#), "{json}");
    }

    /// An archive whose only article (`SALT_HTML`) has a two-paragraph
    /// intro, for the intro-matching search semantics.
    fn intro_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        // Index terms are the stems the query parser produces ("beds" ->
        // "bed"), unprefixed, as libzim indexes with STEM_ALL.
        let index = make_index(&[(
            "C/Salt",
            "salt mineral chlorid sodium himalaya deposit rock bed form sea season food",
            "Salt",
        )]);
        let content = [TestEntry {
            namespace: b'C',
            url: "Salt",
            title: "Salt",
            mime: 0,
            body: SALT_HTML.as_bytes(),
        }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_intro_match_beyond_first_paragraph_reports_intro_section() {
        // "himalaya" matches only the intro's second paragraph: the preview is
        // that paragraph and the intro is reported as the matching region
        // _intro - not the lead, and not without sections.
        let (server, _keep) = intro_test_server();
        let hits = search(&server, "himalaya");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Salt");
        assert_eq!(hit.sections, Some(vec!["_intro".to_string()]), "{hit:?}");
        assert_eq!(hit.preview, "The Himalaya range holds vast deposits of rock salt.");
        let json = serde_json::to_string(hit).unwrap();
        assert!(json.contains(r#""sections":["_intro"]"#), "{json}");
    }

    #[test]
    fn e2e_search_intro_and_section_matches_report_both() {
        // "salt beds" matches the intro's second paragraph ("salt") and the
        // Formation section ("Salt beds ..."): both regions are reported,
        // _intro first, and the preview is the paragraph with the most
        // query-term occurrences across all regions (Formation's, two
        // against the intro paragraphs' one).
        let (server, _keep) = intro_test_server();
        let hits = search(&server, "salt beds");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Salt");
        assert_eq!(
            hit.sections,
            Some(vec!["_intro".to_string(), "Formation".to_string()]),
            "{hit:?}"
        );
        assert_eq!(hit.preview, "Salt beds form when seas evaporate.");
    }

    #[test]
    fn e2e_search_preview_starts_with_best_sentence() {
        // The best-matching paragraph holds several sentences and the query
        // matches a mid-paragraph one: the preview starts with that sentence
        // (a 300-character preview of the whole paragraph would cut the
        // matched words off), continued with the paragraph's remaining
        // sentences. Covered for an HTML article (match in the intro's
        // second paragraph) and a Markdown one (match in a body section).
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Volcano", "volcano crust molten rock erupt reshape land ash cloud aircraft farmer", "Volcano"),
            ("C/Glacier", "glacier ice movement weight flow river meltwater stream", "Glacier"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Volcano", title: "Volcano", mime: 0, body: VOLCANO_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Glacier", title: "Glacier", mime: 1, body: GLACIER_MD.as_bytes() },
        ];
        let bytes = build_archive(&["text/html", "text/markdown"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        let server = ZimMcpServer::new(library);

        let hits = search(&server, "aircraft");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Volcano");
        assert_eq!(hits[0].sections, Some(vec!["_intro".to_string()]));
        assert!(
            hits[0].preview.starts_with("Ash clouds can ground aircraft."),
            "{:?}",
            hits[0].preview
        );
        assert_eq!(
            hits[0].preview,
            "Ash clouds can ground aircraft. Farmers fear the fallout."
        );

        let hits = search(&server, "river");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Glacier");
        assert_eq!(hits[0].sections, Some(vec!["Movement".to_string()]));
        assert!(
            hits[0].preview.starts_with("The flow is slower than a river."),
            "{:?}",
            hits[0].preview
        );
        assert_eq!(
            hits[0].preview,
            "The flow is slower than a river. Meltwater streams out of the ice."
        );
    }

    #[test]
    fn e2e_search_multi_word_ranks_all_words_first() {
        // BM25 alone ranks the "Cherry" document first: thirty repetitions
        // of one term in a short document outweigh a document mentioning
        // each term once. The all-terms AND branch must lift "Dessert
        // Recipes" (the only document with BOTH terms) above it, while the
        // partial match still appears. Two filler documents carry "pie" too,
        // so this is a real re-ranking, not a tie.
        let dir = tempfile::tempdir().unwrap();
        let cherri_terms = "cherri ".repeat(30);
        let dessert_terms = format!("cherri pie{}", " filler".repeat(8));
        let index = make_index(&[
            ("C/Cherry", &cherri_terms, "Cherry"),
            ("C/Dessert_Recipes", &dessert_terms, "Dessert Recipes"),
            ("C/Pie_1", "pie pie", "Pie 1"),
            ("C/Pie_2", "pie pie", "Pie 2"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Cherry", title: "Cherry", mime: 0, body: CHERRY_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Dessert_Recipes", title: "Dessert Recipes", mime: 0, body: CHERRY_HTML.as_bytes() },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        let server = ZimMcpServer::new(library);

        let hits = search(&server, "cherry pie");
        assert_eq!(hits[0].path, "C/Dessert_Recipes", "{hits:?}");
        assert_eq!(hits[0].title, "Dessert Recipes");
        // The partial match (only "cherry") is still reported, right behind.
        assert_eq!(hits[1].path, "C/Cherry", "{hits:?}");
        // Neither hit's lead covers both terms, and "cherry" does match the
        // intro: it is reported as the matching region _intro, and the preview
        // is the intro's matching paragraph.
        assert_eq!(hits[0].sections, Some(vec!["_intro".to_string()]));
        assert!(hits[0].preview.contains("cherry is the fruit"), "{:?}", hits[0].preview);
    }

    #[test]
    fn e2e_search_stopword_only_query_returns_nothing() {
        let (server, _keep) = test_server();

        // A query made only of stopwords matches nothing (they are stopped
        // at index time, so the terms are absent) and must not error.
        let hits = search(&server, "the in of");
        assert!(hits.is_empty(), "{hits:?}");
    }

    #[test]
    fn e2e_search_nonsense_word_still_returns_results() {
        let (server, _keep) = test_server();

        // The indexes have no spelling data, so a nonsense word matches
        // nothing: the all-terms AND branch is empty, but the parsed OR
        // branch still retrieves the results for the real word.
        let hits = search(&server, "apple zzzzqq");
        assert!(!hits.is_empty(), "{hits:?}");
        assert_eq!(hits[0].path, "C/Apple", "{hits:?}");
    }

    #[test]
    fn e2e_search_interleaves_archives_and_dedupes() {
        let dir = tempfile::tempdir().unwrap();
        // Two archives; both carry an "Apple" article (same article, as in an
        // HTML and a Markdown edition of the same ZIM), plus one exclusive
        // article each.
        let index_a = make_index(&[
            ("C/Apple", "appl histori 10 000 year domest wild kazakhstan comput devic nam", "Apple"),
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
        ]);
        let index_b = make_index(&[
            ("C/Apple", "appl comput devic nam", "Apple"),
            ("C/Cherry", "cherri pie fruit tree", "Cherry"),
        ]);
        let content_a = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
        ];
        let content_b = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Cherry", title: "Cherry", mime: 0, body: CHERRY_HTML.as_bytes() },
        ];
        std::fs::write(dir.path().join("a.zim"), build_archive(&["text/html"], &content_a, &[], 0, Some(&index_a))).unwrap();
        std::fs::write(dir.path().join("b.zim"), build_archive(&["text/html"], &content_b, &[], 0, Some(&index_b))).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert_eq!(library.archives.len(), 2);
        let server = ZimMcpServer::new(library);

        // The article present in both archives is reported exactly once.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "apple" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].zim, "a.zim");
        assert_eq!(hits[0].path, "C/Apple");

        // Distinct matches interleave: the best match of each archive first.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "banana cherry" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("a.zim", "C/Banana"));
        assert_eq!((hits[1].zim.as_str(), hits[1].path.as_str()), ("b.zim", "C/Cherry"));
    }

    #[test]
    fn e2e_search_and_get_single_file_library() {
        // A library opened from one ZIM file (no folder scan) behaves like a
        // scanned folder: the archive is addressed by its file name.
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[(
            "C/Salt",
            "salt miner primari sodium chlorid himalaya deposit rock",
            "Salt",
        )]);
        let content = [TestEntry {
            namespace: b'C',
            url: "Salt",
            title: "Salt",
            mime: 0,
            body: SALT_HTML.as_bytes(),
        }];
        let file = dir.path().join("one.zim");
        std::fs::write(&file, build_archive(&["text/html"], &content, &[], 0, Some(&index))).unwrap();
        let library = Arc::new(ZimLibrary::single(&file).unwrap());
        assert_eq!(library.archives.len(), 1);
        let server = ZimMcpServer::new(library);

        let hits = search(&server, "salt");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].zim, "one.zim");
        assert_eq!(hits[0].path, "C/Salt");

        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "one.zim", "path": "C/Salt" }),
        )
        .unwrap();
        let result = ZimGetTool::invoke(&server, params).unwrap();
        assert_eq!(result.title, "Salt");
        assert!(result.content.contains("sodium chloride"));
    }

    /// An article in the shape wikizim_parser emits (`text/markdown`).
    const ZINC_MD: &str = "\
# Zinc

*This article is about the element. For other uses, see [[Zinc (disambiguation)]].*

**Zinc** is a [[Chemical element|chemical element]] with the symbol **Zn**.

## History

Zinc smelting is documented in ancient times.

### India

Ancient India smelted zinc early.
";

    #[test]
    fn e2e_search_and_section_markdown() {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[("C/Zinc", "zinc chemic element symbol smelt ancient india", "Zinc")]);
        let content = [TestEntry {
            namespace: b'C',
            url: "Zinc",
            title: "Zinc",
            mime: 0,
            body: ZINC_MD.as_bytes(),
        }];
        let bytes = build_archive(&["text/markdown"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("md.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        let server = ZimMcpServer::new(library);

        // Search: the preview is plain text derived from the Markdown, free of
        // markup, and is the lead paragraph - the leading `# Zinc` title
        // line (a separate field of every hit) and the hatnote are dropped.
        // An exact title match never carries sections.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "zinc" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].zim, "md.zim");
        assert_eq!(hits[0].path, "C/Zinc");
        assert!(
            hits[0]
                .preview
                .starts_with("Zinc is a chemical element with the symbol Zn."),
            "{:?}",
            hits[0].preview
        );
        assert!(
            !hits[0].preview.contains("disambiguation")
                && !hits[0].preview.contains("**")
                && !hits[0].preview.contains("[[")
                && !hits[0].preview.contains('#'),
            "{:?}",
            hits[0].preview
        );
        assert_eq!(hits[0].sections, None);

        // A query matching only a body section reports the matched
        // sections, the nested one included, and the best-matching
        // paragraph; the intro holds no match, so _intro is absent.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "smelting" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].sections,
            Some(vec!["History".to_string(), "India".to_string()]),
            "{:?}",
            hits[0]
        );
        assert_eq!(hits[0].preview, "Zinc smelting is documented in ancient times.");
        // includes the subsection, reports the heading as written.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "history" }),
        )
        .unwrap();
        let result = ZimGetSectionTool::invoke(&server, params).unwrap();
        assert_eq!(result.section, "History");
        assert!(result.content.contains("ancient times"), "{:?}", result.content);
        assert!(result.content.contains("Ancient India smelted zinc early"));

        // The reserved intro name: the Markdown between the leading title
        // line and the first heading (hatnote and lead, raw), echoed as its
        // reserved name.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "_intro" }),
        )
        .unwrap();
        let result = ZimGetSectionTool::invoke(&server, params).unwrap();
        assert_eq!(result.section, "_intro");
        assert!(
            result.content.contains("For other uses, see [[Zinc (disambiguation)]]."),
            "{:?}",
            result.content
        );
        assert!(result.content.contains("with the symbol **Zn**"), "{:?}", result.content);
        assert!(!result.content.starts_with('#'), "{:?}", result.content);
        assert!(!result.content.contains("History"), "{:?}", result.content);

        // Missing section: same error shape as the HTML path.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "Nope" }),
        )
        .unwrap();
        assert!(matches!(
            ZimGetSectionTool::invoke(&server, params),
            Err(ToolError::SectionNotFound(_))
        ));
    }
}
