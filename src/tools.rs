//! MCP server definition: server struct, tool registration, and the three
//! ZIM tools (`zim_search`, `zim_get`, `zim_get_section`).

use crate::html;
use crate::markdown;
use crate::zim::{Archive, ZimLibrary};
use base64::Engine as _;
use rmcp::handler::server::router::tool::{SyncTool, ToolBase, ToolRouter};
use rmcp::handler::server::router::Router;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{Implementation, ServerInfo};
use rmcp::ErrorData;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;
use thiserror::Error;
use xapian2::{Enquire, Operator, Query, QueryParser, Stem, StemStrategy};

/// Number of results `zim_search` returns in total (across all archives).
const SEARCH_LIMIT: u32 = 20;
/// Maximum characters of the `text` reported per search hit.
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

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Xapian error: {0}")]
    Xapian(#[from] xapian2::Error),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Section not found: {0}")]
    SectionNotFound(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<ToolError> for ErrorData {
    fn from(err: ToolError) -> Self {
        match &err {
            ToolError::InvalidArgument(msg)
            | ToolError::NotFound(msg)
            | ToolError::SectionNotFound(msg) => ErrorData::invalid_params(msg.clone(), None),
            _ => ErrorData::internal_error(err.to_string(), None),
        }
    }
}

#[derive(Clone)]
pub struct ZimMcpServer {
    pub info: ServerInfo,
    pub library: Arc<ZimLibrary>,
}

impl ZimMcpServer {
    pub fn new(library: Arc<ZimLibrary>) -> Self {
        let mut info = ServerInfo::default();
        info.server_info = Implementation::new("szmcp", env!("CARGO_PKG_VERSION"));
        Self { info, library }
    }

    pub fn router(self) -> Router<ZimMcpServer> {
        let tool_router = ToolRouter::new()
            .with_sync_tool::<ZimSearchTool>()
            .with_sync_tool::<ZimGetTool>()
            .with_sync_tool::<ZimGetSectionTool>();

        let mut router = Router::new(self);
        router.tool_router = tool_router;
        router
    }
}

impl ServerHandler for ZimMcpServer {
    fn get_info(&self) -> ServerInfo {
        self.info.clone()
    }
}

/// Map a "article not found" I/O error to a proper MCP invalid-params error.
fn not_found_if_missing(
    result: std::result::Result<crate::zim::Article, std::io::Error>,
) -> Result<crate::zim::Article, ToolError> {
    result.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            ToolError::NotFound(e.to_string())
        } else {
            ToolError::Io(e)
        }
    })
}

/// Find the archive with the given name (relative to the ZIM directory).
fn find_archive(library: &ZimLibrary, name: &str) -> Result<Arc<Archive>, ToolError> {
    let wanted = name.trim().trim_start_matches("./");
    library
        .archives
        .iter()
        .find(|a| a.name == wanted)
        .cloned()
        .ok_or_else(|| {
            ToolError::NotFound(format!(
                "ZIM file not found: {name} (loaded: {})",
                library.archives.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", ")
            ))
        })
}

// ---------------------------------------------------------------------------
// zim_search
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimSearchParams {
    /// The search string to look for in all articles of all ZIM files
    pub query: String,
}

/// One search result.
#[derive(Serialize, JsonSchema, Debug)]
pub struct SearchHit {
    /// ZIM file name, relative to the ZIM directory
    pub zim: String,
    /// Path of the article inside the ZIM file
    pub path: String,
    /// Page/article title
    pub title: String,
    /// First paragraph of the article when the query matches the title or
    /// that paragraph, otherwise the paragraph with the most query matches
    pub text: String,
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

pub struct ZimSearchTool;

impl ToolBase for ZimSearchTool {
    type Parameter = ZimSearchParams;
    type Output = SearchResults;
    type Error = ToolError;

    fn name() -> Cow<'static, str> {
        "zim_search".into()
    }
    fn description() -> Option<Cow<'static, str>> {
        Some(
            "Search all articles in all ZIM files. Results are ranked best first (an exact \
             title match always comes first); each result has the ZIM file name, the article \
             path, the page title, and text - the article's first paragraph when the query \
             matches the title or that paragraph, otherwise the paragraph that best matches \
             the query together with \"sections\", the matching regions' names (the intro \
             listed as \"_intro\"). Use the returned zim and path with the zim_get and \
             zim_get_section tools."
                .into(),
        )
    }
}

impl SyncTool<ZimMcpServer> for ZimSearchTool {
    fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        search_impl(&server.library, &params.query)
    }
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

fn search_impl(library: &ZimLibrary, query: &str) -> Result<SearchResults, ToolError> {
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
        let (text, sections) = hit_text(&article, &terms, *exact, &mut stemmer, is_markdown);
        hits.push(SearchHit {
            zim: arc.name.clone(),
            path: path.clone(),
            title,
            text,
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

/// The `text`/`sections` pair of one search hit, from the article's raw
/// text (`is_markdown` picks the Markdown or the HTML splitter):
///
/// - a hit whose title matched exactly is a title match by definition: its
///   text is the first intro paragraph, no sections;
/// - otherwise, when every query term occurs in the first intro paragraph,
///   same: the lead already covers the query;
/// - otherwise every region is scanned - the intro first (under its
///   `_intro` name, `html::INTRO_SECTION`), then the body sections in
///   document order. The regions holding at least one query match are
///   reported as `sections`, and the text is the best-matching paragraph
///   (most matching word occurrences across all regions; ties keep the
///   earliest, so an intro paragraph beats a body paragraph), truncated
///   to `INTRO_CHARS`;
/// - when no region matches either (only the title in the index matched
///   the query), the text falls back to the first intro paragraph. Regions
///   without paragraphs degrade to empty text, never a panic.
///
/// The intro's paragraphs are extracted first and alone: the two title/lead
/// cases above - the common ones - never need the full article split.
fn hit_text(
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
    let mut best_count = 0usize;
    let mut best_para: Option<&String> = None;
    let mut names: Vec<String> = Vec::new();
    for (name, paras) in &secs {
        let mut matched = false;
        for para in paras {
            let n = para_matches(para, terms, stem);
            if n > 0 {
                matched = true;
                if n > best_count {
                    // Ties keep the earlier paragraph: only a strictly
                    // better count replaces the incumbent.
                    best_count = n;
                    best_para = Some(para);
                }
            }
        }
        if matched {
            names.push(name.clone());
        }
    }
    match best_para {
        Some(p) => (p.chars().take(INTRO_CHARS).collect(), Some(names)),
        None => (lead(), None),
    }
}

// ---------------------------------------------------------------------------
// Tests: end-to-end over a synthetic archive carrying a real Xapian index
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zim::testutil::{build_archive, TestEntry, TestRedirect};
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

    fn test_server() -> (ZimMcpServer, tempfile::TempDir) {
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
        // An exact title match reports the lead paragraph as its text.
        assert_eq!(first.text, "An apple is the fruit of <rosaceae> trees.");
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
        assert_eq!(hits[0].text, "Nitrogen is a colorless, odorless gas.");
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
        // The text is built from the redirect target's content.
        assert_eq!(hits[0].text, "Aeronautics is the science of flight.");
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
        assert_eq!(hits[0].text, "The atmosphere is mostly nitrogen and oxygen.");
        assert_eq!(hits[1].sections, Some(vec!["_intro".to_string()]), "{:?}", hits[1]);
        assert_eq!(hits[1].text, "Nitrogen is a colorless, odorless gas.");
    }

    #[test]
    fn e2e_search_intro_match_reports_lead_without_sections() {
        // Both query terms occur in the lead paragraph ("An apple is the
        // fruit of <rosaceae> trees."): an intro match on a full-text hit
        // (nobody's title is "apple fruit"), so the text is the lead and
        // the serialized JSON carries no "sections" field at all.
        let (server, _keep) = test_server();
        let hits = search(&server, "apple fruit");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Apple");
        assert_eq!(hits[0].text, "An apple is the fruit of <rosaceae> trees.");
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
        assert_eq!(hit.text, "Wild apples grew in Kazakhstan.");
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
        // "himalaya" matches only the intro's second paragraph: the text is
        // that paragraph and the intro is reported as the matching region
        // _intro - not the lead, and not without sections.
        let (server, _keep) = intro_test_server();
        let hits = search(&server, "himalaya");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Salt");
        assert_eq!(hit.sections, Some(vec!["_intro".to_string()]), "{hit:?}");
        assert_eq!(hit.text, "The Himalaya range holds vast deposits of rock salt.");
        let json = serde_json::to_string(hit).unwrap();
        assert!(json.contains(r#""sections":["_intro"]"#), "{json}");
    }

    #[test]
    fn e2e_search_intro_and_section_matches_report_both() {
        // "salt beds" matches the intro's second paragraph ("salt") and the
        // Formation section ("Salt beds ..."): both regions are reported,
        // _intro first, and the text is the paragraph with the most
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
        assert_eq!(hit.text, "Salt beds form when seas evaporate.");
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
        // intro: it is reported as the matching region _intro, and the text
        // is the intro's matching paragraph.
        assert_eq!(hits[0].sections, Some(vec!["_intro".to_string()]));
        assert!(hits[0].text.contains("cherry is the fruit"), "{:?}", hits[0].text);
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

        // Search: the text is plain text derived from the Markdown, free of
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
                .text
                .starts_with("Zinc is a chemical element with the symbol Zn."),
            "{:?}",
            hits[0].text
        );
        assert!(
            !hits[0].text.contains("disambiguation")
                && !hits[0].text.contains("**")
                && !hits[0].text.contains("[[")
                && !hits[0].text.contains('#'),
            "{:?}",
            hits[0].text
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
        assert_eq!(hits[0].text, "Zinc smelting is documented in ancient times.");
        // includes the subsection, reports the heading as written.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "history" }),
        )
        .unwrap();
        let result = ZimGetSectionTool::invoke(&server, params).unwrap();
        assert_eq!(result.section, "History");
        assert!(result.content.contains("ancient times"), "{:?}", result.content);
        assert!(result.content.contains("Ancient India smelted zinc early"));

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

    #[test]
    fn e2e_get() {
        let (server, _keep) = test_server();

        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "test.zim", "path": "C/Apple" }),
        )
        .unwrap();
        let result = ZimGetTool::invoke(&server, params).unwrap();
        assert_eq!(result.title, "Apple");
        assert_eq!(result.path, "C/Apple");
        assert_eq!(result.mime_type.as_deref(), Some("text/html"));
        assert_eq!(result.content_encoding, "utf-8");
        assert!(result.content.contains("10,000 years"));

        // Bare path also works.
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "test.zim", "path": "Banana" }),
        )
        .unwrap();
        let result = ZimGetTool::invoke(&server, params).unwrap();
        assert!(result.content.contains("herbaceous plants"));

        // Unknown article / unknown archive.
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "test.zim", "path": "Nope" }),
        )
        .unwrap();
        assert!(matches!(ZimGetTool::invoke(&server, params), Err(ToolError::NotFound(_))));
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "other.zim", "path": "Apple" }),
        )
        .unwrap();
        assert!(matches!(ZimGetTool::invoke(&server, params), Err(ToolError::NotFound(_))));
    }

    #[test]
    fn e2e_get_guards_internal_and_oversized() {
        let (server, _keep) = test_server();

        // The embedded full-text index is an internal entry, not an article.
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "test.zim", "path": "X/fulltext/xapian" }),
        )
        .unwrap();
        assert!(matches!(
            ZimGetTool::invoke(&server, params),
            Err(ToolError::InvalidArgument(_))
        ));

        // Content larger than the response cap is rejected cleanly.
        let dir = tempfile::tempdir().unwrap();
        let big: &'static [u8] = Box::leak(vec![b'a'; 17 * 1024 * 1024].into_boxed_slice());
        let content = [TestEntry { namespace: b'C', url: "Big", title: "Big", mime: 0, body: big }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, None);
        std::fs::write(dir.path().join("big.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        let server = ZimMcpServer::new(library);
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "big.zim", "path": "C/Big" }),
        )
        .unwrap();
        assert!(matches!(
            ZimGetTool::invoke(&server, params),
            Err(ToolError::InvalidArgument(msg)) if msg.contains("too large")
        ));
    }

    #[test]
    fn e2e_get_section() {
        let (server, _keep) = test_server();

        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "test.zim", "path": "Apple", "section": "History" }),
        )
        .unwrap();
        let result = ZimGetSectionTool::invoke(&server, params).unwrap();
        assert_eq!(result.title, "Apple");
        assert_eq!(result.section, "History");
        assert!(result.content.contains("10,000 years"), "{:?}", result.content);
        // Includes the subsection, stops at the next h2.
        assert!(result.content.contains("Kazakhstan"));
        assert!(!result.content.contains("Computing devices"));

        // Case-insensitive name match; the actual heading text is reported.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "test.zim", "path": "Banana", "section": "growth" }),
        )
        .unwrap();
        let result = ZimGetSectionTool::invoke(&server, params).unwrap();
        assert_eq!(result.section, "Growth");
        assert!(result.content.contains("herbaceous plants"));

        // Missing section.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "test.zim", "path": "Banana", "section": "Nope" }),
        )
        .unwrap();
        assert!(matches!(
            ZimGetSectionTool::invoke(&server, params),
            Err(ToolError::SectionNotFound(_))
        ));
    }
}

// ---------------------------------------------------------------------------
// zim_get
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimGetParams {
    /// ZIM file name, relative to the ZIM directory (as given in search results)
    pub zim: String,
    /// Path of the article/page inside the ZIM file
    pub path: String,
}

#[derive(Serialize, JsonSchema)]
pub struct ZimGetResult {
    /// Page/article title
    pub title: String,
    /// Final path inside the ZIM file (after following redirects)
    pub path: String,
    /// MIME type, if known
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Content encoding: "utf-8" or "base64"
    pub content_encoding: &'static str,
    /// Content of the article/page/object (all of it)
    pub content: String,
}

pub struct ZimGetTool;

/// Entries in the `X` namespace are internal (embedded full-text search
/// indexes), not article content, and can run to hundreds of megabytes.
const MAX_ZIM_GET_BYTES: usize = 16 * 1024 * 1024;

impl ToolBase for ZimGetTool {
    type Parameter = ZimGetParams;
    type Output = ZimGetResult;
    type Error = ToolError;

    fn name() -> Cow<'static, str> {
        "zim_get".into()
    }
    fn description() -> Option<Cow<'static, str>> {
        Some(
            "Get the full content of an article or page from a ZIM file. Arguments: the ZIM file \
             name and the article/page path (as returned by zim_search). Returns the title, \
             final path, MIME type, and the full content (UTF-8 text, or base64 for binary \
             objects)."
                .into(),
        )
    }
}

impl SyncTool<ZimMcpServer> for ZimGetTool {
    fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let arc = find_archive(&server.library, &params.zim)?;
        let article = not_found_if_missing(arc.get_article(&params.path))?;
        if article.full_path.starts_with("X/") {
            return Err(ToolError::InvalidArgument(format!(
                "{} is an internal entry (embedded search index); article content lives under C/ or A/",
                article.full_path
            )));
        }
        if article.bytes.len() > MAX_ZIM_GET_BYTES {
            return Err(ToolError::InvalidArgument(format!(
                "content of {} is {} bytes, too large for one response (max {MAX_ZIM_GET_BYTES})",
                article.full_path,
                article.bytes.len()
            )));
        }
        let (content, encoding) = match std::str::from_utf8(&article.bytes) {
            Ok(text) => (text.to_string(), "utf-8"),
            Err(_) => (
                base64::engine::general_purpose::STANDARD.encode(&article.bytes),
                "base64",
            ),
        };
        Ok(ZimGetResult {
            title: article.title,
            path: article.full_path,
            mime_type: article.mime_type,
            content_encoding: encoding,
            content,
        })
    }
}

// ---------------------------------------------------------------------------
// zim_get_section
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimGetSectionParams {
    /// ZIM file name, relative to the ZIM directory (as given in search results)
    pub zim: String,
    /// Path of the article/page inside the ZIM file
    pub path: String,
    /// Name of the section (heading text) to retrieve
    pub section: String,
}

#[derive(Serialize, JsonSchema)]
pub struct ZimGetSectionResult {
    /// Page/article title
    pub title: String,
    /// The section name (heading) that was found
    pub section: String,
    /// Content of the section (HTML, or Markdown for Markdown articles)
    pub content: String,
}

pub struct ZimGetSectionTool;

impl ToolBase for ZimGetSectionTool {
    type Parameter = ZimGetSectionParams;
    type Output = ZimGetSectionResult;
    type Error = ToolError;

    fn name() -> Cow<'static, str> {
        "zim_get_section".into()
    }
    fn description() -> Option<Cow<'static, str>> {
        Some(
            "Get a single section of an article or page from a ZIM file, identified by its \
             heading text (e.g. \"History\"). Returns the page title, the section name, and \
             the section's content."
                .into(),
        )
    }
}

impl SyncTool<ZimMcpServer> for ZimGetSectionTool {
    fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let arc = find_archive(&server.library, &params.zim)?;
        let article = not_found_if_missing(arc.get_article(&params.path))?;
        let text = std::str::from_utf8(&article.bytes).map_err(|_| {
            ToolError::Internal(format!(
                "article content of {} is not UTF-8 text; section extraction requires text",
                params.path
            ))
        })?;
        // Markdown editions carry plain Markdown, not HTML: pick the matching
        // extractor; anything without a Markdown MIME type takes the HTML path.
        let found = if article.mime_type.as_deref().is_some_and(|m| m.contains("markdown")) {
            markdown::section_content(text, &params.section)
        } else {
            html::section_content(text, &params.section)
        };
        match found {
            Some((section, content)) => Ok(ZimGetSectionResult {
                title: article.title,
                section,
                content,
            }),
            None => Err(ToolError::SectionNotFound(format!(
                "section '{0}' not found in {1} (of {2})",
                params.section, params.path, params.zim
            ))),
        }
    }
}
