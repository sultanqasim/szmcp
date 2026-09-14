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
use xapian2::{Enquire, Operator, QueryParser, StemStrategy};

/// Number of results `zim_search` returns in total (across all archives).
const SEARCH_LIMIT: u32 = 20;
/// Maximum characters of intro text reported per search hit.
const INTRO_CHARS: usize = 300;
/// How many raw bytes of an article are read to derive its intro. Modern
/// MediaWiki pages carry kilobytes of template CSS and infobox markup before
/// the lead paragraph, so this needs generous headroom.
const INTRO_READ_BYTES: u64 = 64 * 1024;

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
    /// Page/article intro
    pub intro: String,
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
            "Full-text search through all articles in all ZIM files. Returns an object with a \
             \"results\" array (best matches first); each result has the ZIM file name, the \
             article path inside the ZIM file, the page title, and a short intro. Use the \
             returned ZIM name and path with the zim_get and zim_get_section tools."
                .into(),
        )
    }
}

impl SyncTool<ZimMcpServer> for ZimSearchTool {
    fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        search_impl(&server.library, &params.query)
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
    // match before any archive contributes its second best.
    let mut merged: Vec<(&Arc<Archive>, String, String)> = Vec::new();
    let mut rank = 0usize;
    loop {
        let mut picked = false;
        for (arc, list) in &per_archive {
            if let Some((_, path, title)) = list.get(rank) {
                merged.push((arc, path.clone(), title.clone()));
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
    let mut seen = std::collections::HashSet::new();
    merged.retain(|(_, path, title)| {
        let key = if title.is_empty() { path.as_str() } else { title.as_str() };
        seen.insert(html::normalize(key))
    });
    merged.truncate(SEARCH_LIMIT as usize);

    let mut hits = Vec::with_capacity(merged.len());
    for (arc, path, idx_title) in &merged {
        let (entry_title, mime, bytes) = match arc.article_preview(path, INTRO_READ_BYTES) {
            Ok(Some((entry_title, mime, bytes))) => (entry_title, mime, bytes),
            _ => (String::new(), None, Vec::new()),
        };
        // Prefer the entry's own title; many openZIM archives leave the
        // directory-entry title empty and only carry the title in the
        // index (which we already read as `idx_title`).
        let title = if !entry_title.is_empty() {
            entry_title
        } else if !idx_title.is_empty() {
            idx_title.clone()
        } else {
            path.clone()
        };
        let text = String::from_utf8_lossy(&bytes);
        // Markdown editions carry plain Markdown, not HTML: pick the matching
        // extractor so the intro is clean text, free of markup.
        let intro = if mime.as_deref().is_some_and(|m| m.contains("markdown")) {
            markdown::intro_from_markdown(&text, INTRO_CHARS)
        } else {
            html::intro_from_html(&text, INTRO_CHARS)
        };
        hits.push(SearchHit { zim: arc.name.clone(), path: path.clone(), title, intro });
    }
    Ok(SearchResults { results: hits })
}

// ---------------------------------------------------------------------------
// Tests: end-to-end over a synthetic archive carrying a real Xapian index
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zim::testutil::{build_archive, TestEntry};
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
        assert!(first.intro.contains("apple is the fruit of"), "{:?}", first.intro);

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

        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "nitrogen" }),
        )
        .unwrap();
        let hits = ZimSearchTool::invoke(&server, params).unwrap().results;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Nitrogen", "{hits:?}");
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

        // Search: the intro is plain text derived from the Markdown, free of
        // markup, and starts with the lead paragraph - the leading `# Zinc`
        // title line (a separate field of every hit) and the hatnote are
        // dropped.
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
                .intro
                .starts_with("Zinc is a chemical element with the symbol Zn."),
            "{:?}",
            hits[0].intro
        );
        assert!(
            !hits[0].intro.contains("disambiguation")
                && !hits[0].intro.contains("**")
                && !hits[0].intro.contains("[[")
                && !hits[0].intro.contains('#'),
            "{:?}",
            hits[0].intro
        );

        // Section extraction on the Markdown article: case-insensitive,
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
