//! MCP server definition: server struct, tool registration, and the ZIM
//! tools (`zim_search`, `zim_get`, `zim_get_section`) - thin wrappers over
//! the pipelines in `search` and `get`.
//!
//! The tool set is shaped by the library's launch mode: a single-file
//! library takes no `zim` argument (there is nothing to name), a scanned
//! directory takes one (required on the get tools, an optional filter on
//! search). The two shapes are separate tool types, registered per mode by
//! [`ZimMcpServer::router`].
//!
//! The tools are async: each one moves its arguments onto a blocking thread
//! (`tokio::task::spawn_blocking`) and awaits the result. The pipelines do
//! hundreds of milliseconds of synchronous work per call; running them
//! inline on async workers (as rmcp does for sync tools) would pin one
//! runtime worker per in-flight call.

use crate::get::{get_article, get_section};
pub use crate::get::{ZimGetResult, ZimGetSectionResult};
use crate::search::search;
pub use crate::search::SearchResults;
use crate::zim::{Mode, ZimLibrary};
use rmcp::handler::server::router::tool::{AsyncTool, ToolBase, ToolRouter};
use rmcp::handler::server::router::Router;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{Implementation, ServerInfo};
use rmcp::ErrorData;
use schemars::JsonSchema;
use serde::Deserialize;
use std::borrow::Cow;
use std::sync::Arc;
use thiserror::Error;

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
        // The launch shape decides the tool set, not the directory
        // contents: one ZIM file means the tools need no `zim` argument.
        let tool_router = match self.library.mode {
            Mode::Single => ToolRouter::new()
                .with_async_tool::<ZimSearchTool>()
                .with_async_tool::<ZimGetTool>()
                .with_async_tool::<ZimGetSectionTool>(),
            Mode::Directory => ToolRouter::new()
                .with_async_tool::<ZimSearchDirTool>()
                .with_async_tool::<ZimGetDirTool>()
                .with_async_tool::<ZimGetSectionDirTool>(),
        };

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

// ---------------------------------------------------------------------------
// zim_search (single mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimSearchParams {
    /// The search string to look for in all articles of all ZIM files
    pub query: String,
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
            "Search all articles in all ZIM files. Results are ranked best first in three \
             tiers: an exact title/URL match comes first (a matching redirect reports the \
             article it points to), then articles whose title contains every query word, \
             then full-text matches ranked by BM25 relevance over all query words (partial \
             matches still return). Each result has the ZIM file name, the article path, \
             the page title, and a preview - the article's first intro sentence for title \
             matches, otherwise the sentence that best matches the query together with \
             \"sections\", the matching regions' names (the intro listed as \"_intro\"). \
             The same article is reported once even when several spellings of it match. \
             Use the returned zim and path with the zim_get and zim_get_section tools."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimSearchTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || search(&library, None, &params.query))
            .await
            .map_err(|e| ToolError::Internal(format!("zim_search task failed: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// zim_search (directory mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimSearchDirParams {
    /// The search string to look for in all articles of all ZIM files
    pub query: String,
    /// ZIM file to search, relative to the ZIM directory (as zim_list
    /// reports); omit to search all ZIM files
    pub zim: Option<String>,
}

pub struct ZimSearchDirTool;

impl ToolBase for ZimSearchDirTool {
    type Parameter = ZimSearchDirParams;
    type Output = SearchResults;
    type Error = ToolError;

    fn name() -> Cow<'static, str> {
        "zim_search".into()
    }
    fn description() -> Option<Cow<'static, str>> {
        Some(
            "Search all articles in all ZIM files. By default every ZIM file is searched; \
             pass \"zim\" (a name from zim_list) to restrict the search to one file. \
             Results are ranked best first in three tiers: an exact title/URL match comes \
             first (a matching redirect reports the article it points to), then articles \
             whose title contains every query word, then full-text matches ranked by BM25 \
             relevance over all query words (partial matches still return). Each result \
             has the ZIM file name, the article path, the page title, and a preview - the \
             article's first intro sentence for title matches, otherwise the sentence \
             that best matches the query together with \"sections\", the matching \
             regions' names (the intro listed as \"_intro\"). The same article is \
             reported once even when several spellings of it match. Use the returned zim \
             and path with the zim_get and zim_get_section tools."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimSearchDirTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        // The pipeline validates the filter name (unknown, or one that
        // leaves the ZIM directory) and reports NotFound.
        tokio::task::spawn_blocking(move || search(&library, params.zim.as_deref(), &params.query))
            .await
            .map_err(|e| ToolError::Internal(format!("zim_search task failed: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// zim_get (single mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimGetParams {
    /// Path of the article inside the ZIM file, or an article title such as "Beaconsfield, Quebec"
    pub path: String,
}

pub struct ZimGetTool;

impl ToolBase for ZimGetTool {
    type Parameter = ZimGetParams;
    type Output = ZimGetResult;
    type Error = ToolError;

    fn name() -> Cow<'static, str> {
        "zim_get".into()
    }
    fn description() -> Option<Cow<'static, str>> {
        Some(
            "Get the full content of an article or page from the ZIM file. Arguments: the \
             article/page path (as returned by zim_search) or an article title such as \
             \"Beaconsfield, Quebec\". Returns the title, final path, MIME type, and the \
             full content (UTF-8 text, or base64 for binary objects)."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimGetTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || {
            // Single mode holds exactly one archive (single() opens one
            // file or fails), so this arm cannot miss.
            let arc = library
                .single_archive()
                .ok_or_else(|| ToolError::Internal("no ZIM archive is loaded".into()))?;
            get_article(&library, &arc.name, &params.path)
        })
        .await
        .map_err(|e| ToolError::Internal(format!("zim_get task failed: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// zim_get (directory mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimGetDirParams {
    /// ZIM file name, relative to the ZIM directory (as given in search results)
    pub zim: String,
    /// Path of the article inside the ZIM file, or an article title such as "Beaconsfield, Quebec"
    pub path: String,
}

pub struct ZimGetDirTool;

impl ToolBase for ZimGetDirTool {
    type Parameter = ZimGetDirParams;
    type Output = ZimGetResult;
    type Error = ToolError;

    fn name() -> Cow<'static, str> {
        "zim_get".into()
    }
    fn description() -> Option<Cow<'static, str>> {
        Some(
            "Get the full content of an article or page from a ZIM file. Arguments: the \
             ZIM file name relative to the ZIM directory (as zim_list reports), and the \
             article/page path (as returned by zim_search) or an article title such as \
             \"Beaconsfield, Quebec\". Returns the title, final path, MIME type, and the \
             full content (UTF-8 text, or base64 for binary objects)."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimGetDirTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || get_article(&library, &params.zim, &params.path))
            .await
            .map_err(|e| ToolError::Internal(format!("zim_get task failed: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// zim_get_section (single mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimGetSectionParams {
    /// Path of the article inside the ZIM file, or an article title such as "Beaconsfield, Quebec"
    pub path: String,
    /// Name of the section to retrieve: heading text, or `_intro` for the
    /// introduction
    pub section: String,
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
            "Get a single section of an article or page from the ZIM file, identified by \
             its heading text (e.g. \"History\"), or by the special name \"_intro\" for \
             the introduction (the content before the first heading). The article is \
             given by its path (as returned by zim_search) or its title (e.g. \
             \"Beaconsfield, Quebec\"). Returns the page title, the section name, and the \
             section's content."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimGetSectionTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || {
            // Single mode holds exactly one archive (single() opens one
            // file or fails), so this arm cannot miss.
            let arc = library
                .single_archive()
                .ok_or_else(|| ToolError::Internal("no ZIM archive is loaded".into()))?;
            get_section(&library, &arc.name, &params.path, &params.section)
        })
        .await
        .map_err(|e| ToolError::Internal(format!("zim_get_section task failed: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// zim_get_section (directory mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimGetSectionDirParams {
    /// ZIM file name, relative to the ZIM directory (as given in search results)
    pub zim: String,
    /// Path of the article inside the ZIM file, or an article title such as "Beaconsfield, Quebec"
    pub path: String,
    /// Name of the section to retrieve: heading text, or `_intro` for the
    /// introduction
    pub section: String,
}

pub struct ZimGetSectionDirTool;

impl ToolBase for ZimGetSectionDirTool {
    type Parameter = ZimGetSectionDirParams;
    type Output = ZimGetSectionResult;
    type Error = ToolError;

    fn name() -> Cow<'static, str> {
        "zim_get_section".into()
    }
    fn description() -> Option<Cow<'static, str>> {
        Some(
            "Get a single section of an article or page from a ZIM file, identified by its \
             heading text (e.g. \"History\"), or by the special name \"_intro\" for the \
             introduction (the content before the first heading). The article is given \
             by its path (as returned by zim_search) or its title (e.g. \
             \"Beaconsfield, Quebec\"), in the ZIM file named by \"zim\" (as zim_list \
             reports). Returns the page title, the section name, and the section's \
             content."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimGetSectionDirTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || {
            get_section(&library, &params.zim, &params.path, &params.section)
        })
        .await
        .map_err(|e| ToolError::Internal(format!("zim_get_section task failed: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::tests::{test_single_server, two_archive_library};
    use rmcp::handler::server::router::tool::AsyncTool;
    use std::future::Future;

    /// Await an async tool invocation (each hops to a blocking thread) from
    /// a sync `#[test]` on a tiny current-thread runtime.
    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn e2e_single_mode_tools_take_no_zim_argument() {
        let (server, _keep) = test_single_server();

        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "path": "C/Apple" }),
        )
        .unwrap();
        let result = block_on(ZimGetTool::invoke(&server, params)).unwrap();
        assert_eq!(result.title, "Apple");
        assert_eq!(result.path, "C/Apple");
        assert!(result.content.contains("10,000 years"));

        // A bare title path works too.
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "path": "Banana" }),
        )
        .unwrap();
        let result = block_on(ZimGetTool::invoke(&server, params)).unwrap();
        assert!(result.content.contains("herbaceous plants"));

        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "path": "Apple", "section": "History" }),
        )
        .unwrap();
        let result = block_on(ZimGetSectionTool::invoke(&server, params)).unwrap();
        assert_eq!(result.section, "History");
        assert!(result.content.contains("10,000 years"), "{:?}", result.content);

        // Search runs without any zim argument, yet every hit still names
        // the file - SearchHit is mode-independent.
        let results = block_on(ZimSearchTool::invoke(
            &server,
            ZimSearchParams { query: "apple".into() },
        ))
        .unwrap();
        assert!(!results.results.is_empty());
        assert_eq!(results.results[0].zim, "test.zim");
        assert_eq!(results.results[0].path, "C/Apple");
    }

    #[test]
    fn e2e_directory_mode_tools_take_and_filter_zim() {
        let (_dir, library) = two_archive_library();
        let server = ZimMcpServer::new(library);

        let params = serde_json::from_value::<ZimGetDirParams>(
            serde_json::json!({ "zim": "a.zim", "path": "C/Banana" }),
        )
        .unwrap();
        let result = block_on(ZimGetDirTool::invoke(&server, params)).unwrap();
        assert_eq!(result.title, "Banana");

        // Unfiltered, the search covers all archives; with the filter, only
        // the named file is searched.
        let all = block_on(ZimSearchDirTool::invoke(
            &server,
            ZimSearchDirParams { query: "banana cherry".into(), zim: None },
        ))
        .unwrap();
        assert_eq!(all.results.len(), 2, "{:?}", all.results);
        let one = block_on(ZimSearchDirTool::invoke(
            &server,
            ZimSearchDirParams { query: "banana cherry".into(), zim: Some("a.zim".into()) },
        ))
        .unwrap();
        assert_eq!(one.results.len(), 1, "{:?}", one.results);
        assert_eq!(one.results[0].zim, "a.zim");

        // Unknown and traversal names are refused.
        for zim in ["nope.zim", "../x.zim"] {
            let params = ZimSearchDirParams { query: "apple".into(), zim: Some(zim.into()) };
            assert!(
                matches!(
                    block_on(ZimSearchDirTool::invoke(&server, params)),
                    Err(ToolError::NotFound(_))
                ),
                "{zim}"
            );
        }
    }
}
