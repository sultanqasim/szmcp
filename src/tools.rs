//! MCP server definition: server struct, tool registration, and the ZIM
//! tools (`zim_search`, `zim_get`, `zim_get_section`) - thin wrappers over
//! the pipelines in `search` and `get`.
//!
//! The tool set is shaped by the library's launch mode: a single-file
//! library takes no `zim` argument (there is nothing to name), a scanned
//! directory takes one (required on the get tools, an optional filter on
//! search) and gains `zim_list` to report the file names the other tools
//! take. The two shapes are separate tool types, registered per mode by
//! [`ZimMcpServer::router`].
//!
//! The tools are async: each one moves its arguments onto a blocking thread
//! (`tokio::task::spawn_blocking`) and awaits the result. The pipelines do
//! hundreds of milliseconds of synchronous work per call; running them
//! inline on async workers (as rmcp does for sync tools) would pin one
//! runtime worker per in-flight call.

use crate::get::{get_article, get_section};
pub use crate::get::{ZimGetResult, ZimGetSectionResult};
use crate::search::{search, DEFAULT_SEARCH_LIMIT};
pub use crate::search::SearchResults;
use crate::zim::{Mode, ZimLibrary};
use rmcp::handler::server::router::tool::{AsyncTool, ToolBase, ToolRouter};
use rmcp::handler::server::router::Router;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{Implementation, ServerInfo};
use rmcp::ErrorData;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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
                .with_async_tool::<ZimGetSectionDirTool>()
                .with_async_tool::<ZimListTool>(),
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
    /// Maximum number of results to return (default 10)
    pub limit: Option<usize>,
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
            "Search for articles in a ZIM file. Returned results are ranked best first. \
             For full-text search matches where only specific article sections match the \
             query, a \"sections\" field in the result lists the relevant section names. \
             Use the returned article path with the zim_get and zim_get_section tools."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimSearchTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || {
            search(&library, None, &params.query, params.limit.unwrap_or(DEFAULT_SEARCH_LIMIT))
        })
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
    /// ZIM file name to search; omit to search all ZIM files
    pub zim: Option<String>,
    /// Maximum number of results to return (default 10)
    pub limit: Option<usize>,
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
            "Search for articles in ZIM files. Returned results are ranked best first. \
             For full-text search matches where only specific article sections match the \
             query, a \"sections\" field in the result lists the relevant section names. \
             By default every available ZIM file is searched; pass \"zim\" (a name from \
             zim_list) to restrict the search to one file. Use the returned ZIM file name \
             and article path with the zim_get and zim_get_section tools."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimSearchDirTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        // The pipeline validates the filter name (unknown, or one that
        // leaves the ZIM directory) and reports NotFound.
        tokio::task::spawn_blocking(move || {
            search(
                &library,
                params.zim.as_deref(),
                &params.query,
                params.limit.unwrap_or(DEFAULT_SEARCH_LIMIT),
            )
        })
            .await
            .map_err(|e| ToolError::Internal(format!("zim_search task failed: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// zim_get (single mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimGetParams {
    /// Path of the article inside the ZIM file, or the article title
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
            "Get the full content of an article or page from the ZIM file. HTML \
             pages are converted to Markdown (wiki infoboxes rendered as a \
             '## Key facts' section).".into(),
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
            get_article(&library, &arc.name, &params.path, false)
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
    /// ZIM file name
    pub zim: String,
    /// Path of the article inside the ZIM file, or the article title
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
            "Get the full content of an article or page from a ZIM file. HTML \
             pages are converted to Markdown (wiki infoboxes rendered as a \
             '## Key facts' section).".into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimGetDirTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || get_article(&library, &params.zim, &params.path, false))
            .await
            .map_err(|e| ToolError::Internal(format!("zim_get task failed: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// zim_get_section (single mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimGetSectionParams {
    /// Path of the article inside the ZIM file, or the article title
    pub path: String,
    /// Name of the section to retrieve
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
             the article lead. HTML pages are converted to Markdown first, so headings are those of the converted text."
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
            get_section(&library, &arc.name, &params.path, &params.section, false)
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
    /// ZIM file name
    pub zim: String,
    /// Path of the article inside the ZIM file, or the article title
    pub path: String,
    /// Name of the section to retrieve
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
            "Get a single section of an article or page from a ZIM file, identified by \
             its heading text (e.g. \"History\"), or by the special name \"_intro\" for \
             the article lead. HTML pages are converted to Markdown first, so headings are those of the converted text."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimGetSectionDirTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || {
            get_section(&library, &params.zim, &params.path, &params.section, false)
        })
        .await
        .map_err(|e| ToolError::Internal(format!("zim_get_section task failed: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// zim_list (directory mode)
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ZimListParams {}

#[derive(Serialize, JsonSchema)]
pub struct ZimListResult {
    /// ZIM file names
    pub files: Vec<String>,
}

pub struct ZimListTool;

impl ToolBase for ZimListTool {
    type Parameter = ZimListParams;
    type Output = ZimListResult;
    type Error = ToolError;

    fn name() -> Cow<'static, str> {
        "zim_list".into()
    }
    fn description() -> Option<Cow<'static, str>> {
        Some(
            "List the loaded ZIM files. Returns their names relative to the ZIM \
             directory - the \"zim\" argument the other tools take."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimListTool {
    async fn invoke(server: &ZimMcpServer, _params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        // Same shape as the real work tools (the listing itself is a
        // trivial clone of the loaded archive names).
        tokio::task::spawn_blocking(move || ZimListResult {
            files: library.archives.iter().map(|a| a.name.clone()).collect(),
        })
        .await
        .map_err(|e| ToolError::Internal(format!("zim_list task failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::tests::{test_server, test_single_server, two_archive_library};
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
            ZimSearchParams { query: "apple".into(), limit: None },
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
            ZimSearchDirParams { query: "banana cherry".into(), zim: None, limit: None },
        ))
        .unwrap();
        assert_eq!(all.results.len(), 2, "{:?}", all.results);
        let one = block_on(ZimSearchDirTool::invoke(
            &server,
            ZimSearchDirParams { query: "banana cherry".into(), zim: Some("a.zim".into()), limit: None },
        ))
        .unwrap();
        assert_eq!(one.results.len(), 1, "{:?}", one.results);
        assert_eq!(one.results[0].zim, "a.zim");

    }

    #[test]
    fn e2e_zim_list_reports_loaded_names() {
        // The names zim_list reports are the form every other tool takes.
        let (server, _keep) = test_server();
        let listing = block_on(ZimListTool::invoke(&server, ZimListParams {})).unwrap();
        assert_eq!(listing.files, vec!["test.zim".to_string()]);
    }
}
