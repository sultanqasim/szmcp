//! MCP server definition: server struct, tool registration, and the three
//! ZIM tools (`zim_search`, `zim_get`, `zim_get_section`) - thin wrappers
//! over the pipelines in `search` and `get`.
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
use crate::zim::ZimLibrary;
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
        let tool_router = ToolRouter::new()
            .with_async_tool::<ZimSearchTool>()
            .with_async_tool::<ZimGetTool>()
            .with_async_tool::<ZimGetSectionTool>();

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
// zim_search
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
            "Search all articles in all ZIM files. Results are ranked best first: an exact \
             title match comes first, followed by articles whose titles contain the query \
             words, then full-text matches; each result has the ZIM file name, the article \
             path, the page title, and preview - the article's first paragraph when the query \
             matches the title or that paragraph, otherwise the sentence that best matches \
             the query together with \"sections\", the matching regions' names (the intro \
             listed as \"_intro\"). Use the returned zim and path with the zim_get and \
             zim_get_section tools."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimSearchTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || search(&library, &params.query))
            .await
            .map_err(|e| ToolError::Internal(format!("zim_search task failed: {e}")))?
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
            "Get the full content of an article or page from a ZIM file. Arguments: the ZIM file \
             name and the article/page path (as returned by zim_search). Returns the title, \
             final path, MIME type, and the full content (UTF-8 text, or base64 for binary \
             objects)."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimGetTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || get_article(&library, &params.zim, &params.path))
            .await
            .map_err(|e| ToolError::Internal(format!("zim_get task failed: {e}")))?
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
            "Get a single section of an article or page from a ZIM file, identified by its \
             heading text (e.g. \"History\"), or by the special name \"_intro\" for the \
             introduction (the content before the first heading). Returns the page title, \
             the section name, and the section's content."
                .into(),
        )
    }
}

impl AsyncTool<ZimMcpServer> for ZimGetSectionTool {
    async fn invoke(server: &ZimMcpServer, params: Self::Parameter) -> Result<Self::Output, Self::Error> {
        let library = server.library.clone();
        tokio::task::spawn_blocking(move || {
            get_section(&library, &params.zim, &params.path, &params.section)
        })
        .await
        .map_err(|e| ToolError::Internal(format!("zim_get_section task failed: {e}")))?
    }
}
