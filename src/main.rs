mod cleanup;
mod convert;
mod entities;
mod get;
mod html;
mod html2md;
mod htmldom;
mod infobox_html;
mod lang_map;
mod markdown;
mod search;
mod tables;
mod tools;
mod util;
mod wikil10n;
mod zim;
mod zimcommon;
mod zimwrite;

use clap::{Parser, Subcommand};
use get::{get_article, get_section};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use search::{search, DEFAULT_SEARCH_LIMIT};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tools::ZimMcpServer;

#[derive(Parser)]
#[command(name = "szmcp")]
#[command(author, version, about = "Sultan's ZIM MCP - MCP server and CLI for content from ZIM files", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[command(rename_all = "snake_case")]
enum Command {
    /// Serve the ZIM tools over the MCP streamable HTTP transport
    Serve {
        /// Path to a ZIM file or a folder containing ZIM files
        zim_path: PathBuf,

        /// Bind address to listen on (default: 127.0.0.1)
        #[arg(long, default_value_t = String::from("127.0.0.1"))]
        bind: String,

        /// Port number to listen on (default: 3001)
        #[arg(long, short, default_value_t = 3001)]
        port: u16,
    },
    /// Search all articles of the ZIM files
    Search {
        /// Path to a ZIM file or a folder containing ZIM files
        zim_path: PathBuf,
        /// The search string to look for
        query: String,
        /// Maximum number of results to return (default 10)
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Get the full content of an article from a ZIM file
    Get {
        /// Path to the ZIM file
        zim_path: PathBuf,
        /// Path of the article inside the ZIM file
        path: String,
        /// Print the article content to stdout instead of the JSON tool
        /// response
        #[arg(long)]
        content: bool,
        /// Skip the HTML->Markdown conversion and output the raw HTML
        /// instead (only affects text/html pages)
        #[arg(long)]
        raw: bool,
    },
    /// Convert an HTML ZIM into a ZIM of Markdown articles with fresh
    /// search indexes (a port of wikizim_parser's zim2zim.py)
    Convert {
        /// Path to the source ZIM file (HTML articles)
        input_zim: PathBuf,
        /// Path of the ZIM file to write
        output_zim: PathBuf,
        /// Process only the first N entries of the source, counting every
        /// entry alike (articles, redirects, media, metadata; -1 = all).
        /// Development/testing aid: articles beyond the cutoff are not
        /// converted, so the output may contain dangling redirects
        #[arg(long, default_value_t = -1)]
        limit: i64,
        /// Full-text-index only each article's intro (lead before the first
        /// "## " heading, hatnotes removed, title kept) instead of the whole
        /// Markdown; shrinks the index, keeps lead-level search
        #[arg(long = "index-intro-only")]
        index_intro_only: bool,
        /// Include redirect titles in the title index (default: exclude
        /// them, zim2zim's --no-redirect-titles behavior; redirects still
        /// resolve)
        #[arg(long = "index-redirect-titles")]
        index_redirect_titles: bool,
    },
    /// Get one section of an article from a ZIM file, by its heading text
    GetSection {
        /// Path to the ZIM file
        zim_path: PathBuf,
        /// Path of the article inside the ZIM file
        path: String,
        /// Name of the section (heading text) to retrieve
        section: String,
        /// Print the section content to stdout instead of the JSON tool
        /// response
        #[arg(long)]
        content: bool,
        /// Skip the HTML->Markdown conversion and extract the section
        /// from the raw HTML instead (only affects text/html pages)
        #[arg(long)]
        raw: bool,
    },
}

/// Open the ZIM library the way every subcommand starts: `zim_path` is either
/// a ZIM file itself or a folder whose ZIM files (recursed) are all loaded.
fn open_library(zim_path: &Path) -> Result<Arc<zim::ZimLibrary>, String> {
    let path = std::fs::canonicalize(zim_path)
        .map_err(|e| format!("Invalid ZIM path {}: {e}", zim_path.display()))?;
    let library = if path.is_dir() {
        zim::ZimLibrary::scan(&path)
            .map_err(|e| format!("Failed to scan ZIM directory {}: {e}", zim_path.display()))
    } else if path.is_file() {
        zim::ZimLibrary::single(&path)
            .map_err(|e| format!("Failed to open ZIM file {}: {e}", zim_path.display()))
    } else {
        Err(format!("{}: not a directory or ZIM file", path.display()))
    };
    library.map(Arc::new)
}

/// Open the one ZIM file at `zim_path` for the `get`/`get_section`
/// subcommands: there the archive is identified by its own path, so the
/// argument must be a ZIM file - a directory would name several archives
/// while the tool call looks up one article in one archive.
fn open_single_zim(zim_path: &Path) -> Result<Arc<zim::ZimLibrary>, String> {
    let path = std::fs::canonicalize(zim_path)
        .map_err(|e| format!("Invalid ZIM path {}: {e}", zim_path.display()))?;
    if path.is_dir() {
        return Err(format!("expected a ZIM file, not a directory: {}", path.display()));
    }
    if !path.is_file() {
        return Err(format!("{}: not a ZIM file", path.display()));
    }
    zim::ZimLibrary::single(&path)
        .map_err(|e| format!("Failed to open ZIM file {}: {e}", zim_path.display()))
        .map(Arc::new)
}

/// The name of the one archive a single-file library holds - the `zim`
/// argument the tool pipelines expect.
fn single_archive_name(library: &zim::ZimLibrary) -> Result<String, String> {
    library
        .archives
        .first()
        .map(|a| a.name.clone())
        .ok_or_else(|| "no ZIM archive was loaded".to_string())
}

/// Print one tool response to stdout as pretty JSON - the same JSON the MCP
/// tool returns, without the MCP wrapper. Nothing else may reach stdout.
fn print_result<T: serde::Serialize>(result: &T) -> Result<(), String> {
    println!("{}", serde_json::to_string_pretty(result).map_err(|e| e.to_string())?);
    Ok(())
}

async fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Serve { zim_path, bind, port } => serve(&zim_path, bind, port).await,
        Command::Search { zim_path, query, limit } => {
            let library = open_library(&zim_path)?;
            let limit = limit.unwrap_or(DEFAULT_SEARCH_LIMIT);
            let results = search(&library, None, &query, limit).map_err(|e| e.to_string())?;
            print_result(&results)
        }
        Command::Get { zim_path, path, content, raw } => {
            let library = open_single_zim(&zim_path)?;
            let zim = single_archive_name(&library)?;
            let result = get_article(&library, &zim, &path, raw).map_err(|e| e.to_string())?;
            if content {
                println!("{}", result.content);
                Ok(())
            } else {
                print_result(&result)
            }
        }
        Command::Convert { input_zim, output_zim, limit, index_intro_only, index_redirect_titles } => {
            // Synchronous, blocking: no await point is involved (the tokio
            // runtime simply hosts this call).
            convert::convert(&input_zim, &output_zim, limit, index_intro_only, index_redirect_titles)
        }
        Command::GetSection { zim_path, path, section, content, raw } => {
            let library = open_single_zim(&zim_path)?;
            let zim = single_archive_name(&library)?;
            let result =
                get_section(&library, &zim, &path, &section, raw).map_err(|e| e.to_string())?;
            if content {
                println!("{}", result.content);
                Ok(())
            } else {
                print_result(&result)
            }
        }
    }
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    match run(Cli::parse().command).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run the MCP server (streamable HTTP transport, endpoint at the root path).
async fn serve(zim_path: &Path, bind: String, port: u16) -> Result<(), String> {
    let library = open_library(zim_path)?;

    let addr = format!("{bind}:{port}");

    let factory_library = library.clone();
    let service_factory = move || Ok(ZimMcpServer::new(factory_library.clone()));

    let session_manager = Arc::new(LocalSessionManager::default());


    let any_origin = bind == "0.0.0.0" || bind == "*" || bind == "::";
    let config = if any_origin {
        // External binding: accept any Host/Origin.
        StreamableHttpServerConfig::default().disable_allowed_hosts()
    } else {
        // Loopback binding: restrict Hosts (DNS rebinding).
        StreamableHttpServerConfig::default().with_allowed_hosts(["localhost", "127.0.0.1", "::1"])
    };

    let mcp_service = StreamableHttpService::new(service_factory, session_manager, config);

    let app = axum::Router::new()
        .fallback_service(mcp_service)
        .layer(axum::middleware::from_fn_with_state(
            any_origin,
            cors_middleware,
        ));

    eprintln!("szmcp - Sultan's ZIM MCP");
    eprintln!("ZIM path: {}", zim_path.display());
    for a in &library.archives {
        eprintln!(
            "  loaded: {} (entries: {}, searchable: {})",
            a.name,
            a.article_count(),
            a.searchable()
        );
    }
    eprintln!("Listening on: http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("Failed to bind {addr}: {e}"))?;
    axum::serve(listener, app).await.map_err(|e| e.to_string())?;

    Ok(())
}

/// CORS middleware (the tower-http CorsLayer behavior the server relied
/// on): every OPTIONS request is answered here as a preflight (200, empty,
/// with the method/header allowlists), and every other response carries the
/// expose list. With `any_origin` (external binds) all origins are allowed
/// with the literal `*`; otherwise only loopback origins (localhost /
/// 127.0.0.1 / [::1], optional :port, or `null`) pass and are mirrored,
/// with `Vary: Origin` so caches keep origins apart.
async fn cors_middleware(State(any_origin): State<bool>, req: Request, next: Next) -> Response {
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    if req.method() == Method::OPTIONS {
        let mut res = Response::builder()
            .status(StatusCode::OK)
            .body(Body::empty())
            .unwrap();
        put_cors_headers(&mut res, any_origin, origin.as_deref(), true);
        if !any_origin {
            res.headers_mut()
                .insert(header::VARY, HeaderValue::from_static("Origin"));
        }
        return res;
    }
    let mut res = next.run(req).await;
    put_cors_headers(&mut res, any_origin, origin.as_deref(), false);
    if !any_origin {
        res.headers_mut()
            .insert(header::VARY, HeaderValue::from_static("Origin"));
    }
    res
}

/// The `Access-Control-*` headers of one response: a preflight gets the
/// method/header allowlists, an actual response the expose list; the
/// allowed origin (always `*` for external binds) goes on both.
fn put_cors_headers(res: &mut Response, any_origin: bool, origin: Option<&str>, preflight: bool) {
    let h = res.headers_mut();
    let acao = if any_origin {
        Some(HeaderValue::from_static("*"))
    } else {
        origin
            .filter(|o| loopback_origin(o))
            .and_then(|o| HeaderValue::from_str(o).ok())
    };
    if let Some(acao) = acao {
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, acao);
    }
    if preflight {
        h.insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("*"),
        );
        h.insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("*"),
        );
    } else {
        h.insert(
            header::ACCESS_CONTROL_EXPOSE_HEADERS,
            HeaderValue::from_static("*"),
        );
    }
}

/// The loopback-origin predicate of the loopback bind mode.
/// The loopback-origin predicate of the loopback bind mode.
fn loopback_origin(origin: &str) -> bool {
    origin.starts_with("http://localhost:")
        || origin == "http://localhost"
        || origin.starts_with("http://127.0.0.1:")
        || origin == "http://127.0.0.1"
        || origin.starts_with("http://[::1]:")
        || origin == "http://[::1]"
        || origin == "null"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_single_zim_rejects_directories() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = std::fs::canonicalize(dir.path()).unwrap();
        let Err(err) = open_single_zim(dir.path()) else {
            panic!("a directory must be rejected");
        };
        assert!(err.contains("expected a ZIM file, not a directory"), "{err}");
        assert!(err.contains(&canonical.display().to_string()), "{err}");
    }

}
