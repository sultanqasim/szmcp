mod cleanup;
mod entities;
mod get;
mod html;
mod html2md;
mod htmldom;
mod infobox_html;
mod markdown;
mod search;
mod tables;
mod tools;
mod util;
mod wikil10n;
mod zim;

use clap::{Parser, Subcommand};
use get::{get_article, get_section};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
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
    // Log to stderr in every mode: stdout carries only JSON (the CLI
    // subcommands' tool responses).
    tracing_subscriber::fmt().with_writer(std::io::stderr).init();

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
    let service_factory = move || Ok(ZimMcpServer::new(factory_library.clone()).router());

    let session_manager = Arc::new(LocalSessionManager::default());

    use tower_http::cors::{AllowOrigin, CorsLayer};

    let (config, cors_layer) = if bind == "0.0.0.0" || bind == "*" || bind == "::" {
        // External binding: accept any Host/Origin.
        let config = StreamableHttpServerConfig::default().disable_allowed_hosts();
        let cors_layer = CorsLayer::permissive();
        (config, cors_layer)
    } else {
        // Loopback binding: restrict Hosts (DNS rebinding) and Origins.
        let config = StreamableHttpServerConfig::default()
            .with_allowed_hosts(["localhost", "127.0.0.1", "::1"]); // block DNS rebinding
        let cors_layer = CorsLayer::permissive()
            .allow_origin(AllowOrigin::predicate(|origin, _parts| {
                origin.to_str().ok().map_or(false, |s| {
                    s.starts_with("http://localhost:")
                        || s == "http://localhost"
                        || s.starts_with("http://127.0.0.1:")
                        || s == "http://127.0.0.1"
                    || s.starts_with("http://[::1]:")
                        || s == "http://[::1]"
                        || s == "null"
                })
            }));
        (config, cors_layer)
    };

    let mcp_service = StreamableHttpService::new(service_factory, session_manager, config);

    let app = axum::Router::new()
        .fallback_service(mcp_service)
        .layer(cors_layer);

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
