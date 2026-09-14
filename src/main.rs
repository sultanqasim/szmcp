mod html;
mod markdown;
mod tools;
mod zim;

use clap::{Parser, Subcommand};
use rmcp::handler::server::router::tool::SyncTool;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tools::{
    ZimGetParams, ZimGetSectionParams, ZimGetSectionTool, ZimGetTool, ZimMcpServer, ZimSearchParams,
    ZimSearchTool,
};

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
        /// Path to a folder containing ZIM files
        zim_dir: PathBuf,

        /// Bind address to listen on (default: 127.0.0.1)
        #[arg(long, default_value_t = String::from("127.0.0.1"))]
        bind: String,

        /// Port number to listen on (default: 3001)
        #[arg(long, short, default_value_t = 3001)]
        port: u16,
    },
    /// Search all articles of all ZIM files in a folder
    Search {
        /// Path to a folder containing ZIM files
        zim_dir: PathBuf,
        /// The search string to look for
        query: String,
    },
    /// Get the full content of an article from a ZIM file
    Get {
        /// Path to a folder containing ZIM files
        zim_dir: PathBuf,
        /// ZIM file name, relative to the ZIM directory (as given in search results)
        zim: String,
        /// Path of the article inside the ZIM file
        path: String,
    },
    /// Get one section of an article from a ZIM file, by its heading text
    GetSection {
        /// Path to a folder containing ZIM files
        zim_dir: PathBuf,
        /// ZIM file name, relative to the ZIM directory (as given in search results)
        zim: String,
        /// Path of the article inside the ZIM file
        path: String,
        /// Name of the section (heading text) to retrieve
        section: String,
    },
}

/// Scan a folder of ZIM files, the way every subcommand starts.
fn open_library(zim_dir: &Path) -> Result<Arc<zim::ZimLibrary>, String> {
    let dir = std::fs::canonicalize(zim_dir)
        .map_err(|e| format!("Invalid ZIM directory {}: {e}", zim_dir.display()))?;
    zim::ZimLibrary::scan(&dir)
        .map(Arc::new)
        .map_err(|e| format!("Failed to scan ZIM directory {}: {e}", zim_dir.display()))
}

/// Run one of the MCP tools against a freshly opened library and print its
/// response to stdout - the same JSON the MCP tool returns, without the MCP
/// wrapper. Nothing else may reach stdout.
fn run_tool<T: SyncTool<ZimMcpServer>>(
    library: Arc<zim::ZimLibrary>,
    params: T::Parameter,
) -> Result<(), String>
where
    T::Error: std::fmt::Display,
{
    match T::invoke(&ZimMcpServer::new(library), params) {
        Ok(out) => {
            println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
            Ok(())
        }
        Err(e) => Err(e.to_string()),
    }
}

async fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Serve { zim_dir, bind, port } => serve(&zim_dir, bind, port).await,
        Command::Search { zim_dir, query } => {
            run_tool::<ZimSearchTool>(open_library(&zim_dir)?, ZimSearchParams { query })
        }
        Command::Get { zim_dir, zim, path } => {
            run_tool::<ZimGetTool>(open_library(&zim_dir)?, ZimGetParams { zim, path })
        }
        Command::GetSection { zim_dir, zim, path, section } => run_tool::<ZimGetSectionTool>(
            open_library(&zim_dir)?,
            ZimGetSectionParams { zim, path, section },
        ),
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
async fn serve(zim_dir: &Path, bind: String, port: u16) -> Result<(), String> {
    let library = open_library(zim_dir)?;

    let addr = format!("{bind}:{port}");

    // 1. Setup MCP Service Factory
    // ZimMcpServer::router() returns a Router<ZimMcpServer>
    let factory_library = library.clone();
    let service_factory = move || Ok(ZimMcpServer::new(factory_library.clone()).router());

    // 2. Setup Session Manager
    let session_manager = Arc::new(LocalSessionManager::default());

    // 3. Setup Streamable HTTP Config with Host validation based on bind address
    use tower_http::cors::{AllowOrigin, CorsLayer};

    let (config, cors_layer) = if bind == "0.0.0.0" || bind == "*" || bind == "::" {
        // External binding: allow any Host and Origin header from remote clients
        let config = StreamableHttpServerConfig::default().disable_allowed_hosts();
        let cors_layer = CorsLayer::permissive();
        (config, cors_layer)
    } else {
        // Localhost binding: restrict to loopback origins
        let config = StreamableHttpServerConfig::default()
            .with_allowed_hosts(["localhost", "127.0.0.1", "::1"]); // block DNS rebinding
        let cors_layer = CorsLayer::permissive()
            .allow_origin(AllowOrigin::predicate(|origin, _headers| {
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

    // 4. Create the Streamable HTTP Service
    let mcp_service = StreamableHttpService::new(service_factory, session_manager, config);

    // 5. Setup Axum router with CORS
    let app = axum::Router::new()
        .fallback_service(mcp_service)
        .layer(cors_layer);

    eprintln!("szmcp - Sultan's ZIM MCP");
    eprintln!("ZIM directory: {}", zim_dir.display());
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
