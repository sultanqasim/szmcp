mod html;
mod markdown;
mod tools;
mod zim;

use clap::Parser;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use std::sync::Arc;
use tools::ZimMcpServer;

#[derive(Parser)]
#[command(name = "szmcp")]
#[command(author, version, about = "Sultan's ZIM MCP - MCP server serving content from ZIM files", long_about = None)]
struct Args {
    /// Path to a folder containing ZIM files
    zim_dir: std::path::PathBuf,

    /// Bind address to listen on (default: 127.0.0.1)
    #[arg(long, default_value_t = String::from("127.0.0.1"))]
    bind: String,

    /// Port number to listen on (default: 3001)
    #[arg(long, short, default_value_t = 3001)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    let zim_dir = std::fs::canonicalize(&args.zim_dir)
        .map_err(|e| format!("Invalid ZIM directory {}: {e}", args.zim_dir.display()))?;

    let library = Arc::new(
        zim::ZimLibrary::scan(&zim_dir)
            .map_err(|e| format!("Failed to scan ZIM directory {}: {e}", args.zim_dir.display()))?,
    );

    let addr = format!("{}:{}", args.bind, args.port);

    // 1. Setup MCP Service Factory
    // ZimMcpServer::router() returns a Router<ZimMcpServer>
    let factory_library = library.clone();
    let service_factory = move || Ok(ZimMcpServer::new(factory_library.clone()).router());

    // 2. Setup Session Manager
    let session_manager = Arc::new(LocalSessionManager::default());

    // 3. Setup Streamable HTTP Config with Host validation based on bind address
    use tower_http::cors::{AllowOrigin, CorsLayer};

    let (config, cors_layer) = if args.bind == "0.0.0.0" || args.bind == "*" || args.bind == "::" {
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

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
