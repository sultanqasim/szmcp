//! The article and section fetch pipeline behind the `zim_get` and
//! `zim_get_section` tools: archive lookup, redirect following, and the
//! internal-entry and oversize checks, then section extraction dispatch.

use crate::html;
use crate::markdown;
use crate::tools::ToolError;
use crate::zim::{Archive, ZimLibrary};
use base64::Engine as _;
use schemars::JsonSchema;
use serde::Serialize;
use std::sync::Arc;

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

/// Entries in the `X` namespace are internal (embedded full-text search
/// indexes), not article content, and can run to hundreds of megabytes.
const MAX_ZIM_GET_BYTES: usize = 16 * 1024 * 1024;

/// Normalize the `path` argument of the article lookups: a namespaced ZIM
/// path (`C/Salt`, `A/Foo`, `-/x`) is returned unchanged, anything else is
/// treated as a Wikipedia article title and mapped to the path layout
/// Wikipedia ZIMs use: `"C/" + title with spaces replaced by underscores`.
///
/// A path is recognized by its shape: the first byte an ASCII letter or `-`
/// and the second byte `/`. That keeps every explicit path working - legacy
/// namespaces, the `-` namespace, and non-Wikipedia ZIMs' opaque URL paths -
/// while bare titles (which never contain `/`) convert. The conversion
/// cannot know an archive's exact path spelling, so a title must match the
/// ZIM's path casing; if the lookup fails, the error reports the converted
/// path and the caller can retry with the exact path from search results.
fn article_path(raw: &str) -> String {
    let bytes = raw.as_bytes();
    if bytes.len() >= 2
        && bytes[1] == b'/'
        && (bytes[0].is_ascii_alphabetic() || bytes[0] == b'-')
    {
        return raw.to_string();
    }
    format!("C/{}", raw.replace(" ", "_"))
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

/// The full content of one article or page from the `zim` archive: `path` is
/// resolved inside it (redirects followed); internal entries and content over
/// the response cap are refused.
pub fn get_article(
    library: &ZimLibrary,
    zim: &str,
    path: &str,
) -> Result<ZimGetResult, ToolError> {
    let arc = find_archive(library, zim)?;
    // Accept a bare article title as well as a namespaced path.
    let path = article_path(path);
    let article = not_found_if_missing(arc.get_article(&path))?;
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

#[derive(Serialize, JsonSchema)]
pub struct ZimGetSectionResult {
    /// Page/article title
    pub title: String,
    /// The section name (heading) that was found
    pub section: String,
    /// Content of the section (HTML, or Markdown for Markdown articles)
    pub content: String,
}

/// One section of an article from the `zim` archive, by heading text or by the
/// reserved `_intro` name, extracted from its HTML or Markdown.
pub fn get_section(
    library: &ZimLibrary,
    zim: &str,
    path: &str,
    section: &str,
) -> Result<ZimGetSectionResult, ToolError> {
    let arc = find_archive(library, zim)?;
    // Accept a bare article title as well as a namespaced path; converting
    // first also makes the not-found / section errors below report the
    // converted path.
    let path = article_path(path);
    let article = not_found_if_missing(arc.get_article(&path))?;
    let text = std::str::from_utf8(&article.bytes).map_err(|_| {
        ToolError::Internal(format!(
            "article content of {} is not UTF-8 text; section extraction requires text",
            path
        ))
    })?;
    // Markdown editions carry plain Markdown, not HTML: pick the matching
    // extractor; anything without a Markdown MIME type takes the HTML path.
    let found = if article.mime_type.as_deref().is_some_and(|m| m.contains("markdown")) {
        markdown::section_content(text, section)
    } else {
        html::section_content(text, section)
    };
    match found {
        Some((section, content)) => Ok(ZimGetSectionResult {
            title: article.title,
            section,
            content,
        }),
        None => Err(ToolError::SectionNotFound(format!(
            "section '{0}' not found in {1} (of {2})",
            section, path, zim
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests: end-to-end over a synthetic archive carrying a real Xapian index
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::tests::test_server;
    use crate::tools::{
        ZimGetParams, ZimGetSectionParams, ZimGetSectionTool, ZimGetTool, ZimMcpServer,
    };
    use crate::zim::testutil::{build_archive, TestEntry};
    use rmcp::handler::server::router::tool::AsyncTool;
    use std::future::Future;

    /// Run an async tool invocation to completion on this thread: the tools
    /// are async (each hops to a blocking thread), and these tests are sync
    /// `#[test]`s, so each invocation gets its own tiny current-thread
    /// runtime to await in.
    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn e2e_get() {
        let (server, _keep) = test_server();

        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "test.zim", "path": "C/Apple" }),
        )
        .unwrap();
        let result = block_on(ZimGetTool::invoke(&server, params)).unwrap();
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
        let result = block_on(ZimGetTool::invoke(&server, params)).unwrap();
        assert!(result.content.contains("herbaceous plants"));

        // Unknown article / unknown archive.
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "test.zim", "path": "Nope" }),
        )
        .unwrap();
        assert!(matches!(block_on(ZimGetTool::invoke(&server, params)), Err(ToolError::NotFound(_))));
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "other.zim", "path": "Apple" }),
        )
        .unwrap();
        assert!(matches!(block_on(ZimGetTool::invoke(&server, params)), Err(ToolError::NotFound(_))));
    }

    #[test]
    fn article_path_maps_titles_and_keeps_paths() {
        // Titles convert to the C/ path layout Wikipedia ZIMs use.
        assert_eq!(article_path("Beaconsfield, Quebec"), "C/Beaconsfield,_Quebec");
        assert_eq!(article_path("Salt"), "C/Salt");
        assert_eq!(
            article_path("17α-Hydroxycorticosteroid"),
            "C/17α-Hydroxycorticosteroid"
        );
        // Namespaced paths pass through unchanged: modern, legacy, and `-`.
        assert_eq!(article_path("C/Salt"), "C/Salt");
        assert_eq!(article_path("A/Foo"), "A/Foo");
        assert_eq!(article_path("-/x"), "-/x");
    }

    #[test]
    fn e2e_get_accepts_article_titles() {
        let dir = tempfile::tempdir().unwrap();
        let content = [
            TestEntry {
                namespace: b'C',
                url: "Salt",
                title: "Salt",
                mime: 0,
                body: b"<html><body><h1>Salt</h1><p>Salt is a mineral.</p>",
            },
            TestEntry {
                namespace: b'C',
                url: "Dishwasher_salt",
                title: "Dishwasher salt",
                mime: 0,
                body: b"<html><body><h1>Dishwasher salt</h1><p>Dishwasher salt is coarse-grained.</p>",
            },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, None);
        std::fs::write(dir.path().join("salt.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        let server = ZimMcpServer::new(library);

        // A single-word title resolves to its C/ path.
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "salt.zim", "path": "Salt" }),
        )
        .unwrap();
        let result = block_on(ZimGetTool::invoke(&server, params)).unwrap();
        assert_eq!(result.path, "C/Salt");
        assert_eq!(result.title, "Salt");

        // A multi-word title maps to the underscored path - a bare path
        // containing a space could never resolve by itself.
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "salt.zim", "path": "Dishwasher salt" }),
        )
        .unwrap();
        let result = block_on(ZimGetTool::invoke(&server, params)).unwrap();
        assert_eq!(result.path, "C/Dishwasher_salt");
        assert_eq!(result.title, "Dishwasher salt");

        // zim_get_section takes titles too.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "salt.zim", "path": "Salt", "section": "_intro" }),
        )
        .unwrap();
        let result = block_on(ZimGetSectionTool::invoke(&server, params)).unwrap();
        assert_eq!(result.section, "_intro");
        assert_eq!(result.title, "Salt");
        assert!(result.content.contains("Salt is a mineral"), "{:?}", result.content);

        // A title that matches nothing errors with the converted path.
        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "salt.zim", "path": "No Such Article" }),
        )
        .unwrap();
        assert!(matches!(
            block_on(ZimGetTool::invoke(&server, params)),
            Err(ToolError::NotFound(msg)) if msg.contains("C/No_Such_Article")
        ));
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
            block_on(ZimGetTool::invoke(&server, params)),
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
            block_on(ZimGetTool::invoke(&server, params)),
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
        let result = block_on(ZimGetSectionTool::invoke(&server, params)).unwrap();
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
        let result = block_on(ZimGetSectionTool::invoke(&server, params)).unwrap();
        assert_eq!(result.section, "Growth");
        assert!(result.content.contains("herbaceous plants"));

        // The reserved intro name: the intro region (everything before the
        // first heading), echoed as its reserved name.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "test.zim", "path": "Apple", "section": "_intro" }),
        )
        .unwrap();
        let result = block_on(ZimGetSectionTool::invoke(&server, params)).unwrap();
        assert_eq!(result.section, "_intro");
        assert!(
            result.content.contains("An <b>apple</b> is the fruit of"),
            "{:?}",
            result.content
        );
        assert!(!result.content.contains("10,000 years"), "{:?}", result.content);

        // Missing section.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "test.zim", "path": "Banana", "section": "Nope" }),
        )
        .unwrap();
        assert!(matches!(
            block_on(ZimGetSectionTool::invoke(&server, params)),
            Err(ToolError::SectionNotFound(_))
        ));
    }
}
