//! HTML -> Markdown conversion (the Rust port of wikizim_parser's
//! html2md.py, using the lxml-shaped DOM of htmldom). Wikipedia articles
//! (div.mw-parser-output) convert exactly like the Python module; any
//! other HTML page falls back to its <body> element.
//!
//! The public entry point is [`html_to_md`]; also exported: [`inline_text`],
//! [`block_children_md`], and (via [`tables`]) `render_table`.

use crate::cleanup;
use crate::htmldom::{Dom, NodeId, NodeKind, NodeRef};
use crate::tables;
use crate::util::{collapse_space_tab, collapse_ws, fre, fre_sub, parse_query, percent_decode, re, urlsplit};

// ---------------------------------------------------------------------------
// Conventions copied from wiki2md.py
// ---------------------------------------------------------------------------

// Section headings whose entire section is dropped live in cleanup.rs.

/// Link namespaces dropped entirely (files/media/categories): the whole
/// link vanishes while its surroundings stay.
const DROP_LINK_NS: &[&str] = &["file:", "image:", "media:", "category:"];

fn in_dropped_ns(target: &str) -> bool {
    let target = target.trim_start_matches(':');
    match target.split_once(':') {
        Some((head, _)) => {
            let head = format!("{}:", head.to_lowercase());
            DROP_LINK_NS.contains(&head.as_str())
        }
        None => false,
    }
}

/// Wiki language name -> markdown fence language.
const LANG_MAP: &[(&str, &str)] = &[
    ("c++", "cpp"), ("cxx", "cpp"), ("python", "python"), ("py", "python"),
    ("javascript", "javascript"), ("js", "javascript"), ("typescript", "typescript"),
    ("ts", "typescript"), ("c#", "csharp"), ("csharp", "csharp"), ("cs", "csharp"),
    ("c", "c"), ("cpp", "cpp"), ("java", "java"), ("kotlin", "kotlin"),
    ("scala", "scala"), ("swift", "swift"), ("go", "go"), ("rust", "rust"),
    ("ruby", "ruby"), ("perl", "perl"), ("php", "php"), ("lua", "lua"),
    ("r", "r"), ("matlab", "matlab"), ("haskell", "haskell"), ("fortran", "fortran"),
    ("pascal", "pascal"), ("html", "html"), ("xml", "xml"), ("css", "css"),
    ("json", "json"), ("yaml", "yaml"), ("yml", "yaml"), ("toml", "toml"),
    ("ini", "ini"), ("bash", "bash"), ("sh", "bash"), ("shell", "bash"),
    ("zsh", "bash"), ("sql", "sql"), ("makefile", "makefile"), ("cmake", "cmake"),
    ("diff", "diff"), ("dockerfile", "docker"), ("nginx", "nginx"),
    ("tex", "latex"), ("latex", "latex"), ("plain", ""), ("text", ""),
    ("wikitext", ""), ("wiki", ""), ("console", "console"), ("output", "text"),
];

fn fence_lang(raw: &str) -> String {
    let raw = raw.to_lowercase();
    LANG_MAP
        .iter()
        .find(|(k, _)| *k == raw)
        .map(|(_, v)| v.to_string())
        .unwrap_or(raw)
}

fn heading_level(tag: &str) -> Option<u32> {
    match tag {
        "h1" => Some(1),
        "h2" => Some(2),
        "h3" => Some(3),
        "h4" => Some(4),
        "h5" => Some(5),
        "h6" => Some(6),
        _ => None,
    }
}

/// Elements dropped with their entire contents (citation/maintenance/metadata).
const DROP_CLASSES: &[&str] = &[
    "reference", "noprint", "mw-editsection", "metadata", "shortdescription",
    "mw-jump-link", "mw-indicators", "catlinks", "printfooter",
    "mw-references-wrap", "reflist", "refbegin", "navbox", "sistersitebox",
    "sidebar", "side-box", "ambox", "mw-empty-elt", "toc", "navbox-styles",
    "thumb", "mw-editsection-bracket", "mw-ref", "referencetooltip",
    "mw-cite-backlink", "hatnote-dummy", "ext-phonos",
    "ext-phonos-PhonosButton", "ext-phonos-attribution",
];

const DROP_TAGS: &[&str] = &[
    "style", "script", "link", "figcaption", "figure", "img", "gallery",
    "input", "button", "audio", "video", "source",
];

/// True when `el` is a <math> element or contains one with a LaTeX source
/// (alttext/alt attribute) among its descendants: display:none MathML
/// wrappers stay alive so the math branch can extract the LaTeX.
fn carries_math(el: NodeRef) -> bool {
    if el.tag() == Some("math") {
        return true;
    }
    el.find_all("math").iter().any(|m| {
        m.attr("alttext").is_some() || m.attr("alt").is_some()
    })
}

/// True when this element and its contents vanish entirely.
pub(crate) fn is_dropped(el: NodeRef) -> bool {
    let tag = match el.tag() {
        Some(t) => t,
        None => return true, // comment / PI
    };
    if DROP_TAGS.contains(&tag) {
        return true;
    }
    if el.attr("role") == Some("navigation") {
        return true;
    }
    if tag == "a" && is_wikidata_badge(el) {
        return true;
    }
    if el.has_any_class(DROP_CLASSES) {
        return true;
    }
    if el
        .attr("typeof")
        .unwrap_or("")
        .split_whitespace()
        .any(|t| t == "mw:File")
    {
        return true;
    }
    let sty = el.attr("style").unwrap_or("");
    if sty.replace(' ', "").contains("display:none") || sty.contains("display: none") {
        // MediaWiki renders formulas twice: a visible image fallback plus a
        // hidden MathML twin; a display:none element carrying <math>
        // survives so its alttext LaTeX can be extracted.
        if !carries_math(el) {
            return true;
        }
    }
    if el.attr("aria-hidden") == Some("true") && matches!(tag, "span" | "sup" | "div") {
        return true;
    }
    if tag == "sup" {
        // Note-reference superscripts vanish with the References machinery.
        let ident = format!(
            "{} {}",
            el.attr("id").unwrap_or(""),
            el.attr("class").unwrap_or("")
        );
        if ["cite", "citation", "ref"].iter().any(|w| ident.contains(w)) {
            return true;
        }
        if el.find_all("a").iter().any(|a| {
            let href = a.attr("href").unwrap_or("");
            href.starts_with("#endnote") || href.starts_with("#cite_note")
        }) {
            return true;
        }
    }
    false
}

pub(crate) fn is_infobox_table(el: NodeRef) -> bool {
    el.tag() == Some("table") && el.has_class("infobox")
}

/// A French-Wikipedia (Parsoid) infobox container div.
pub(crate) fn is_infobox_wrapper(el: NodeRef) -> bool {
    el.tag() == Some("div") && el.has_class("infobox")
}

/// True for the single-letter Wikidata item badge links ('[d]').
fn is_wikidata_badge(el: NodeRef) -> bool {
    let href = el.attr("href").unwrap_or("");
    if !href.contains("wikidata.org") {
        return false;
    }
    let parts = urlsplit(href);
    if !matches!(parts.netloc.to_lowercase().as_str(), "wikidata.org" | "www.wikidata.org") {
        return false;
    }
    if !parts.path.starts_with("/wiki/") {
        return false;
    }
    let label = collapse_ws(&el.text_content()).trim().to_string();
    label.chars().count() == 1 && label.chars().next().map_or(false, |c| c.is_alphabetic())
}

/// Direct children of `el` that are rendered at all.
pub(crate) fn renderable_children<'a>(el: NodeRef<'a>) -> Vec<NodeRef<'a>> {
    el.children()
        .filter(|c| c.is_element() && !is_dropped(*c))
        .collect()
}

// ---------------------------------------------------------------------------
// Link/URL helpers
// ---------------------------------------------------------------------------

/// Make a URL safe inside a Markdown `[label](url)`: unbalanced parentheses
/// are percent-encoded.
fn md_link_url(url: &str) -> String {
    if url.matches('(').count() != url.matches(')').count() {
        return url.replace('(', "%28").replace(')', "%29");
    }
    url.to_string()
}

/// Wrap contents in Markdown emphasis markers; edge whitespace moves
/// outside the markers so `* vis *` never breaks the run.
fn wrap_md_emphasis(inner: &str, mark: &str) -> String {
    let core = inner.trim();
    if core.is_empty() {
        return inner.to_string();
    }
    let lead = &inner[..inner.len() - inner.trim_start().len()];
    let trail = &inner[inner.trim_end().len()..];
    format!("{}{}{}{}{}", lead, mark, core, mark, trail)
}

/// Escape literal asterisks in a plain-text fragment; `$...$` math spans
/// are masked first (LaTeX is verbatim).
pub(crate) fn escape_plain_asterisks(chunk: &str) -> String {
    if !chunk.contains('*') {
        return chunk.to_string();
    }
    // Split on `$...$` math spans (Python's _MATH_SPLIT_RE.split).
    let mut parts: Vec<&str> = Vec::new();
    let bytes = chunk.as_bytes();
    let mut rest_pos = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            if let Some(q) = chunk[i + 1..].find('$') {
                if !chunk[i + 1..i + 1 + q].contains('\n') {
                    if rest_pos < i {
                        parts.push(&chunk[rest_pos..i]);
                    }
                    parts.push(&chunk[i..i + q + 2]);
                    i += q + 2;
                    rest_pos = i;
                    continue;
                }
            }
        }
        i += 1;
    }
    if rest_pos < chunk.len() {
        parts.push(&chunk[rest_pos..]);
    }
    let mut out = String::new();
    for part in parts {
        if part.starts_with('$') && part.ends_with('$') && part.len() > 1 {
            out.push_str(part);
        } else {
            out.push_str(&part.replace('*', "\\*"));
        }
    }
    out
}

/// (path, fragment) of an href; path has no './' prefix or query string.
fn split_href(href: &str) -> (String, String) {
    let href = href.strip_prefix("./").unwrap_or(href);
    let (path, frag) = match href.split_once('#') {
        Some((p, f)) => (p, f),
        None => (href, ""),
    };
    let path = path.split('?').next().unwrap_or("");
    (path.to_string(), frag.to_string())
}

/// The wikilink target from already-split href parts (percent-decoded,
/// underscores folded, `#fragment` appended unless it is a citation).
fn link_target_parts(path: &str, frag: &str) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let path = path.strip_prefix(':').unwrap_or(path);
    let decoded = if path.contains('%') { percent_decode(path) } else { path.to_string() };
    let mut target = decoded.replace('_', " ");
    if !frag.is_empty() && !frag.starts_with("cite_note") {
        target.push('#');
        target.push_str(&frag.replace('_', " "));
    }
    Some(target)
}

/// Lower-cased scheme of an href ("http", "geo", ...) or None.
fn href_scheme(href: &str) -> Option<String> {
    let i = href.find(':')?;
    if i == 0 {
        return None;
    }
    let head = &href[..i];
    let ok = head
        .chars()
        .next()
        .map_or(false, |c| c.is_ascii_alphabetic())
        && head
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if ok {
        Some(head.to_lowercase())
    } else {
        None
    }
}

/// URL-path extensions treated as media files: such anchors drop entirely.
const MEDIA_URL_EXTS: &[&str] = &[
    ".mp3", ".ogg", ".oga", ".spx", ".flac", ".wav", ".mid", ".midi",
    ".opus", ".ogv", ".webm", ".mp4",
];

fn seg_is_media(seg: &str) -> bool {
    match seg.rfind('.') {
        Some(dot) if dot > 0 => {
            let ext = seg[dot..].to_lowercase();
            MEDIA_URL_EXTS.contains(&ext.as_str())
        }
        _ => false,
    }
}

/// True when `href` points at a media file (upload.wikimedia.org host, or
/// a media-file extension on the URL path).
fn is_media_href(href: &str) -> bool {
    if href.is_empty() {
        return false;
    }
    let plain = !href.contains(':')
        && !href.contains('%')
        && !href.contains('#')
        && !href.contains('?')
        && !href.contains('\t')
        && !href.contains('\r')
        && !href.contains('\n')
        && !href.starts_with("//")
        && href.as_bytes()[0] > b' ';
    if plain {
        let seg = href.rsplit('/').next().unwrap_or("");
        return seg_is_media(seg);
    }
    let parts = urlsplit(href);
    if parts.netloc.to_lowercase() == "upload.wikimedia.org" {
        return true;
    }
    let decoded_path = percent_decode(&parts.path);
    let seg = decoded_path.rsplit('/').next().unwrap_or("");
    seg_is_media(seg)
}

/// Article title of a Wikipedia edit/admin URL (action=edit), or None.
fn wiki_edit_target(href: &str) -> Option<String> {
    if href.is_empty() {
        return None;
    }
    let parts = urlsplit(href);
    let host = parts.netloc.to_lowercase();
    if host != "wikipedia.org" && !host.ends_with(".wikipedia.org") {
        return None;
    }
    if parts.path.to_lowercase() != "/w/index.php" {
        return None;
    }
    let q = parse_query(&parts.query);
    let action = q
        .iter()
        .find(|(k, _)| k == "action")
        .map(|(_, v)| v.trim().to_lowercase());
    if action.as_deref() != Some("edit") {
        return None;
    }
    let title = q
        .iter()
        .find(|(k, _)| k == "title")
        .map(|(_, v)| v.trim())?;
    if title.is_empty() {
        return None;
    }
    Some(title.replace('_', " "))
}

/// A hatnote's duplicate article mention around the [[title]] wikilink
/// rendered from its edit-widget anchor.
const HN_MENTION: &str = concat!(
    r"(?<!\S)(\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]|",
    r"([^\[\]\s][^\[\]]*?))[\s.,;:!?]*\[\[(?:\2|\3)\]\]"
);

/// Keep a wikilink mention verbatim; absorb a bare-text mention into a
/// [[title]] link.
fn absorb_hatnote_mentions(txt: &str) -> String {
    fre_sub(fre(HN_MENTION), txt, |c| {
        let g1 = c.get(1).map(|m| m.as_str()).unwrap_or("");
        if g1.starts_with("[[") {
            g1.to_string()
        } else {
            format!("[[{}]]", g1)
        }
    })
}

/// `*[[Target]]*` when the label equals the target, else `*[[Target|label]]*`.
fn emph_wikilink(mark: &str, target: &str, core: &str) -> String {
    if core == target {
        format!("{}[[{}]]{}", mark, target, mark)
    } else {
        format!("{}[[{}|{}]]{}", mark, target, core, mark)
    }
}

/// (mark, core) when `label` is exactly one whole emphasis run.
fn emph_is_whole(label: &str) -> Option<(&'static str, &str)> {
    for mark in ["**", "*"] {
        if label.starts_with(mark) && label.ends_with(mark) && label.len() > 2 * mark.len() {
            let core = &label[mark.len()..label.len() - mark.len()];
            if !core.contains(mark) {
                return Some((mark, core));
            }
        }
    }
    None
}

/// Whitespace stranded before dropped-element punctuation is pulled up.
/// Whitespace stranded before dropped-element punctuation is pulled up.
fn drop_boundary_punct(tail: &str) -> bool {
    tail.chars()
        .next()
        .map_or(false, |c| matches!(c, ';' | ':' | ',' | '.' | ')' | ']' | '!' | '?'))
}

/// HTML block-level tags (what would start a new block when rendered).
const BLOCK_LEVEL_TAGS: &[&str] = &[
    "address", "article", "aside", "blockquote", "center", "details",
    "dialog", "dir", "div", "dl", "dd", "dt", "fieldset", "figcaption",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hgroup", "hr", "li", "main", "menu", "nav", "ol", "p",
    "pre", "section", "summary", "table", "ul",
];

/// Classes whose template CSS overrides the tag's block display.
const INLINE_DISPLAY_CLASSES: &[&str] = &["ib-settlement-fn"];

/// True when `ch` would start a new block when rendered.
fn starts_block(ch: NodeRef) -> bool {
    let Some(tag) = ch.tag() else { return false };
    if !BLOCK_LEVEL_TAGS.contains(&tag) {
        return false;
    }
    if ch.has_any_class(INLINE_DISPLAY_CLASSES) {
        return false;
    }
    !re(r"display\s*:\s*inline").is_match(ch.attr("style").unwrap_or(""))
}

// ---------------------------------------------------------------------------
// Inline rendering
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum BrMode {
    Space,
    Keep,
    Nl,
}

#[derive(Clone, Copy)]
struct InlineCtx {
    no_escape: bool,
    in_link: bool,
    br_mode: BrMode,
    in_hatnote: bool,
}

impl Default for InlineCtx {
    fn default() -> Self {
        InlineCtx { no_escape: false, in_link: false, br_mode: BrMode::Space, in_hatnote: false }
    }
}

/// Render one element (or text node) in inline context.
fn render_inline(node: NodeRef, ctx: InlineCtx) -> String {
    if let NodeKind::Text(t) = node.kind() {
        return if ctx.no_escape {
            t.to_string()
        } else {
            escape_plain_asterisks(t)
        };
    }
    if !node.is_element() || is_dropped(node) {
        return String::new();
    }
    let tag = node.tag().unwrap_or("");
    match tag {
        "a" => render_anchor(node, ctx),
        "i" | "em" | "b" | "strong" => {
            let inner = render_children(node, ctx);
            if ctx.in_link {
                return inner;
            }
            let mark = if tag == "i" || tag == "em" { "*" } else { "**" };
            wrap_md_emphasis(&inner, mark)
        }
        "sup" | "sub" => {
            // Bracketed sups are citation/note markers, not superscripts;
            // empty slots emit nothing (orphan ~~ / ^^ pairs would render
            // as GFM strikethrough).
            let mark = if tag == "sup" { "^" } else { "~" };
            let inner = render_children(node, ctx).trim().to_string();
            if tag == "sup" && inner.starts_with('[') && inner.ends_with(']') {
                return String::new();
            }
            if inner.is_empty() {
                return String::new();
            }
            format!("{}{}{}", mark, inner, mark)
        }
        "br" => {
            if ctx.br_mode == BrMode::Keep {
                return "<br>".to_string();
            }
            // The layout <br> of a CSS-stacked sup/sub construct only
            // stacks both scripts in one visual slot: it is not a
            // separator, so it renders as nothing.
            if let (Some(prev), Some(next)) = (node.prev_element(), node.next_element()) {
                let is_script = |n: NodeRef| matches!(n.tag(), Some("sup") | Some("sub"));
                if is_script(prev) && is_script(next) {
                    return String::new();
                }
            }
            match ctx.br_mode {
                BrMode::Nl => "\n".to_string(),
                _ => " ".to_string(),
            }
        }
        "code" => {
            // fresh context: '*' verbatim, never inside a link label
            let inner = render_children(node, InlineCtx { no_escape: true, ..Default::default() });
            let inner = inner.replace('\n', " ");
            let inner = inner.trim();
            if inner.is_empty() {
                String::new()
            } else {
                format!("`{}`", inner)
            }
        }
        "math" => {
            let Some(alt) = node.attr("alttext").or_else(|| node.attr("alt")) else {
                return String::new();
            };
            let alt = alt.replace(r"\displaystyle ", "");
            if node.attr("display") == Some("block") {
                format!("\n\n$$\n{}\n$$\n\n", alt)
            } else {
                format!("${}$", alt)
            }
        }
        "pre" => {
            let content = node.text_content();
            let code = content.trim_matches('\n');
            format!("\n\n```\n{}\n```\n\n", code)
        }
        "table" => String::new(),
        _ => render_children(node, ctx),
    }
}

/// Render the children of `el` in inline context: a separator kept at
/// block boundaries, punctuation pulled up after dropped elements. The
/// leading text is handled upfront; every child's tail is consumed at the
/// child (text nodes are tails; comments render empty but keep theirs).
fn render_children(el: NodeRef, ctx: InlineCtx) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut last: Option<char> = None;
    if let Some(t) = el.text() {
        out.push(if ctx.no_escape { t.to_string() } else { escape_plain_asterisks(t) });
        if !t.is_empty() {
            last = t.chars().last();
        }
    }
    for child in el.children() {
        if child.is_text() {
            continue; // the leading text or a tail consumed below
        }
        let piece = if child.is_element() { render_inline(child, ctx) } else { String::new() };
        let tail = child.tail().unwrap_or("");
        // A block-level child starts a new block: keep one space between
        // the inline content before it and the block unless the HTML
        // already separates them.
        if let Some(l) = last {
            if !piece.is_empty()
                && !l.is_whitespace()
                && !piece.chars().next().map_or(true, |c| c.is_whitespace())
                && child.is_element()
                && starts_block(child)
            {
                out.push(" ".to_string());
                last = Some(' ');
            }
        }
        // A dropped element directly followed by punctuation: pull up the
        // whitespace stranded between the surviving text and the
        // punctuation.
        if piece.is_empty() && !out.is_empty() && !tail.is_empty() && drop_boundary_punct(tail) {
            while let Some(s) = out.last() {
                if s.trim_matches(|c| c == ' ' || c == '\t').is_empty() {
                    out.pop();
                } else {
                    break;
                }
            }
            if let Some(s) = out.last_mut() {
                let keep = s.trim_end_matches(|c| c == ' ' || c == '\t').len();
                s.truncate(keep);
            }
            last = out.last().and_then(|s| s.chars().last());
        }
        let piece_empty = piece.is_empty();
        if !piece_empty {
            last = piece.chars().last();
        }
        out.push(piece);
        if !tail.is_empty() {
            out.push(if ctx.no_escape { tail.to_string() } else { escape_plain_asterisks(tail) });
            last = tail.chars().last();
        }
    }
    out.join("")
}


/// Render an <a> element: wikilink, external link, fragment-only label,
/// or dropped media link.
fn render_anchor(el: NodeRef, ctx: InlineCtx) -> String {
    let href = el.attr("href").unwrap_or("");
    // Media-file links are dropped entirely, like images.
    if is_media_href(href) {
        return String::new();
    }
    // Whole-label emphasis: the anchor's content is exactly one <i>/<em>
    // or <b>/<strong> element.
    let mut emph_mark: Option<&str> = None;
    if !ctx.in_link {
        let kids: Vec<NodeRef> = el.element_children().collect();
        if kids.len() == 1
            && el.text().map_or(true, |t| t.trim().is_empty())
            && kids[0].tail().map_or(true, |t| t.trim().is_empty())
        {
            emph_mark = match kids[0].tag() {
                Some("i") | Some("em") => Some("*"),
                Some("b") | Some("strong") => Some("**"),
                _ => None,
            };
        }
    }
    // fresh link-label context: '*' verbatim, emphasis stripped (the
    // outer <br> and hatnote contexts do not leak into the label)
    let label = render_children(el, InlineCtx { no_escape: true, in_link: true, ..Default::default() });
    let label_stripped = label.trim().to_string();

    // Hatnote edit/admin anchor: an edit-section widget inside a hatnote
    // renders as a clean wikilink to the URL's title parameter.
    if ctx.in_hatnote {
        if let Some(edit_target) = wiki_edit_target(href) {
            if in_dropped_ns(&edit_target) {
                return String::new();
            }
            return format!("[[{}]]", edit_target);
        }
    }

    let scheme = href_scheme(href);
    if el.has_class("external") || el.attr("rel") == Some("nofollow") || scheme.is_some() {
        if scheme.as_deref() == Some("geo") {
            // coordinate microformat links: the label carries the same
            // data as the URL; render coordinates as plain text
            return label;
        }
        if label_stripped.is_empty() || label_stripped == href {
            return format!("<{}>", href);
        }
        let lead = &label[..label.len() - label.trim_start().len()];
        let trail = &label[label.trim_end().len()..];
        return format!("{}[{}]({}){}", lead, label_stripped, md_link_url(href), trail);
    }

    // internal link (or interwiki / fragment-only)
    let (path, frag) = split_href(href);
    if path.is_empty() {
        if href.starts_with("#cite_note") {
            return String::new(); // citation residue: drop entirely
        }
        return label; // fragment-only: keep label text
    }
    let target = match link_target_parts(&path, &frag) {
        Some(t) if !t.is_empty() => t,
        _ => return label,
    };
    if in_dropped_ns(&target) {
        return String::new(); // [[File:…]]/[[Category:…]]-style link: dropped whole
    }
    if let Some(mark) = emph_mark {
        return emph_wikilink(mark, &target, &label_stripped);
    }
    if !ctx.in_link {
        if let Some((mark, core)) = emph_is_whole(&label_stripped) {
            return emph_wikilink(mark, &target, core);
        }
    }
    // omit |label when the label equals the target page title
    let bare = path.replace('_', " ");
    let label_cmp = label_stripped.replace('_', " ");
    if label_cmp == bare || label_cmp == target {
        return format!("[[{}]]", target);
    }
    format!("[[{}|{}]]", target, label_stripped)
}

/// Render an element in inline context to markdown text (links,
/// bold/italic, sup/sub; citations dropped; plain asterisks escaped).
pub(crate) fn inline_text(el: NodeRef) -> String {
    render_inline(el, InlineCtx::default())
}

/// Inline rendering with plain-text asterisk escaping off (the tables
/// contract for HTML-table cell and caption content).
pub(crate) fn inline_raw(el: NodeRef, keep_br: bool) -> String {
    render_inline(
        el,
        InlineCtx {
            no_escape: true,
            br_mode: if keep_br { BrMode::Keep } else { BrMode::Space },
            ..Default::default()
        },
    )
}

/// The default inline rendering (used by the infobox renderer).
pub(crate) fn render_inline_default(el: NodeRef) -> String {
    render_inline(el, InlineCtx::default())
}

// ---------------------------------------------------------------------------
// Block-level rendering
// ---------------------------------------------------------------------------

/// One recursive pass over a <ul>/<ol>: the Markdown lines of its items,
/// two-space indent per nesting level. raw=True selects the HTML-table-
/// cell variant (`* item` markers, `<br>` kept, text verbatim).
pub(crate) fn list_item_lines(list_el: NodeRef, depth: usize, raw: bool) -> Vec<String> {
    let marker = if list_el.tag() == Some("ol") {
        "1. "
    } else if raw {
        "* "
    } else {
        "- "
    };
    let indent = "  ".repeat(depth);
    let br_mode = if raw { BrMode::Keep } else { BrMode::Space };
    let mut out: Vec<String> = Vec::new();
    for li in list_el.children() {
        if li.tag() != Some("li") {
            continue;
        }
        let mut parts: Vec<String> = Vec::new();
        let mut subs: Vec<NodeRef> = Vec::new();
        for ch in li.children() {
            match ch.kind() {
                NodeKind::Text(t) => {
                    parts.push(if raw { t.to_string() } else { escape_plain_asterisks(t) });
                }
                NodeKind::Element { .. } if matches!(ch.tag(), Some("ul") | Some("ol")) => {
                    subs.push(ch);
                }
                NodeKind::Element { .. } => {
                    parts.push(render_inline(ch, InlineCtx { no_escape: raw, br_mode, ..Default::default() }));
                }
                _ => {}
            }
        }
        let text = collapse_ws(&parts.join("")).trim().to_string();
        if !text.is_empty() {
            out.push(format!("{}{}{}", indent, marker, text));
        }
        for sub in subs {
            out.extend(list_item_lines(sub, depth + 1, raw));
        }
    }
    out
}

/// Render <ul>/<ol> with 2-space indent per nesting level.
pub(crate) fn render_list(el: NodeRef, depth: usize) -> String {
    list_item_lines(el, depth, false)
        .into_iter()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Definition list: <dt> -> `- **term**`, <dt>+<dd> -> `- **term**: def`,
/// lone <dd> -> `- def`; 2-space indent per nesting level.  A <dd>'s
/// definition text is its inline content with any direct <ul>/<ol>
/// children excluded (they render as sub-list lines instead of leaking,
/// flattened, into the definition); it is attached to the open term's
/// line — or emitted as a lone `- ` item — *before* the sub-lists, so
/// definition and sub-list stay in document order.
pub(crate) fn render_dl(el: NodeRef, depth: usize) -> String {
    let mut out: Vec<String> = Vec::new();
    let indent = "  ".repeat(depth);
    let mut last_term: Option<String> = None;
    let mut term_has_def = false;
    for ch in el.children() {
        if !ch.is_element() || is_dropped(ch) {
            continue;
        }
        match ch.tag() {
            Some("dt") => {
                let term = inline_text(ch).trim().to_string();
                last_term = Some(term.clone());
                term_has_def = false;
                if !term.is_empty() {
                    out.push(format!("{}- **{}**", indent, term));
                }
            }
            Some("dd") => {
                // Definition text: the dd's inline content with its direct
                // <ul>/<ol> children excluded — their items render as
                // sub-list lines below and must not leak (flattened) into
                // the definition.  Skipped lists' tails arrive as the
                // following Text child; Comments/PIs are skipped too.
                let mut parts: Vec<String> = Vec::new();
                for sub in ch.children() {
                    match sub.kind() {
                        NodeKind::Text(t) => parts.push(escape_plain_asterisks(t)),
                        NodeKind::Element { .. } if matches!(sub.tag(), Some("ul") | Some("ol")) => {}
                        NodeKind::Element { .. } => parts.push(inline_text(sub)),
                        _ => {}
                    }
                }
                let defn = collapse_ws(&parts.join("")).trim().to_string();
                // Attach the definition BEFORE any sub-list renders: an
                // open term's line is still out[-1] here (nothing was
                // pushed since it), so the definition lands on the term
                // line and the sub-lists follow it in document order.
                if !defn.is_empty() {
                    match last_term.as_deref() {
                        Some(t) if !t.is_empty() && !term_has_def => {
                            let idx = out.len() - 1;
                            out[idx] = format!("{}: {}", out[idx], defn);
                            term_has_def = true;
                        }
                        _ => out.push(format!("{}- {}", indent, defn)),
                    }
                }
                for sub in ch.element_children() {
                    if matches!(sub.tag(), Some("ul") | Some("ol")) && !is_dropped(sub) {
                        let sub_md = render_list(sub, depth + 1);
                        if !sub_md.is_empty() {
                            out.push(sub_md);
                        }
                    }
                }
            }
            Some("ul") | Some("ol") => {
                let sub_md = render_list(ch, depth + 1);
                if !sub_md.is_empty() {
                    out.push(sub_md);
                }
            }
            Some("dl") => {
                let sub_md = render_dl(ch, depth + 1);
                if !sub_md.is_empty() {
                    out.push(sub_md);
                }
            }
            _ => {}
        }
    }
    out.into_iter()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render one <p>: <br> splits the paragraph (the <br>'s tail opens the
/// next segment — the next quote line in a blockquote), everything else
/// is inline.
fn render_paragraph(p: NodeRef, in_blockquote: bool) -> Vec<String> {
    let ctx = InlineCtx {
        br_mode: if in_blockquote { BrMode::Nl } else { BrMode::Space },
        ..Default::default()
    };
    let mut segs: Vec<String> = vec![String::new()];
    if let Some(t) = p.text() {
        segs[0].push_str(&escape_plain_asterisks(t));
    }
    for ch in p.children() {
        if ch.is_text() {
            continue; // the leading text or a tail consumed at its owner
        }
        if ch.is_element() && ch.tag() == Some("br") {
            segs.push(escape_plain_asterisks(ch.tail().unwrap_or("")));
            continue;
        }
        let piece = if ch.is_element() { render_inline(ch, ctx) } else { String::new() };
        let tail = ch.tail().unwrap_or("");
        if piece.is_empty()
            && !tail.is_empty()
            && !segs.last().unwrap().is_empty()
            && drop_boundary_punct(tail)
        {
            let last_seg = segs.last_mut().unwrap();
            let keep = last_seg.trim_end_matches(|c| c == ' ' || c == '\t').len();
            last_seg.truncate(keep);
        }
        segs.last_mut().unwrap().push_str(&piece);
        if !tail.is_empty() {
            segs.last_mut().unwrap().push_str(&escape_plain_asterisks(tail));
        }
    }
    let mut out: Vec<String> = Vec::new();
    for s in &segs {
        let s = if s.contains("  ") || s.contains('\t') { collapse_space_tab(s) } else { s.clone() };
        let s = s.trim();
        if !s.is_empty() {
            out.push(s.to_string());
        } else if in_blockquote {
            out.push(String::new());
        }
    }
    out
}

/// Fenced code block; language from a `lang-xxx` class via LANG_MAP.
fn render_pre(pre: NodeRef) -> String {
    let lang = pre
        .class_tokens()
        .find_map(|c| c.strip_prefix("lang-").map(fence_lang))
        .unwrap_or_default();
    let content = pre.text_content();
    let code = content.trim_matches('\n');
    format!("```{}\n{}\n```", lang, code)
}

/// `> ` line-per-line blockquote; empty lines inside become `>`.
fn render_blockquote(bq: NodeRef) -> String {
    let inner = block_children_md(bq, None, true);
    let inner = inner.trim_matches('\n');
    if inner.is_empty() {
        return String::new();
    }
    inner
        .split('\n')
        .map(|l| {
            if l.trim().is_empty() {
                ">".to_string()
            } else {
                format!("> {}", l)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A generic block container: recurse into its children; hatnote
/// containers render as standalone fully-italic paragraphs.
fn render_block_container(el: NodeRef, in_blockquote: bool) -> String {
    if el.attr("role") == Some("note") || el.has_class("hatnote") {
        return render_hatnote(el);
    }
    block_children_md(el, None, in_blockquote)
}

/// Render a hatnote container as a standalone fully-italic paragraph; only
/// [[...]] links survive (emphasis flattened) and anchors pointing at a
/// wiki edit/admin URL become [[title]] wikilinks.
fn render_hatnote(el: NodeRef) -> String {
    let txt = collapse_ws(&render_inline(
        el,
        InlineCtx { in_link: true, in_hatnote: true, ..Default::default() },
    ))
    .trim()
    .to_string();
    if txt.is_empty() {
        return String::new();
    }
    let has_edit = el
        .find_all("a")
        .iter()
        .any(|a| wiki_edit_target(a.attr("href").unwrap_or("")).is_some());
    let txt = if has_edit { absorb_hatnote_mentions(&txt) } else { txt };
    let txt = txt.trim().to_string();
    if txt.is_empty() {
        return String::new();
    }
    format!("*{}*", txt)
}

/// (level, markdown heading text) for a bare hN or a mw-heading wrapper.
fn heading_of(el: NodeRef) -> Option<(u32, String)> {
    let tag = el.tag()?;
    let h = if heading_level(tag).is_some() {
        el
    } else if tag == "div" && el.class_tokens().any(|c| c.starts_with("mw-heading")) {
        el.element_children()
            .find(|ch| ch.tag().and_then(heading_level).is_some())?
    } else {
        return None;
    };
    let mut level = heading_level(h.tag()?).unwrap();
    if level == 1 {
        level = 2; // in-body h1 is not expected; treat defensively as h2
    }
    let text = collapse_ws(&inline_text(h)).trim().to_string();
    Some((level, text))
}

/// Markdown of ONE block-level child element ('' when it renders to
/// nothing) — the per-child dispatch of block_children_md.
pub(crate) fn block_md(ch: NodeRef, in_blockquote: bool) -> String {
    let tag = ch.tag().unwrap_or("");
    match tag {
        "p" => {
            let p_lines = render_paragraph(ch, in_blockquote);
            if in_blockquote {
                p_lines.join("\n")
            } else {
                p_lines.join("\n\n")
            }
        }
        "ul" | "ol" => render_list(ch, 0),
        "dl" => render_dl(ch, 0),
        "blockquote" => render_blockquote(ch),
        "pre" => render_pre(ch),
        "table" => tables::render_table(ch),
        "div" => match heading_of(ch) {
            Some((level, text)) => format!("{} {}", "#".repeat(level as usize), text),
            None => render_block_container(ch, in_blockquote),
        },
        "span" => collapse_ws(&inline_text(ch)).trim().to_string(),
        t if heading_level(t).is_some() => match heading_of(ch) {
            Some((level, text)) => format!("{} {}", "#".repeat(level as usize), text),
            None => String::new(),
        },
        _ => render_block_container(ch, in_blockquote),
    }
}

/// Render an element's children as block Markdown (blocks joined with
/// blank lines). A non-empty `key_facts` block is emitted structurally
/// immediately before the first heading (or at the end for lead-only
/// pages). `in_blockquote` keeps <br>-separated paragraph lines within one
/// quote paragraph.
pub(crate) fn block_children_md(
    el: NodeRef,
    key_facts: Option<&str>,
    in_blockquote: bool,
) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut key_facts_pending = key_facts.filter(|k| !k.is_empty());
    for ch in renderable_children(el) {
        if tables::is_infobox_container(ch) {
            continue; // infoboxes are rendered separately (key_facts_of)
        }
        let tag = ch.tag().unwrap_or("");
        // one heading lookup for the Key facts placement test
        let heading =
            if tag == "div" || heading_level(tag).is_some() { heading_of(ch) } else { None };
        if let (Some(kf), Some(_)) = (key_facts_pending, heading) {
            out.push(kf.to_string());
            key_facts_pending = None;
        }
        let md = block_md(ch, in_blockquote);
        if !md.is_empty() {
            out.push(md);
        }
    }
    if let Some(kf) = key_facts_pending {
        out.push(kf.to_string());
    }
    out.into_iter()
        .filter(|o| !o.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ---------------------------------------------------------------------------
// Top-level conversion
// ---------------------------------------------------------------------------

/// Find div.mw-parser-output (child of #mw-content-text); match by class
/// token, not exact class string.
fn get_parser_output(root: NodeRef) -> Option<NodeRef> {
    for ct in root.descendants() {
        if ct.attr("id") == Some("mw-content-text") {
            for ch in ct.element_children() {
                if ch.has_class("mw-parser-output") {
                    return Some(ch);
                }
            }
            for el in ct.descendants() {
                if el.is_element() && el.has_class("mw-parser-output") {
                    return Some(el);
                }
            }
        }
    }
    // last resort: first element with the class token anywhere
    root.descendants()
        .into_iter()
        .find(|el| el.is_element() && el.has_class("mw-parser-output"))
}

/// Normalize Parsoid-sectioned HTML back to the legacy flat shape: replace
/// every <section> wrapper under the body container by its children.
fn flatten_parsoid_sections(dom: &mut Dom, body: crate::htmldom::NodeId) {
    let has_section = dom.ref_(body).find("section").is_some();
    if !has_section {
        return;
    }
    let sections: Vec<crate::htmldom::NodeId> =
        dom.ref_(body).find_all("section").iter().map(|n| n.id()).collect();
    for sec in sections {
        dom.replace_with_children(sec);
    }
}

fn first_h1_title(dom: &Dom) -> Option<String> {
    for h1 in dom.root().find_all("h1") {
        if h1.attr("id") == Some("firstHeading") {
            return Some(collapse_ws(&h1.text_content()).trim().to_string());
        }
    }
    None
}

/// The rendered infoboxes as a '## Key facts' block ('' when none).
fn key_facts_of(dom: &Dom, lang: Option<&str>) -> String {
    let boxes = crate::infobox_html::extract_infoboxes(dom);
    crate::infobox_html::infoboxes_to_markdown(dom, &boxes, lang).trim().to_string()
}

/// Render the article body around the already-located container: Parsoid
/// sections flattened, blocks rendered, sections assembled, title line
/// prepended and the cleanup pass applied.
fn render_article(
    mut dom: Dom,
    body: crate::htmldom::NodeId,
    key_facts: String,
    title: Option<&str>,
    lang: Option<&str>,
) -> String {
    flatten_parsoid_sections(&mut dom, body);
    let body_md = block_children_md(
        dom.ref_(body),
        if key_facts.is_empty() { None } else { Some(&key_facts) },
        false,
    );
    let content = cleanup::assemble(&body_md, lang);
    let title = title.map(|t| t.to_string()).or_else(|| first_h1_title(&dom));
    let mut parts: Vec<String> = Vec::new();
    if let Some(t) = &title {
        parts.push(format!("# {}", t.trim()));
    }
    if !content.trim().is_empty() {
        parts.push(content);
    }
    cleanup::cleanup(&parts.join("\n\n"))
}

/// Convert HTML to Markdown.
///
/// A Wikipedia article (div.mw-parser-output under #mw-content-text)
/// renders exactly like `zim2zim.py --infobox` does: infoboxes as a '## Key
/// facts' block placed structurally after the intro. Any other page renders
/// from its `<body>` element (html5ever always materializes one, so head
/// junk — title/style/script/meta — stays out), which makes arbitrary
/// scraped pages convert too. On such pages the first in-body `<h1>`
/// supplies the `# Title` line when `title` is None, and an in-body `<h1>`
/// whose text equals the output title is removed from the body before
/// rendering (the title line replaces it); a differing `<h1>` stays and
/// renders as a `## ` heading. `lang` (a ZIM/BCP-47 code; 'fra'/'fre'/'fr'
/// select French) localizes the '## Key facts' heading and the dropped
/// boilerplate sections.
pub fn html_to_md(
    html_str: &str,
    title: Option<&str>,
    lang: Option<&str>,
) -> String {
    let mut dom = Dom::parse(html_str);
    let key_facts = key_facts_of(&dom, lang);
    // The wiki article container, else the page's <body>.
    let wiki_body = get_parser_output(dom.root()).map(|b| b.id());
    let body =
        wiki_body.unwrap_or_else(|| dom.root().find("body").map_or(dom.root().id(), |b| b.id()));
    let mut title = title.map(str::to_string);
    if wiki_body.is_none() {
        if title.is_none() {
            title = body_h1_title(&dom, body);
        }
        if let Some(t) = &title {
            drop_title_h1(&mut dom, body, t);
        }
    }
    render_article(dom, body, key_facts, title.as_deref(), lang)
}

/// The first in-body h1's collapsed text, or None when the body has no h1.
fn body_h1_title(dom: &Dom, body: NodeId) -> Option<String> {
    let h1 = dom.ref_(body).find("h1")?;
    let text = collapse_ws(&h1.text_content()).trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

/// Remove the first in-body h1 when its text equals `title`: the `# Title`
/// line would otherwise say it twice.
fn drop_title_h1(dom: &mut Dom, body: NodeId, title: &str) {
    let id = {
        let scope = dom.ref_(body);
        match scope.find("h1") {
            Some(h1) if collapse_ws(&h1.text_content()).trim() == title => Some(h1.id()),
            _ => None,
        }
    };
    if let Some(id) = id {
        dom.detach(id);
    }
}

#[cfg(test)]
mod tests {
    use super::html_to_md;

    #[test]
    fn comment_tail_survives() {
        let md = html_to_md(
            "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\"><p>in 1889.<!--note--> It was made from <a href=\"./Nitrocellulose\">nitrocellulose</a> known as nitrate.</p></div></div></body></html>",
            Some("Nitro"),
            None,
        );
        assert!(md.contains("1889. It was made"), "{md}");
    }

    #[test]
    fn dropped_namespace_links_vanish() {
        let md = html_to_md(
            "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\"><p>Text <a href=\"Category%3AFoo\"Category:Foo\">label</a><a href=\"./File%3ABar\">img</a>.</p></div></div></body></html>",
            Some("T"),
            None,
        );
        assert_eq!(md, "# T\n\nText.\n");
    }

    #[test]
    fn br_splits_the_paragraph_without_losing_text() {
        let md = html_to_md(
            "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\"><p>a<br>b <i>c</i></p></div></div></body></html>",
            Some("T"),
            None,
        );
        assert_eq!(md, "# T\n\na\n\nb *c*\n");
    }

    #[test]
    fn arbitrary_html_converts_from_the_body() {
        let md = html_to_md(
            "<html><head><title>Doc</title><style>p { margin: 0 }</style>\
             <meta name=\"x\" content=\"y\"></head><body>\
             <h1>Head</h1><p>Para <b>bold</b>.</p>\
             <p>See <a href=\"https://x.example/a?b=1&amp;c=2\">link</a>.</p>\
             </body></html>",
            None,
            None,
        );
        assert_eq!(md, "# Head\n\nPara **bold**.\n\nSee [link](https://x.example/a?b=1&c=2).\n");
    }

    #[test]
    fn in_body_h1_differs_from_the_passed_title_stays() {
        let md = html_to_md(
            "<html><body><h1>Intro</h1><p>Body text.</p></body></html>",
            Some("Doc"),
            None,
        );
        assert_eq!(md, "# Doc\n\n## Intro\n\nBody text.\n");
    }

    fn wiki_dl(inner: &str) -> String {
        html_to_md(
            &format!(
                "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\">{}</div></div></body></html>",
                inner
            ),
            Some("T"),
            None,
        )
    }

    /// A <dd> with both inline text and a sub-list: the definition goes on
    /// the term line, the sub-list renders after it — the sub-list's items
    /// must not also leak (flattened) into the definition text.
    #[test]
    fn dd_text_and_sublist_define_on_the_term_line_then_the_sublist() {
        let md = wiki_dl("<dl><dt>T</dt><dd>Def <i>x</i><ul><li>s1</li><li>s2</li></ul></dd></dl>");
        assert_eq!(md, "# T\n\n- **T**: Def *x*\n  - s1\n  - s2\n");
    }

    /// Lone <dd> with text and a sub-list: its own `- ` line, then the
    /// sub-list — no duplicated flattened text.
    #[test]
    fn lone_dd_with_text_and_sublist_puts_the_text_on_its_own_line() {
        let md = wiki_dl("<dl><dd>Def <i>x</i><ul><li>s1</li></ul></dd></dl>");
        assert_eq!(md, "# T\n\n- Def *x*\n  - s1\n");
    }

    /// A <dd> holding only a sub-list adds no definition: the term line is
    /// untouched and the sub-list follows it.
    #[test]
    fn dd_with_only_a_sublist_leaves_the_term_line_untouched() {
        let md = wiki_dl("<dl><dt>T</dt><dd><ul><li>s1</li><li>s2</li></ul></dd></dl>");
        assert_eq!(md, "# T\n\n- **T**\n  - s1\n  - s2\n");
    }
}
