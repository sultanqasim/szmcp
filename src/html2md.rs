//! HTML -> Markdown conversion (the Rust port of wikizim_parser's
//! html2md.py, using the lxml-shaped DOM of htmldom). Wikipedia articles
//! (div.mw-parser-output) convert exactly like the Python module; any
//! other HTML page falls back to its <body> element.
//!
//! The public entry point is [`html_to_md`]; also exported: [`inline_text`],
//! [`block_children_md`], and (via [`tables`]) `render_table`.

use crate::cleanup;
use crate::htmldom::{Dom, NodeKind, NodeRef};
use crate::tables;
use crate::util::{
    collapse_space_tab, collapse_ws, parse_query, percent_decode, prev_char, re, run_end, urlsplit,
};

// ---------------------------------------------------------------------------
// Conventions copied from wiki2md.py
// ---------------------------------------------------------------------------

// Section headings whose entire section is dropped live in cleanup.rs.

/// Link namespaces dropped entirely (files/media/categories): the whole
/// link vanishes while its surroundings stay.
const DROP_LINK_NS: &[&str] = &["file:", "image:", "media:", "category:"];

fn in_dropped_ns(target: &str) -> bool {
    let Some((head, _)) = target.trim_start_matches(':').split_once(':') else {
        return false;
    };
    DROP_LINK_NS.contains(&format!("{}:", head.to_lowercase()).as_str())
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
    match tag.as_bytes() {
        [b'h', d] if (b'1'..=b'6').contains(d) => Some((d - b'0') as u32),
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
    // mwoffliner's JS-pagination placeholders around a category page's
    // member lists ("Next items are not visible in browsers without
    // Javascript"): browser noise, never content.
    "mwo-cat-pagination",
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
        None => return true, // comment / processing instruction
    };
    if DROP_TAGS.contains(&tag) {
        return true;
    }
    // One pass over the attributes for the lookups below (each used to
    // cost its own linear `attr()` scan); first occurrence wins, like
    // `attr()`.
    let (mut role, mut type_of, mut style, mut aria_hidden, mut id, mut class) =
        (None, None, None, None, None, None);
    if let NodeKind::Element { attrs, .. } = el.kind() {
        for (name, value) in attrs {
            let slot = match name.as_str() {
                "role" => &mut role,
                "typeof" => &mut type_of,
                "style" => &mut style,
                "aria-hidden" => &mut aria_hidden,
                "class" => &mut class,
                "id" => &mut id,
                _ => continue,
            };
            if slot.is_none() {
                *slot = Some(value.as_str());
            }
        }
    }
    if role == Some("navigation") {
        return true;
    }
    if tag == "a" && is_wikidata_badge(el) {
        return true;
    }
    if class.unwrap_or("").split_whitespace().any(|c| DROP_CLASSES.contains(&c)) {
        return true;
    }
    if type_of.unwrap_or("").split_whitespace().any(|t| t == "mw:File") {
        return true;
    }
    if let Some(sty) = style {
        // Only styled elements with a `display` in the style reach the
        // de-spaced copy the check needs.
        if sty.contains("display")
            && (sty.replace(' ', "").contains("display:none") || sty.contains("display: none"))
        {
            // MediaWiki renders formulas twice: a visible image fallback plus a
            // hidden MathML twin; a display:none element carrying <math>
            // survives so its alttext LaTeX can be extracted.
            if !carries_math(el) {
                return true;
            }
        }
    }
    if aria_hidden == Some("true") && matches!(tag, "span" | "sup" | "div") {
        return true;
    }
    if tag == "sup" {
        // Note-reference superscripts vanish with the References machinery.
        let ident = format!("{} {}", id.unwrap_or(""), class.unwrap_or(""));
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
    // A '$' opens a math span when another '$' follows on the same line;
    // otherwise it stays plain text. A plain run that begins and ends
    // with a rejected '$' is left verbatim, like its Python original.
    fn push_plain(out: &mut String, part: &str) {
        if part.starts_with('$') && part.ends_with('$') && part.len() > 1 {
            out.push_str(part);
        } else {
            out.push_str(&part.replace('*', "\\*"));
        }
    }
    let mut out = String::new();
    let bytes = chunk.as_bytes();
    let mut plain_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            if let Some(q) = chunk[i + 1..].find('$') {
                if !chunk[i + 1..i + 1 + q].contains('\n') {
                    push_plain(&mut out, &chunk[plain_start..i]);
                    out.push_str(&chunk[i..i + q + 2]); // the $...$ span
                    i += q + 2;
                    plain_start = i;
                    continue;
                }
            }
        }
        i += 1;
    }
    push_plain(&mut out, &chunk[plain_start..]);
    out
}

/// (path, fragment) of an href; path has no './' prefix but keeps its
/// query string — `./Foo?x=1#Sec` yields ("Foo?x=1", "Sec") — so the
/// wikilink target still identifies the same resource (stripping `?x=1`
/// would collapse different queries onto one target).
fn split_href(href: &str) -> (String, String) {
    let href = href.strip_prefix("./").unwrap_or(href);
    let (path, frag) = match href.split_once('#') {
        Some((p, f)) => (p, f),
        None => (href, ""),
    };
    (path.to_string(), frag.to_string())
}

/// The wikilink target from already-split href parts (percent-decoded,
/// underscores folded, `#fragment` appended unless it is a citation).
/// The path may carry a query string (`Foo?x=1`); it decodes and folds
/// like the rest of the target.
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
    // Fast path for plain internal hrefs: with none of the urlsplit-
    // triggering characters present, urlsplit's cleaning is the identity —
    // except that it would trim trailing C0/space, which this path (like
    // the Python original's identical fast path) deliberately keeps.
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
        return seg_is_media(href.rsplit('/').next().unwrap_or(""));
    }
    let parts = urlsplit(href);
    if parts.netloc.to_lowercase() == "upload.wikimedia.org" {
        return true;
    }
    let decoded = percent_decode(&parts.path);
    seg_is_media(decoded.rsplit('/').next().unwrap_or(""))
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

/// The `[\s.,;:!?]*` gap and the `[[key]]` after a hatnote mention;
/// returns the offset just past the closing `]]`.
fn hn_close(s: &str, from: usize, key: &str) -> Option<usize> {
    let end = run_end(s, from, |c| {
        c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':' | '!' | '?')
    });
    if s[end..].starts_with("[[")
        && s[end + 2..].starts_with(key)
        && s[end + 2 + key.len()..].starts_with("]]")
    {
        Some(end + 2 + key.len() + 2)
    } else {
        None
    }
}

/// One match of the old hatnote-mention pattern starting exactly at byte
/// offset `i` (a char boundary; the `(?<!\S)` part is checked by the
/// caller): returns the mention text and the offset just past its
/// closing `[[…]]`.
///
/// The mention is a whole wikilink `[[target(#frag)?(|label)?]]` (the
/// key is the target) or lazily growing bare text (the key is the text
/// itself).  The wikilink parts are maximal plain-class runs anchored to
/// their lead chars, so the maximal scan decides them (no shorter run
/// can reach a `]]` the maximal one missed); the bare text grows one
/// char at a time and the first length whose closing link matches wins.
fn hn_mention_at(s: &str, i: usize) -> Option<(&str, usize)> {
    let b = s.as_bytes();
    if s[i..].starts_with("[[") {
        // [[target(#frag)?(|label)?]]: target = 1+ chars none of [ ] | #,
        // fragment after '#' (0+ chars none of [ ] |), label after '|'
        // (0+ chars none of [ ]).
        let t0 = i + 2;
        let mut p = t0;
        while p < s.len() && !matches!(b[p], b'[' | b']' | b'|' | b'#') {
            p += 1;
        }
        if p == t0 {
            return None; // the target needs at least one char
        }
        let tend = p; // the key is the target only (not #frag/|label)
        if p < s.len() && b[p] == b'#' {
            p += 1;
            while p < s.len() && !matches!(b[p], b'[' | b']' | b'|') {
                p += 1;
            }
        }
        if p < s.len() && b[p] == b'|' {
            p += 1;
            while p < s.len() && !matches!(b[p], b'[' | b']') {
                p += 1;
            }
        }
        if !(p + 1 < s.len() && b[p] == b']' && b[p + 1] == b']') {
            return None;
        }
        hn_close(s, p + 2, &s[t0..tend]).map(|end| (&s[i..p + 2], end))
    } else {
        // Bare text: first char neither bracket nor whitespace, then the
        // mention grows lazily while its next char is not a bracket.
        let c0 = s[i..].chars().next()?;
        if c0.is_whitespace() || c0 == '[' || c0 == ']' {
            return None;
        }
        let mut n = c0.len_utf8();
        loop {
            if let Some(end) = hn_close(s, i + n, &s[i..i + n]) {
                return Some((&s[i..i + n], end));
            }
            let next = s[i + n..].chars().next()?;
            if next == '[' || next == ']' {
                return None;
            }
            n += next.len_utf8();
        }
    }
}

/// Keep a wikilink mention verbatim; absorb a bare-text mention into a
/// [[title]] link.  Hand-rolled port of the old hatnote-mention pattern
/// `(?<!\S)(\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]|([^\[\]\s][^\[\]]*?))[\s.,;:!?]*\[\[(?:\2|\3)\]\]`
/// with Python's `re.sub` scan semantics: try each position left to
/// right — only ones after whitespace (or at the very start) can match —
/// replace on a full match and continue after it, otherwise advance one
/// character.
fn absorb_hatnote_mentions(txt: &str) -> String {
    let mut out = String::with_capacity(txt.len());
    let mut last = 0usize;
    let mut i = 0usize;
    while i < txt.len() {
        if i == 0 || prev_char(txt, i).is_some_and(|c| c.is_whitespace()) {
            if let Some((mention, end)) = hn_mention_at(txt, i) {
                out.push_str(&txt[last..i]);
                if mention.starts_with("[[") {
                    out.push_str(mention);
                } else {
                    out.push_str("[[");
                    out.push_str(mention);
                    out.push(']');
                    out.push(']');
                }
                last = end;
                i = end;
                continue;
            }
        }
        i += txt[i..].chars().next().unwrap().len_utf8();
    }
    out.push_str(&txt[last..]);
    out
}

/// `*[[Target]]*` when the label equals the target, else `*[[Target|label]]*`.
fn emph_wikilink(mark: &str, target: &str, core: &str) -> String {
    if core == target {
        format!("{}[[{}]]{}", mark, target, mark)
    } else {
        format!("{}[[{}|{}]]{}", mark, target, core, mark)
    }
}

/// `[[target]]` when the label equals the target page title (query
/// string included, since split_href keeps it), else `[[target|label]]`.
fn labeled_wikilink(target: &str, path: &str, label: &str) -> String {
    let bare = path.replace('_', " ");
    let label_cmp = label.replace('_', " ");
    if label_cmp == bare || label_cmp == target {
        format!("[[{}]]", target)
    } else {
        format!("[[{}|{}]]", target, label)
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

/// <br> rendering: a space, the literal `<br>`, or a newline.
#[derive(Clone, Copy, PartialEq, Default)]
enum BrMode {
    #[default]
    Space,
    Keep,
    Nl,
}

#[derive(Clone, Copy, Default)]
struct InlineCtx {
    no_escape: bool,
    in_link: bool,
    br_mode: BrMode,
    in_hatnote: bool,
    /// `[[Category:…]]` links are the subject of the page being rendered and
    /// survive the dropped-namespace rule: in a category page's member
    /// section, and — when categories are included — in ordinary article
    /// text (a hatnote's "See also: Category:…" link must survive). False
    /// for tables, infoboxes and headings, where the flag is never threaded.
    keep_category_links: bool,
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
        if !piece.is_empty() {
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

/// The link markdown around an anchor's already-rendered label: external
/// `[label](url)`/`<url>`, wikilink `[[Target]]`/`[[Target|label]]`, the
/// bare label for fragment-only hrefs, or '' (dropped media/namespace
/// link). Shared by render_anchor (which adds the whole-label emphasis
/// handling) and the block-level `a` arm of block_md.
fn anchor_md(
    el: NodeRef,
    href: &str,
    label: &str,
    emph_mark: Option<&str>,
    in_link: bool,
    keep_category_links: bool,
) -> String {
    let label_stripped = label.trim();
    let scheme = href_scheme(href);
    if el.has_class("external") || el.attr("rel") == Some("nofollow") || scheme.is_some() {
        if scheme.as_deref() == Some("geo") {
            // coordinate microformat links: the label carries the same
            // data as the URL; render coordinates as plain text
            return label.to_string();
        }
        if label_stripped.is_empty() || label_stripped == href {
            return format!("<{}>", href);
        }
        let lead = &label[..label.len() - label.trim_start().len()];
        let trail = &label[label.trim_end().len()..];
        return format!("{}[{}]({}){}", lead, label_stripped, md_link_url(href), trail);
    }

    // internal link (or interwiki / fragment-only); the citation-residue
    // check reads the raw href — a '#…' href has an empty path by
    // construction, so `#cite_note…` is detected there and not via frag.
    let (path, frag) = split_href(href);
    if path.is_empty() {
        if href.starts_with("#cite_note") {
            return String::new(); // citation residue: drop entirely
        }
        return label.to_string(); // fragment-only: keep label text
    }
    let target = match link_target_parts(&path, &frag) {
        Some(t) if !t.is_empty() => t,
        _ => return label.to_string(),
    };
    if in_dropped_ns(&target) && !(keep_category_links && is_category_target(&target)) {
        return String::new(); // [[File:…]]/[[Category:…]]-style link: dropped whole
    }
    if let Some(mark) = emph_mark {
        return emph_wikilink(mark, &target, label_stripped);
    }
    if !in_link {
        if let Some((mark, core)) = emph_is_whole(label_stripped) {
            return emph_wikilink(mark, &target, core);
        }
    }
    labeled_wikilink(&target, &path, label_stripped)
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

    anchor_md(el, href, &label, emph_mark, ctx.in_link, ctx.keep_category_links)
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
    list_item_lines_ctx(list_el, depth, raw, false)
}

/// [`list_item_lines`] with the body's keep-category-links flag (see
/// [`InlineCtx`]).
pub(crate) fn list_item_lines_ctx(
    list_el: NodeRef,
    depth: usize,
    raw: bool,
    keep_category_links: bool,
) -> Vec<String> {
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
                    parts.push(render_inline(
                        ch,
                        InlineCtx {
                            no_escape: raw,
                            br_mode,
                            keep_category_links,
                            ..Default::default()
                        },
                    ));
                }
                _ => {}
            }
        }
        let text = collapse_ws(&parts.join("")).trim().to_string();
        if !text.is_empty() {
            out.push(format!("{}{}{}", indent, marker, text));
        }
        for sub in subs {
            out.extend(list_item_lines_ctx(sub, depth + 1, raw, keep_category_links));
        }
    }
    out
}

/// Render <ul>/<ol> with 2-space indent per nesting level.
pub(crate) fn render_list(el: NodeRef, depth: usize) -> String {
    render_list_ctx(el, depth, false)
}

/// [`render_list`] with the body's keep-category-links flag (see
/// [`InlineCtx`]).
pub(crate) fn render_list_ctx(el: NodeRef, depth: usize, keep_category_links: bool) -> String {
    list_item_lines_ctx(el, depth, false, keep_category_links).join("\n")
}

/// [`inline_text`] carrying the enclosing render's keep-category-links flag.
fn inline_text_keep(el: NodeRef, keep_category_links: bool) -> String {
    render_inline(el, InlineCtx { keep_category_links, ..Default::default() })
}

/// Definition list: <dt> -> `- **term**`, <dt>+<dd> -> `- **term**: def`,
/// lone <dd> -> `- def`; 2-space indent per nesting level.  A <dd>'s
/// definition text is its inline content with any direct <ul>/<ol>
/// children excluded (they render as sub-list lines instead of leaking,
/// flattened, into the definition); it is attached to the open term's
/// line — or emitted as a lone `- ` item — *before* the sub-lists, so
/// definition and sub-list stay in document order.
pub(crate) fn render_dl(el: NodeRef, depth: usize) -> String {
    render_dl_ctx(el, depth, false)
}

/// [`render_dl`] with the body's keep-category-links flag (see
/// [`InlineCtx`]).
pub(crate) fn render_dl_ctx(el: NodeRef, depth: usize, keep_category_links: bool) -> String {
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
                let term = inline_text_keep(ch, keep_category_links).trim().to_string();
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
                        NodeKind::Element { .. } => parts.push(inline_text_keep(sub, keep_category_links)),
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
                        let sub_md = render_list_ctx(sub, depth + 1, keep_category_links);
                        if !sub_md.is_empty() {
                            out.push(sub_md);
                        }
                    }
                }
            }
            Some("ul") | Some("ol") => {
                let sub_md = render_list_ctx(ch, depth + 1, keep_category_links);
                if !sub_md.is_empty() {
                    out.push(sub_md);
                }
            }
            Some("dl") => {
                let sub_md = render_dl_ctx(ch, depth + 1, keep_category_links);
                if !sub_md.is_empty() {
                    out.push(sub_md);
                }
            }
            _ => {}
        }
    }
    out.join("\n")
}

/// Render one <p>: <br> splits the paragraph (the <br>'s tail opens the
/// next segment — the next quote line in a blockquote), everything else
/// is inline. `keep_category_links` threads the enclosing render's flag
/// into the paragraph's inline context.
fn render_paragraph(p: NodeRef, in_blockquote: bool, keep_category_links: bool) -> Vec<String> {
    let ctx = InlineCtx {
        br_mode: if in_blockquote { BrMode::Nl } else { BrMode::Space },
        keep_category_links,
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
        let s = collapse_space_tab(s).trim().to_string();
        if !s.is_empty() {
            out.push(s);
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
fn render_blockquote(bq: NodeRef, keep_category_links: bool) -> String {
    let inner = block_children_md(bq, None, true, keep_category_links);
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
fn render_block_container(
    el: NodeRef,
    in_blockquote: bool,
    keep_category_links: bool,
) -> String {
    if el.attr("role") == Some("note") || el.has_class("hatnote") {
        return render_hatnote(el, keep_category_links);
    }
    block_children_md(el, None, in_blockquote, keep_category_links)
}

/// Render a hatnote container as a standalone fully-italic paragraph; only
/// [[...]] links survive (emphasis flattened) and anchors pointing at a
/// wiki edit/admin URL become [[title]] wikilinks. `keep_category_links`
/// threads the enclosing render's flag: with categories included, a
/// hatnote's "See also: Category:…" link survives the dropped-namespace
/// rule like any other article-text category link.
fn render_hatnote(el: NodeRef, keep_category_links: bool) -> String {
    let txt = collapse_ws(&render_inline(
        el,
        InlineCtx {
            in_link: true,
            in_hatnote: true,
            keep_category_links,
            ..Default::default()
        },
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
    block_md_in(ch, in_blockquote, false)
}

/// [`block_md`] with the body's keep-category-links flag (see
/// [`InlineCtx`]).
pub(crate) fn block_md_in(ch: NodeRef, in_blockquote: bool, keep_category_links: bool) -> String {
    // a bare hN, or a div.mw-heading wrapper, renders as a Markdown heading;
    // a heading with no text (category pages' TOC groups carry
    // `<h3>&nbsp;</h3>` for the non-letter keys) renders nothing instead of
    // a bare `##` line.
    if let Some((level, text)) = heading_of(ch) {
        return if text.is_empty() {
            String::new()
        } else {
            format!("{} {}", "#".repeat(level as usize), text)
        };
    }
    match ch.tag().unwrap_or("") {
        "p" => {
            let p_lines = render_paragraph(ch, in_blockquote, keep_category_links);
            if in_blockquote {
                p_lines.join("\n")
            } else {
                p_lines.join("\n\n")
            }
        }
        "ul" | "ol" => render_list_ctx(ch, 0, keep_category_links),
        "dl" => render_dl_ctx(ch, 0, keep_category_links),
        "blockquote" => render_blockquote(ch, keep_category_links),
        "pre" => render_pre(ch),
        "table" => tables::render_table(ch),
        "span" => collapse_ws(&inline_text_keep(ch, keep_category_links)).trim().to_string(),
        // Block-level anchor: inline anchor semantics (via anchor_md) with
        // the content rendered as inline text, the span arm's mechanism —
        // block containers walk element children only, so a bare <a> (a
        // mwoffliner meta-refresh redirect stub) used to render empty and
        // leave its page title-only.  Deliberate divergence from
        // zim2zim.py, which still emits these pages title-only.
        "a" => {
            let href = ch.attr("href").unwrap_or("");
            if is_media_href(href) {
                return String::new();
            }
            let label = collapse_ws(&render_children(
                ch,
                InlineCtx { no_escape: true, in_link: true, ..Default::default() },
            ))
            .trim()
            .to_string();
            anchor_md(ch, href, &label, None, false, keep_category_links)
        }
        _ => render_block_container(ch, in_blockquote, keep_category_links),
    }
}

/// Render an element's children as block Markdown (blocks joined with
/// blank lines). A non-empty `key_facts` block is emitted structurally
/// immediately before the first heading (or at the end for lead-only
/// pages). `in_blockquote` keeps <br>-separated paragraph lines within one
/// quote paragraph. `keep_category_links` is the body's keep-category-links
/// flag (see [`InlineCtx`]).
pub(crate) fn block_children_md(
    el: NodeRef,
    key_facts: Option<&str>,
    in_blockquote: bool,
    keep_category_links: bool,
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
        let md = block_md_in(ch, in_blockquote, keep_category_links);
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

/// First descendant element carrying the given id attribute.
fn find_id<'a>(el: NodeRef<'a>, id: &str) -> Option<NodeRef<'a>> {
    el.descendants()
        .into_iter()
        .find(|d| d.is_element() && d.attr("id") == Some(id))
}

/// True for a `[[Category:…]]` target (leading colons tolerated, like
/// `in_dropped_ns`'s head match).
fn is_category_target(target: &str) -> bool {
    target
        .trim_start_matches(':')
        .split_once(':')
        .is_some_and(|(head, _)| head.to_lowercase() == "category")
}

/// The category page's member/subcategory section: div.mw-category-generated,
/// a sibling of div.mw-parser-output (both under #mw-content-text) and so
/// OUTSIDE the render root. Real category pages carry it whenever they list
/// subcategories or members (empty ones, which mwoffliner does not ship, have
/// no lists and no section); located by the class token, present on every
/// member-listing category page in the archive.
fn category_generated_id(dom: &Dom) -> Option<crate::htmldom::NodeId> {
    dom.root()
        .descendants()
        .into_iter()
        .find(|el| el.is_element() && el.has_class("mw-category-generated"))
        .map(|el| el.id())
}

/// The page's normal categories as a '## Categories' section ('' when the
/// page has none or carries no catlinks bar): one wikilink bullet per
/// anchor of #mw-normal-catlinks (hidden categories stay excluded), in
/// page order. The section is emitted only for at least one surviving
/// category; anchors render_anchor drops (media hrefs, dropped-namespace
/// targets) are skipped — except the category targets themselves, the
/// section's subject. The heading comes from wikil10n::categories_title.
fn categories_section(root: NodeRef, lang: Option<&str>) -> String {
    let Some(bar) = find_id(root, "catlinks") else {
        return String::new();
    };
    let Some(normal) = find_id(bar, "mw-normal-catlinks") else {
        return String::new();
    };
    let mut bullets: Vec<String> = Vec::new();
    for li in normal.find_all("li") {
        for a in li.find_all("a") {
            let href = a.attr("href").unwrap_or("");
            if is_media_href(href) {
                continue;
            }
            let (path, frag) = split_href(href);
            let Some(target) = link_target_parts(&path, &frag).filter(|t| !t.is_empty()) else {
                continue;
            };
            if in_dropped_ns(&target) && !is_category_target(&target) {
                continue;
            }
            // Like render_anchor's label: verbatim text, flattened markup.
            let label = render_children(
                a,
                InlineCtx { no_escape: true, in_link: true, ..Default::default() },
            )
            .trim()
            .to_string();
            if label.is_empty() {
                continue; // no text to label the category with
            }
            bullets.push(format!("- {}", labeled_wikilink(&target, &path, &label)));
        }
    }
    if bullets.is_empty() {
        return String::new();
    }
    format!("## {}\n\n{}", crate::wikil10n::categories_title(lang), bullets.join("\n"))
}

/// Render the article body around the already-located container: Parsoid
/// sections flattened, blocks rendered, sections assembled, title line
/// prepended, the category page's member section (when rendering) and the
/// categories section appended, and the cleanup pass applied. The body's
/// inline contexts carry `keep_category_links` (true for ordinary articles
/// when categories are included — a hatnote's "See also: Category:…"
/// link survives; always true for the category page's member section).
fn render_article(
    mut dom: Dom,
    body: crate::htmldom::NodeId,
    key_facts: String,
    category_generated: Option<crate::htmldom::NodeId>,
    categories: String,
    keep_category_links: bool,
    title: Option<&str>,
    lang: Option<&str>,
) -> String {
    flatten_parsoid_sections(&mut dom, body);
    let body_md = block_children_md(
        dom.ref_(body),
        if key_facts.is_empty() { None } else { Some(&key_facts) },
        false,
        keep_category_links,
    );
    let content = cleanup::assemble(&body_md, lang);
    // The category page's member/subcategory section renders through the
    // same block walker as the body: its own <h2>Subcategories</h2> /
    // <h2>Pages in category "…"</h2> become '## ' headings and its
    // mw-category-group <ul><li><a> lists become the normal wikilink
    // bullets (member hrefs are bare paths, subcategory ones
    // ./Category:… — split_href handles both). DOM order also puts it
    // before the catlinks bar, so the generated section is appended before
    // the '## Categories' section below.
    // The member section always forces the flag true: its Category links
    // are the section's subject regardless of the body's mode.
    let generated = category_generated
        .map(|id| block_children_md(dom.ref_(id), None, false, true))
        .unwrap_or_default();
    let title = title.map(|t| t.to_string()).or_else(|| first_h1_title(&dom));
    let mut parts: Vec<String> = Vec::new();
    if let Some(t) = &title {
        parts.push(format!("# {}", t.trim()));
    }
    if !content.trim().is_empty() {
        parts.push(content);
    }
    if !generated.trim().is_empty() {
        parts.push(generated);
    }
    if !categories.is_empty() {
        parts.push(categories);
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
/// boilerplate sections. `include_categories` opt-in appends a localized
/// '## Categories' section listing the page's normal categories after the
/// body content, and — for ordinary article text — lets `[[Category:…]]`
/// links through the dropped-namespace rule (a hatnote's "See also:
/// Category:…" link survives; without categories such links still drop,
/// their target being absent from the output). It also renders category
/// pages usefully: their member and
/// subcategory lists live in a sibling div.mw-category-generated outside
/// the render root and are dropped with the flag off (the page renders
/// title/description-only, like zim2zim.py); with the flag on that section
/// renders through the normal block walker — its own '## Subcategories' and
/// '## Pages in category "…"' headings and wikilink lists — placed before
/// the '## Categories' section, in DOM order.
pub fn html_to_md(
    html_str: &str,
    title: Option<&str>,
    lang: Option<&str>,
    include_categories: bool,
) -> String {
    let mut dom = Dom::parse(html_str);
    let key_facts = key_facts_of(&dom, lang);
    let categories =
        if include_categories { categories_section(dom.root(), lang) } else { String::new() };
    // The category page's member section: its node id must be located
    // before `dom` moves into render_article (only rendered when the flag
    // is on — the page's lists live outside the render root).
    let category_generated = if include_categories { category_generated_id(&dom) } else { None };
    // The wiki article container, else the page's <body>.
    let wiki_body = get_parser_output(dom.root()).map(|b| b.id());
    let body =
        wiki_body.unwrap_or_else(|| dom.root().find("body").map_or(dom.root().id(), |b| b.id()));
    let mut title = title.map(str::to_string);
    if wiki_body.is_none() {
        // On non-wiki pages the first in-body <h1> supplies the title when
        // none was passed, and an <h1> equal to the output title is
        // dropped: the `# Title` line would otherwise say it twice.
        let h1 = dom.ref_(body).find("h1").map(|h| {
            let text = collapse_ws(&h.text_content()).trim().to_string();
            (h.id(), text)
        });
        if let Some((id, text)) = h1.filter(|(_, t)| !t.is_empty()) {
            if title.is_none() {
                title = Some(text.clone());
            }
            if title.as_deref() == Some(text.as_str()) {
                dom.detach(id);
            }
        }
    }
    render_article(
        dom,
        body,
        key_facts,
        category_generated,
        categories,
        include_categories,
        title.as_deref(),
        lang,
    )
}

#[cfg(test)]
mod tests {
    use super::{absorb_hatnote_mentions, html_to_md};

    /// A bare-text mention of the linked article is absorbed into the
    /// wikilink ("…from X , [[X]]" → "…from [[X]]").
    #[test]
    fn hatnote_bare_mention_is_absorbed() {
        assert_eq!(
            absorb_hatnote_mentions("This section is an excerpt from X , [[X]]."),
            "This section is an excerpt from [[X]]."
        );
    }

    /// A wikilink mention (labelled or not) survives: the duplicate
    /// closing mention is dropped, the first kept verbatim.
    #[test]
    fn hatnote_wikilink_mention_survives() {
        assert_eq!(
            absorb_hatnote_mentions("This section is an excerpt from [[X|label]] , [[X]]."),
            "This section is an excerpt from [[X|label]]."
        );
        assert_eq!(
            absorb_hatnote_mentions("For other uses, see [[X]] , [[X]] ."),
            "For other uses, see [[X]] ."
        );
    }

    /// Mentions whose keys differ never match, and an interior match must
    /// not drop the gap it spans: both shapes stay as they are.
    #[test]
    fn hatnote_non_matches_stay() {
        assert_eq!(absorb_hatnote_mentions("ab , [[b]]"), "ab , [[b]]");
        assert_eq!(absorb_hatnote_mentions("[[a]] , [[b]]"), "[[a]] , [[b]]");
        assert_eq!(
            absorb_hatnote_mentions("[[a]] , [[b]] [[b]]"),
            "[[a]] , [[b]]"
        );
    }

    #[test]
    fn comment_tail_survives() {
        let md = html_to_md(
            "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\"><p>in 1889.<!--note--> It was made from <a href=\"./Nitrocellulose\">nitrocellulose</a> known as nitrate.</p></div></div></body></html>",
            Some("Nitro"),
            None,
            false,
        );
        assert!(md.contains("1889. It was made"), "{md}");
    }

    #[test]
    fn dropped_namespace_links_vanish() {
        let md = html_to_md(
            "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\"><p>Text <a href=\"Category%3AFoo\"Category:Foo\">label</a><a href=\"./File%3ABar\">img</a>.</p></div></div></body></html>",
            Some("T"),
            None,
            false,
        );
        assert_eq!(md, "# T\n\nText.\n");
    }

    #[test]
    fn br_splits_the_paragraph_without_losing_text() {
        let md = html_to_md(
            "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\"><p>a<br>b <i>c</i></p></div></div></body></html>",
            Some("T"),
            None,
            false,
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
            false,
        );
        assert_eq!(md, "# Head\n\nPara **bold**.\n\nSee [link](https://x.example/a?b=1&c=2).\n");
    }

    #[test]
    fn in_body_h1_differs_from_the_passed_title_stays() {
        let md = html_to_md(
            "<html><body><h1>Intro</h1><p>Body text.</p></body></html>",
            Some("Doc"),
            None,
            false,
        );
        assert_eq!(md, "# Doc\n\n## Intro\n\nBody text.\n");
    }

    /// Wiki article shell: `inner` inside div.mw-parser-output.
    fn wiki_doc(inner: &str) -> String {
        html_to_md(
            &format!(
                "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\">{}</div></div></body></html>",
                inner
            ),
            Some("T"),
            None,
            false,
        )
    }

    /// A query string stays part of the internal link target: stripping it
    /// would collapse `Foo?x=1` and `Foo?x=2` onto the same `Foo` target.
    #[test]
    fn internal_link_keeps_its_query_string() {
        let md = wiki_doc("<p>See <a href=\"./Foo?x=1\">L</a> here.</p>");
        assert_eq!(md, "# T\n\nSee [[Foo?x=1|L]] here.\n");
    }

    /// Query and fragment combine in the target, in href order.
    #[test]
    fn internal_link_keeps_query_and_fragment() {
        let md = wiki_doc("<p>See <a href=\"./Foo?x=1#Sec\">L</a> here.</p>");
        assert_eq!(md, "# T\n\nSee [[Foo?x=1#Sec|L]] here.\n");
    }

    /// A query-free internal link renders exactly as before.
    #[test]
    fn internal_link_without_query_is_unchanged() {
        let md = wiki_doc("<p>See <a href=\"./Foo\">L</a> here.</p>");
        assert_eq!(md, "# T\n\nSee [[Foo|L]] here.\n");
    }

    /// A <dd> with both inline text and a sub-list: the definition goes on
    /// the term line, the sub-list renders after it — the sub-list's items
    /// must not also leak (flattened) into the definition text.
    #[test]
    fn dd_text_and_sublist_define_on_the_term_line_then_the_sublist() {
        let md = wiki_doc("<dl><dt>T</dt><dd>Def <i>x</i><ul><li>s1</li><li>s2</li></ul></dd></dl>");
        assert_eq!(md, "# T\n\n- **T**: Def *x*\n  - s1\n  - s2\n");
    }

    /// Lone <dd> with text and a sub-list: its own `- ` line, then the
    /// sub-list — no duplicated flattened text.
    #[test]
    fn lone_dd_with_text_and_sublist_puts_the_text_on_its_own_line() {
        let md = wiki_doc("<dl><dd>Def <i>x</i><ul><li>s1</li></ul></dd></dl>");
        assert_eq!(md, "# T\n\n- Def *x*\n  - s1\n");
    }

    /// A <dd> holding only a sub-list adds no definition: the term line is
    /// untouched and the sub-list follows it.
    #[test]
    fn dd_with_only_a_sublist_leaves_the_term_line_untouched() {
        let md = wiki_doc("<dl><dt>T</dt><dd><ul><li>s1</li><li>s2</li></ul></dd></dl>");
        assert_eq!(md, "# T\n\n- **T**\n  - s1\n  - s2\n");
    }

    /// A complex table (spans force the HTML-table path) keeps its
    /// caption's <br> verbatim, consistent with its <br>-keeping cells;
    /// the pre-fix port degraded it to a space (the Python original's
    /// dead keep_br flag).
    #[test]
    fn html_table_caption_keeps_br() {
        let md = wiki_doc(
            "<table><caption>Statistiques 1991-2020 (à 10 km)<br>Records établis depuis 1888</caption><tr><td rowspan=\"2\">a</td><td>b</td></tr><tr><td>c</td></tr></table>",
        );
        assert!(
            md.contains(
                "<caption>Statistiques 1991-2020 (à 10 km)<br>Records établis depuis 1888</caption>"
            ),
            "{md}"
        );
    }

    /// A simple table renders as a pipe table: the caption must stay
    /// single-line, so its <br> degrades to a space.
    #[test]
    fn pipe_table_caption_joins_br_lines_with_a_space() {
        let md = wiki_doc(
            "<table><caption>Line one<br>Line two</caption><tr><th>H</th></tr><tr><td>a</td></tr></table>",
        );
        assert_eq!(
            md,
            "# T\n\n*Line one Line two*\n\n| H |\n| --- |\n| a |\n"
        );
    }

    /// Small tables are real content: a lone non-empty cell renders as a
    /// pipe table (no stub-dropping), while a table whose every cell is
    /// empty — or a bare <br>, the HTML-path's image residue — drops
    /// entirely, and blank rows never render next to real ones.
    #[test]
    fn lone_cell_table_renders_and_an_all_empty_table_drops() {
        let md = wiki_doc("<table><tr><td>solo</td></tr></table>");
        assert_eq!(md, "# T\n\n| solo |\n| --- |\n");
        let md = wiki_doc("<table><tr><td>solo</td></tr><tr><td></td><td> </td></tr></table>");
        assert_eq!(md, "# T\n\n| solo |\n| --- |\n");
        let md = wiki_doc("<table><tr><td></td></tr><tr><td>   </td></tr></table>");
        assert_eq!(md, "# T\n");
        let md = wiki_doc("<table><tr><td rowspan=\"2\"><br></td><td></td></tr></table>");
        assert_eq!(md, "# T\n");
    }

    /// A list nested one element deeper in a cell (inside a <span>) still
    /// forces the HTML-table path and renders as `* item` lines: it must
    /// not be flattened to space-separated text in a pipe table.
    #[test]
    fn list_nested_in_a_span_makes_the_table_complex() {
        let md = wiki_doc(
            "<table><tr><th>H</th></tr><tr><td><span><ul><li>a</li><li>b</li></ul></span></td></tr></table>",
        );
        assert!(md.contains("<table>"), "{md}");
        assert!(md.contains("<td>* a\n* b</td>"), "{md}");
        assert!(!md.contains("a b"), "{md}");
    }

    /// A direct <ul> in a cell renders exactly as before (regression
    /// guard), and a list-free sibling cell keeps its <br> verbatim.
    #[test]
    fn direct_ul_cell_renders_as_before() {
        let md = wiki_doc(
            "<table><tr><th>H</th></tr><tr><td><ul><li>a</li><li>b</li></ul></td><td><span>one<br>two</span></td></tr></table>",
        );
        assert!(md.contains("<td>* a\n* b</td>"), "{md}");
        assert!(md.contains("<td>one<br>two</td>"), "{md}");
    }

    /// A <div> wrapping text and a list keeps the text as prose and the
    /// list as `* item` lines in document order; a wrapped lone list item
    /// still degrades to plain text.
    #[test]
    fn div_with_text_and_list_keeps_prose_and_list_lines() {
        let md = wiki_doc(
            "<table><tr><th>H</th></tr><tr><td><div>text<ul><li>x</li></ul></div></td></tr></table>",
        );
        assert!(md.contains("<td>text\n* x</td>"), "{md}");
        let md = wiki_doc(
            "<table><tr><th>H</th></tr><tr><td><div><ul><li>solo</li></ul></div></td></tr></table>",
        );
        assert!(md.contains("<td>solo</td>"), "{md}");
    }

    // Block-level anchors mirror inline anchor semantics; a deliberate
    // divergence from zim2zim.py, which leaves bare <a> pages title-only.

    /// A bare block-level <a> — mwoffliner's meta-refresh redirect stub —
    /// renders as a wikilink under the title line instead of leaving the
    /// page title-only (block containers walk element children only, so
    /// the stub's bare-text label used to vanish).
    #[test]
    fn block_anchor_stub_page_renders_as_a_wikilink() {
        let md = html_to_md(
            "<html><body><a href=\"./Equals_sign#Not_equal\">!=</a></body></html>",
            Some("!="),
            None,
            false,
        );
        assert_eq!(md, "# !=\n\n[[Equals sign#Not equal|!=]]\n");
    }

    /// A block-level anchor whose label equals its target renders the
    /// bare `[[Target]]` form, like its inline counterpart.
    #[test]
    fn block_anchor_with_equal_label_renders_the_bare_wikilink() {
        let md = wiki_doc("<a href=\"./Target\">Target</a>");
        assert_eq!(md, "# T\n\n[[Target]]\n");
    }

    /// A block-level external anchor renders `[label](url)`; without a
    /// label it degrades to `<url>`, like its inline counterpart.
    #[test]
    fn block_external_anchor_renders_a_markdown_link() {
        let md = wiki_doc("<a class=\"external\" href=\"https://x.example/a\">Site</a>");
        assert_eq!(md, "# T\n\n[Site](https://x.example/a)\n");
        let md = wiki_doc("<a class=\"external\" href=\"https://x.example/a\"></a>");
        assert_eq!(md, "# T\n\n<https://x.example/a>\n");
    }

    /// A block-level anchor wrapping a <span> renders the span's inline
    /// content as the link label (asterisks stay verbatim, like inline).
    #[test]
    fn block_anchor_wrapping_a_span_uses_it_as_the_label() {
        let md = wiki_doc("<a href=\"./Foo\"><span>la *bel*</span></a>");
        assert_eq!(md, "# T\n\n[[Foo|la *bel*]]\n");
    }

    /// Fragment-only and href-less block anchors render the bare label.
    #[test]
    fn block_fragment_only_and_missing_href_render_the_label() {
        let md = wiki_doc("<a href=\"#Sec\">label</a>");
        assert_eq!(md, "# T\n\nlabel\n");
        let md = wiki_doc("<a>label</a>");
        assert_eq!(md, "# T\n\nlabel\n");
    }

    /// Media hrefs and File:/Category:-style targets drop the whole block
    /// anchor, exactly like inline anchors.
    #[test]
    fn block_media_and_file_hrefs_drop_the_anchor() {
        let md = wiki_doc("<a href=\"./File:Foo.jpg\">img</a>");
        assert_eq!(md, "# T\n");
        let md = wiki_doc("<a href=\"./Song.ogg\">song</a>");
        assert_eq!(md, "# T\n");
        let md = wiki_doc("<a href=\"https://upload.wikimedia.org/x/song.ogg\">song</a>");
        assert_eq!(md, "# T\n");
    }

    /// Slashes pass through the wikilink target and underscores fold to
    /// spaces on both sides: `./Page/Sub_page` labelled `Page/Sub_page`
    /// renders `[[Page/Sub page]]`.
    #[test]
    fn block_anchor_target_keeps_slashes_and_folds_underscores() {
        let md = wiki_doc("<a href=\"./Page/Sub_page\">Page/Sub_page</a>");
        assert_eq!(md, "# T\n\n[[Page/Sub page]]\n");
        let md = wiki_doc("<a href=\"./Page/Sub_page\">Sub page</a>");
        assert_eq!(md, "# T\n\n[[Page/Sub page|Sub page]]\n");
    }

    // The opt-in '## Categories' section (html_to_md's include_categories).

    /// The catlinks bar as real articles carry it: normal categories as
    /// percent-encoded anchors inside #mw-normal-catlinks, hidden ones
    /// beside it, the whole bar outside div.mw-parser-output.
    const CAT_BAR: &str = concat!(
        "<div id=\"catlinks\" class=\"catlinks\">",
        "<div id=\"mw-normal-catlinks\" class=\"mw-normal-catlinks\">Categories: <ul>",
        "<li><a href=\"Category%3A2003_albums\" title=\"Category:2003 albums\">2003 albums</a></li>",
        "<li><a href=\"Category%3AEvan_Parker_albums\" title=\"Category:Evan Parker albums\">Evan Parker albums</a></li>",
        "</ul></div>",
        "<div id=\"mw-hidden-catlinks\" class=\"mw-hidden-catlinks\">Hidden categories: <ul>",
        "<li>Use mdy dates from August 2026</li></ul></div></div>"
    );

    /// Wiki article shell whose content div is followed by `tail` (real
    /// articles carry the catlinks bar there, outside the render root).
    fn wiki_doc_with_tail(
        inner: &str,
        tail: &str,
        lang: Option<&str>,
        include_categories: bool,
    ) -> String {
        html_to_md(
            &format!(
                "<html><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\">{}</div></div>{}</body></html>",
                inner, tail
            ),
            Some("T"),
            lang,
            include_categories,
        )
    }

    /// Flag on: the bar's normal categories become a '## Categories'
    /// section of wikilink bullets after the body; the hidden categories
    /// stay excluded.
    #[test]
    fn categories_section_appended_when_enabled() {
        let md = wiki_doc_with_tail("<p>Body.</p>", CAT_BAR, None, true);
        assert_eq!(
            md,
            "# T\n\nBody.\n\n## Categories\n\n\
             - [[Category:2003 albums|2003 albums]]\n\
             - [[Category:Evan Parker albums|Evan Parker albums]]\n"
        );
    }

    /// Flag off: byte-identical to the pre-flag output — the bar stays
    /// dropped entirely.
    #[test]
    fn categories_flag_off_is_byte_identical() {
        let md = wiki_doc_with_tail("<p>Body.</p>", CAT_BAR, None, false);
        assert_eq!(md, "# T\n\nBody.\n");
    }

    /// Hidden-only bar, or a present but empty normal list: no section —
    /// the >=1-category gate and the #mw-normal-catlinks scope both hold.
    #[test]
    fn without_normal_categories_no_section() {
        let hidden_only = concat!(
            "<div id=\"catlinks\" class=\"catlinks\">",
            "<div id=\"mw-hidden-catlinks\" class=\"mw-hidden-catlinks\">Hidden categories: <ul>",
            "<li><a href=\"Category%3AUse_mdy_dates\" title=\"Category:Use mdy dates\">Use mdy dates</a></li>",
            "</ul></div></div>"
        );
        let md = wiki_doc_with_tail("<p>Body.</p>", hidden_only, None, true);
        assert_eq!(md, "# T\n\nBody.\n");
        let empty_normal = concat!(
            "<div id=\"catlinks\" class=\"catlinks\">",
            "<div id=\"mw-normal-catlinks\" class=\"mw-normal-catlinks\">Categories: <ul></ul></div></div>"
        );
        let md = wiki_doc_with_tail("<p>Body.</p>", empty_normal, None, true);
        assert_eq!(md, "# T\n\nBody.\n");
    }

    /// No catlinks bar at all (arbitrary non-wiki HTML): no section, flag
    /// on or off.
    #[test]
    fn without_catlinks_bar_no_section() {
        let md = wiki_doc_with_tail("<p>Body.</p>", "", None, true);
        assert_eq!(md, "# T\n\nBody.\n");
    }

    // Category links in article text: including the categories also
    // re-allows `[[Category:…]]` targets in the body's inline contexts
    // (hatnotes, paragraphs, block anchors), not just the Categories
    // section — a hatnote's "See also: Category:…" link must survive the
    // dropped-namespace rule when the target page is part of the output.
    // Under --exclude-categories the target is absent, so the link drops
    // again (a dangling wikilink would be worse); File:/Media: drops stay
    // unconditional either way. HTML shapes are the archive's real ones
    // (e.g. "See also: Category:Children of Gaia" — label carries the
    // namespace, href is the percent-encoded colon).

    /// A hatnote linking a category, as the archive carries it.
    const HATNOTE_SEE_ALSO_CAT: &str = concat!(
        "<div role=\"note\" class=\"hatnote navigation-not-searchable\">See also: ",
        "<a rel=\"mw:WikiLink\" href=\"Category%3A1838_deaths\" ",
        "title=\"Category:1838 deaths\">Category:1838 deaths</a></div>"
    );

    /// Flag on (the default): the hatnote keeps its Category link as a
    /// wikilink; the label equals the target, so the bare `[[Target]]`
    /// form renders.
    #[test]
    fn hatnote_category_link_survives_when_categories_included() {
        let md = wiki_doc_with_tail(HATNOTE_SEE_ALSO_CAT, "", None, true);
        assert_eq!(md, "# T\n\n*See also: [[Category:1838 deaths]]*\n");
    }

    /// Flag off (--exclude-categories): the whole anchor drops, leaving
    /// the bare "*See also:*" run.
    #[test]
    fn hatnote_category_link_drops_when_categories_excluded() {
        let md = wiki_doc_with_tail(HATNOTE_SEE_ALSO_CAT, "", None, false);
        assert_eq!(md, "# T\n\n*See also:*\n");
    }

    /// A Category link inline in a paragraph with surrounding text keeps
    /// it when categories are included and drops it (surroundings intact)
    /// when they are excluded.
    #[test]
    fn paragraph_category_link_follows_the_categories_flag() {
        let inner = concat!(
            "<p>Full lists: <a rel=\"mw:WikiLink\" href=\"Category%3A1838_deaths\" ",
            "title=\"Category:1838 deaths\">Category:1838 deaths</a> and its subcategories.</p>"
        );
        let md = wiki_doc_with_tail(inner, "", None, true);
        assert_eq!(
            md,
            "# T\n\nFull lists: [[Category:1838 deaths]] and its subcategories.\n"
        );
        let md = wiki_doc_with_tail(inner, "", None, false);
        assert_eq!(md, "# T\n\nFull lists: and its subcategories.\n");
    }

    /// File: links stay dropped in BOTH modes — only Category targets are
    /// re-allowed.
    #[test]
    fn file_links_still_drop_in_both_modes() {
        let inner = concat!(
            "<p>Depicted <a rel=\"mw:WikiLink\" href=\"File%3AExample.jpg\" ",
            "title=\"File:Example.jpg\">File:Example.jpg</a> above.</p>"
        );
        for flag in [false, true] {
            let md = wiki_doc_with_tail(inner, "", None, flag);
            assert_eq!(md, "# T\n\nDepicted above.\n", "flag={flag}");
        }
    }

    // The category page's member section (div.mw-category-generated, a sibling
    // of div.mw-parser-output outside the render root). Real structures
    // probed in the archive (e.g. Category:2018 in men's international
    // association football, cluster 2; Category:1957 Hindi-language films,
    // cluster 40).

    /// mwoffliner's JS-pagination placeholder pair around a category list
    /// (one div before it, one after).
    fn cat_pagination() -> &'static str {
        "<div class=\"mwo-cat-pagination\"><span class=\"mwo-no-js\">Next items are not visible in browsers without Javascript</span><span class=\"mwo-js\"></span></div>"
    }

    /// The mw-category-generated section as the archive's category pages
    /// carry it: #mw-subcategories and #mw-pages blocks, each with its own
    /// <h2>, count paragraph, pagination noise and a
    /// mw-content-ltr > mw-category > mw-category-group(h3 + ul) list tree,
    /// plus the closing reduced-ZIM note. Subcategory hrefs are
    /// percent-encoded Category%3A… links, member hrefs bare underscored
    /// paths — both carry target="_parent".
    fn cat_generated() -> String {
        let pag = cat_pagination();
        [
            "<div class=\"mw-category-generated\">",
            "<div id=\"mw-subcategories\">",
            "<h2>Subcategories</h2>",
            "<p>This category has the following 2 subcategories, out of 2 total.</p>",
            pag,
            "<div class=\"mw-content-ltr\"><div class=\"mw-category\">",
            // a non-letter group key carries an &nbsp;-only <h3> header
            "<div class=\"mw-category-group\"><h3>&nbsp;</h3><ul>",
            "<li><a target=\"_parent\" href=\"Category%3A1957_films_by_country\" title=\"Category:1957 films by country\">1957 films by country</a></li>",
            "</ul></div>",
            "<div class=\"mw-category-group\"><h3>C</h3><ul>",
            "<li><a target=\"_parent\" href=\"./Category:Comedy_films_of_1957\" title=\"Category:Comedy films of 1957\">Comedy films of 1957</a></li>",
            "<li><a target=\"_parent\" href=\"Category%3ACrime_films_of_1957\" title=\"Category:Crime films of 1957\">Crime films of 1957</a></li>",
            "</ul></div></div></div>",
            pag,
            "</div>",
            "<div id=\"mw-pages\">",
            "<h2>Pages in category \"1957 Hindi-language films\"</h2>",
            "<p>The following 2 pages are in this category, out of 2 total.</p>",
            pag,
            "<div class=\"mw-content-ltr\"><div class=\"mw-category\">",
            "<div class=\"mw-category-group\"><h3>A</h3><ul>",
            "<li><a target=\"_parent\" href=\"Aag_(1957_film)\" title=\"Aag (1957 film)\">Aag (1957 film)</a></li>",
            "</ul></div>",
            "<div class=\"mw-category-group\"><h3>M</h3><ul>",
            "<li><a target=\"_parent\" href=\"Mother_India\" title=\"Mother India\">Mother India</a></li>",
            "</ul></div></div></div>",
            pag,
            "</div>",
            "<p><em>This category content has been reduced to only pages contained in the ZIM file.</em></p>",
            "</div>",
        ]
        .concat()
    }

    /// A two-entry normal-catlinks bar for the fixture's tail.
    fn cat_bar_local() -> String {
        [
            "<div id=\"catlinks\" class=\"catlinks\">",
            "<div id=\"mw-normal-catlinks\" class=\"mw-normal-catlinks\">Categories: <ul>",
            "<li><a href=\"Category%3A1957_films\" title=\"Category:1957 films\">1957 films</a></li>",
            "<li><a href=\"Category%3A1957_films_by_language\" title=\"Category:1957 films by language\">1957 films by language</a></li>",
            "</ul></div></div>",
        ]
        .concat()
    }

    /// Category-page shell: `inner` inside div.mw-parser-output followed by
    /// `tail` (the mw-category-generated section and the catlinks bar sit
    /// there, outside the render root), rendered under the page's real title.
    fn category_doc(
        title: &str,
        inner: &str,
        tail: &str,
        lang: Option<&str>,
        include_categories: bool,
    ) -> String {
        html_to_md(
            &format!(
                "<html><body class=\"ns-14 ns-subject\"><div id=\"mw-content-text\"><div class=\"mw-parser-output\">{}</div>{}</div></body></html>",
                inner, tail
            ),
            Some(title),
            lang,
            include_categories,
        )
    }

    /// (a) Subcategories + members + description, flag on: the section's own
    /// Subcategories / Pages-in-category headings render as '## ' headings
    /// with the count paragraphs and the wikilink bullet lists (the
    /// &nbsp;-keyed TOC group header renders nothing), the reduced-ZIM note
    /// renders as an italic paragraph, and the section precedes the
    /// '## Categories' bar in DOM order.
    #[test]
    fn category_page_with_subcategories_members_and_description() {
        let md = category_doc(
            "Category:1957 Hindi-language films",
            "<p>This category is for <b><a rel=\"mw:WikiLink\" href=\"Hindi_language\" title=\"Hindi language\" class=\"mw-redirect\">Hindi-language</a></b> <b><a rel=\"mw:WikiLink\" href=\"Film\" title=\"Film\">films</a></b>.</p>",
            &format!("{}{}", cat_generated(), cat_bar_local()),
            None,
            true,
        );
        assert_eq!(
            md,
            concat!(
                "# Category:1957 Hindi-language films\n\n",
                "This category is for **[[Hindi language|Hindi-language]]** **[[Film|films]]**.\n\n",
                "## Subcategories\n\n",
                "This category has the following 2 subcategories, out of 2 total.\n\n",
                "- [[Category:1957 films by country|1957 films by country]]\n\n",
                "### C\n\n",
                "- [[Category:Comedy films of 1957|Comedy films of 1957]]\n",
                "- [[Category:Crime films of 1957|Crime films of 1957]]\n\n",
                "## Pages in category \"1957 Hindi-language films\"\n\n",
                "The following 2 pages are in this category, out of 2 total.\n\n",
                "### A\n\n",
                "- [[Aag (1957 film)]]\n\n",
                "### M\n\n",
                "- [[Mother India]]\n\n",
                "*This category content has been reduced to only pages contained in the ZIM file.*\n\n",
                "## Categories\n\n",
                "- [[Category:1957 films|1957 films]]\n",
                "- [[Category:1957 films by language|1957 films by language]]\n"
            )
        );
        // the JS-pagination placeholders never leak into the output
        assert!(!md.contains("Next items"), "{md}");
        assert!(!md.contains("mwo-"), "{md}");
    }

    /// (b) Members only, no description: the parse-output root is empty, so
    /// the pages block renders right under the title.
    #[test]
    fn members_only_category_page() {
        let generated = concat!(
            "<div class=\"mw-category-generated\">",
            "<div id=\"mw-pages\">",
            "<h2>Pages in category \"Cinematographers from Georgia (country)\"</h2>",
            "<p>This category contains only the following page.</p>",
            "<div class=\"mw-content-ltr\"><div class=\"mw-category\">",
            "<div class=\"mw-category-group\"><h3>A</h3><ul>",
            "<li><a target=\"_parent\" href=\"Vasil_Amashukeli\" title=\"Vasil Amashukeli\">Vasil Amashukeli</a></li>",
            "</ul></div></div></div>",
            "</div>",
            "<p><em>This category content has been reduced to only pages contained in the ZIM file.</em></p>",
            "</div>"
        );
        let md = category_doc(
            "Category:Cinematographers from Georgia (country)",
            "",
            generated,
            None,
            true,
        );
        assert_eq!(
            md,
            concat!(
                "# Category:Cinematographers from Georgia (country)\n\n",
                "## Pages in category \"Cinematographers from Georgia (country)\"\n\n",
                "This category contains only the following page.\n\n",
                "### A\n\n",
                "- [[Vasil Amashukeli]]\n\n",
                "*This category content has been reduced to only pages contained in the ZIM file.*\n"
            )
        );
    }

    /// (iii) Subcategories only (no #mw-pages block; shape probed on
    /// Category:Fencing in North America by country, cluster 2): the
    /// section renders under its own heading, alone.
    #[test]
    fn subcategories_only_category_page() {
        let generated = concat!(
            "<div class=\"mw-category-generated\">",
            "<div id=\"mw-subcategories\">",
            "<h2>Subcategories</h2>",
            "<p>This category has the following 2 subcategories, out of 2 total.</p>",
            "<div class=\"mw-content-ltr\"><div class=\"mw-category\">",
            "<div class=\"mw-category-group\"><h3>C</h3><ul>",
            "<li><a target=\"_parent\" href=\"Category%3AFencing_in_Canada\" title=\"Category:Fencing in Canada\">Fencing in Canada</a></li>",
            "</ul></div></div></div>",
            "</div>",
            "<p><em>This category content has been reduced to only pages contained in the ZIM file.</em></p>",
            "</div>"
        );
        let md = category_doc(
            "Category:Fencing in North America by country",
            "",
            generated,
            None,
            true,
        );
        assert_eq!(
            md,
            concat!(
                "# Category:Fencing in North America by country\n\n",
                "## Subcategories\n\n",
                "This category has the following 2 subcategories, out of 2 total.\n\n",
                "### C\n\n",
                "- [[Category:Fencing in Canada|Fencing in Canada]]\n\n",
                "*This category content has been reduced to only pages contained in the ZIM file.*\n"
            )
        );
    }

    /// (c) An empty category (no member/subcategory section — MediaWiki
    /// emits no lists and the archive ships no such page; the parser-output
    /// root is empty too) stays a no-content page: title only, flag on or
    /// off. A section reduced to pagination noise alone renders nothing.
    #[test]
    fn empty_category_page_stays_title_only() {
        let md = wiki_doc_with_tail("", "", None, true);
        assert_eq!(md, "# T\n");
        let md = wiki_doc_with_tail("", "", None, false);
        assert_eq!(md, "# T\n");
        let md = wiki_doc_with_tail("", cat_pagination(), None, true);
        assert_eq!(md, "# T\n");
        let md = wiki_doc_with_tail(
            "",
            &format!("<div class=\"mw-category-generated\">{}</div>", cat_pagination()),
            None,
            true,
        );
        assert_eq!(md, "# T\n");
    }

    /// (d) Flag off on a full category page: byte-identical to the
    /// pre-member-section output — title and description only, the
    /// mw-category-generated section dropped with the rest of the page
    /// furniture (parity with zim2zim.py's title-only category pages).
    #[test]
    fn category_page_flag_off_is_byte_identical() {
        let md = category_doc(
            "Category:1957 Hindi-language films",
            "<p>This category is for <b><a rel=\"mw:WikiLink\" href=\"Hindi_language\" title=\"Hindi language\" class=\"mw-redirect\">Hindi-language</a></b> <b><a rel=\"mw:WikiLink\" href=\"Film\" title=\"Film\">films</a></b>.</p>",
            &format!("{}{}", cat_generated(), cat_bar_local()),
            None,
            false,
        );
        assert_eq!(
            md,
            "# Category:1957 Hindi-language films\n\n\
             This category is for **[[Hindi language|Hindi-language]]** **[[Film|films]]**.\n"
        );
    }

    /// Member-section lists render through the normal list walker, so the
    /// anchor rules hold verbatim: `./Category:…` and percent-encoded hrefs
    /// both fold to `[[Category:…|label]]`, a label equal to the full
    /// category title renders the bare `[[Category:…]]` form, and member
    /// links fold underscores (label==target -> bare).
    #[test]
    fn member_section_link_shapes_render_through_the_list_walker() {
        let generated = concat!(
            "<div class=\"mw-category-generated\">",
            "<div id=\"mw-subcategories\"><h2>Subcategories</h2>",
            "<div class=\"mw-content-ltr\"><div class=\"mw-category\">",
            "<div class=\"mw-category-group\"><h3>C</h3><ul>",
            "<li><a target=\"_parent\" href=\"./Category:Comedy_films_of_1957\">Comedy films of 1957</a></li>",
            "<li><a target=\"_parent\" href=\"Category%3A1957_films\">Category:1957 films</a></li>",
            "<li><a target=\"_parent\" href=\"Category%3ACrime_films_of_1957\" title=\"Category:Crime films of 1957\">Crime films of 1957</a></li>",
            "</ul></div></div></div></div>",
            "<div id=\"mw-pages\"><div class=\"mw-content-ltr\"><div class=\"mw-category\">",
            "<div class=\"mw-category-group\"><h3>A</h3><ul>",
            "<li><a target=\"_parent\" href=\"Aag_(1957_film)\">Aag (1957 film)</a></li>",
            "<li><a target=\"_parent\" href=\"Mother_India\">Mother India</a></li>",
            "</ul></div></div></div></div>",
            "</div>"
        );
        let md = wiki_doc_with_tail("", generated, None, true);
        assert_eq!(
            md,
            concat!(
                "# T\n\n",
                "## Subcategories\n\n",
                "### C\n\n",
                "- [[Category:Comedy films of 1957|Comedy films of 1957]]\n",
                "- [[Category:1957 films]]\n",
                "- [[Category:Crime films of 1957|Crime films of 1957]]\n\n",
                "### A\n\n",
                "- [[Aag (1957 film)]]\n",
                "- [[Mother India]]\n"
            )
        );
    }

    /// The non-letter TOC group key renders as an &nbsp;-only <h3>: an empty
    /// heading renders nothing instead of a bare `###` line.
    #[test]
    fn non_letter_toc_group_header_renders_nothing() {
        let generated = concat!(
            "<div class=\"mw-category-generated\">",
            "<div id=\"mw-pages\">",
            "<div class=\"mw-category\">",
            "<div class=\"mw-category-group\"><h3>&nbsp;</h3><ul>",
            "<li><a target=\"_parent\" href=\"Aag_(1957_film)\">Aag (1957 film)</a></li>",
            "</ul></div></div></div></div>"
        );
        let md = wiki_doc_with_tail("", generated, None, true);
        assert_eq!(md, "# T\n\n- [[Aag (1957 film)]]\n");
    }

    /// The French archive language localizes the section heading.
    #[test]
    fn categories_heading_is_localized() {
        let md = wiki_doc_with_tail("<p>Body.</p>", CAT_BAR, Some("fra"), true);
        assert_eq!(
            md,
            "# T\n\nBody.\n\n## Catégories\n\n\
             - [[Category:2003 albums|2003 albums]]\n\
             - [[Category:Evan Parker albums|Evan Parker albums]]\n"
        );
    }
}
