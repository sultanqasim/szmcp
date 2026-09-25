//! Tiny, dependency-free HTML helpers: text extraction for article intros and
//! heading-based section extraction. Not a full HTML parser; it is a byte
//! scanner good enough for the well-formed pages ZIMs (mostly Wikipedia) hold.

/// Case-insensitive byte substring search from index `from`. Returns `None`
/// when `needle` cannot occur in `hay` (empty, or longer than `hay`).
fn find_ci(hay: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    let limit = hay.len() - needle.len();
    let first = needle[0];
    let mut i = from;
    while i <= limit {
        // A match starts with the needle's first byte: scan for that one
        // byte per position instead of comparing the whole needle, which
        // keeps the long close-tag scans (a heading's `</hN>` can lie tens
        // of kilobytes away) cheap.
        let Some(step) = hay[i..=limit].iter().position(|&c| c.eq_ignore_ascii_case(&first)) else {
            return None;
        };
        i += step;
        if hay[i..i + needle.len()].eq_ignore_ascii_case(needle) {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Decode a character entity starting at `i` (i.e. bytes[i] == b'&').
/// Returns (text, index after the semicolon).
fn decode_entity(s: &str, i: usize) -> Option<(String, usize)> {
    let rest = &s[i..];
    let semi = rest.find(';')?;
    if semi == 0 {
        return None;
    }
    let body = &rest[1..semi];
    let ch = if let Some(d) = body.strip_prefix('#') {
        let code = if let Some(h) = d.strip_prefix('x').or_else(|| d.strip_prefix('X')) {
            u32::from_str_radix(h, 16).ok()?
        } else {
            d.parse::<u32>().ok()?
        };
        char::from_u32(code)?
    } else {
        match body {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            "nbsp" => '\u{00a0}',
            _ => return None,
        }
    };
    Some((ch.to_string(), i + semi + 1))
}

/// Elements whose whole content stays out of an article's lead: CSS/JS,
/// tables (an infobox precedes the lead in MediaWiki output), figures
/// (captions) and reference markers. `title` and `h1` carry the page title
/// (in the head, or in the body on pages from older scrapers), which every
/// search hit already reports as its own field; other heading levels are
/// content and pass through.
const INTRO_SKIP_TAGS: [&str; 7] = ["style", "script", "table", "figure", "sup", "title", "h1"];

/// The value of attribute `name` in an open tag's attribute text (the part
/// of the tag left over after the tag name), or `None`. Names match
/// case-insensitively but must start a fresh attribute (preceded by
/// whitespace or the `/` separator), so `data-class` does not read as
/// `class`; values may be double- or single-quoted or bare.
fn attr_value<'a>(attrs: &'a str, name: &str) -> Option<&'a str> {
    let bytes = attrs.as_bytes();
    let mut from = 0usize;
    while let Some(p) = find_ci(bytes, from, name.as_bytes()) {
        from = p + 1;
        if p > 0 && !bytes[p - 1].is_ascii_whitespace() && bytes[p - 1] != b'/' {
            continue; // mid-word: a longer name (`data-class`)
        }
        // Followed by `=`, then the value: quoted (`".."`, `'..'`) or bare.
        let Some(rest) = attrs[p + name.len()..].trim_start().strip_prefix('=') else {
            continue; // no `=` here: not this attribute (or a value's text)
        };
        let rest = rest.trim_start();
        return Some(match rest.as_bytes().first() {
            Some(b'"') | Some(b'\'') => {
                let q = rest.as_bytes()[0] as char;
                rest[1..].split(q).next().unwrap_or("")
            }
            _ => rest.split_ascii_whitespace().next().unwrap_or(""),
        });
    }
    None
}

/// Whether an open tag's attributes mark a MediaWiki hatnote (`{{about}}`
/// and friends render as e.g.
/// `<div role="note" class="hatnote navigation-not-searchable">`).
fn is_hatnote_attrs(attrs: &str) -> bool {
    attr_value(attrs, "class").is_some_and(|v| {
        v.split_ascii_whitespace().any(|t| t.eq_ignore_ascii_case("hatnote"))
    }) || attr_value(attrs, "role").is_some_and(|v| v.trim().eq_ignore_ascii_case("note"))
}

/// The index just past the close tag matching the open `<tag_name ...>` at
/// `open_i`, counting nested same-name elements. Returns the end of the
/// input when the element is never closed (e.g. a prefix truncated inside
/// the element): everything after it is then the element's content.
fn element_end(html: &str, open_i: usize, tag_name: &str) -> usize {
    let bytes = html.as_bytes();
    let mut depth = 1usize;
    let mut j = open_i + 1;
    while let Some(rel) = bytes[j..].iter().position(|&c| c == b'<') {
        let lt = j + rel;
        let rest = &html[lt + 1..];
        let (is_close, name) = match rest.strip_prefix('/') {
            Some(r) => (true, r),
            None => (false, rest),
        };
        let name_end = name.find(|c: char| !c.is_ascii_alphanumeric()).unwrap_or(name.len());
        if name_end == tag_name.len() && name[..name_end].eq_ignore_ascii_case(tag_name) {
            if is_close {
                depth -= 1;
                if depth == 0 {
                    return html[lt..].find('>').map_or(html.len(), |g| lt + g + 1);
                }
            } else {
                depth += 1;
            }
        }
        j = lt + 1;
    }
    html.len()
}

/// The tag starting at `<i>` (`html[i] == b'<'`): its name (as written),
/// its attribute text (empty for close tags and comments), whether it is an
/// open tag, and the index just past its `>`. `None` when the fragment ends
/// inside the tag.
fn tag_at(html: &str, i: usize) -> Option<(&str, &str, bool, usize)> {
    let bytes = html.as_bytes();
    let gt = i + bytes[i..].iter().position(|&c| c == b'>')?;
    let tag = &html[i + 1..gt];
    let name_end = tag
        .find(|c: char| c.is_whitespace() || c == '/')
        .unwrap_or(tag.len());
    let (name, attrs) = tag.split_at(name_end);
    let opens = name.as_bytes().first().is_some_and(|&c| c.is_ascii_alphabetic());
    Some((name, attrs, opens, gt + 1))
}

/// The index just past the close tag matching the skipped open element at
/// `open_i`. Tables nest (an infobox holds nested tables), so they are
/// matched by depth; other skipped elements never nest and end at their
/// first close tag (a script's text may contain "<script", so depth
/// counting could overshoot there). An element never closed within the
/// scanned text ends the skip at the end of it.
fn skip_end(html: &str, open_i: usize, tag_name: &str) -> usize {
    if tag_name.eq_ignore_ascii_case("table") {
        element_end(html, open_i, tag_name)
    } else {
        let close = format!("</{tag_name}");
        find_ci(html.as_bytes(), open_i + 1, close.as_bytes()).map_or(html.len(), |cp| {
            html[cp..].find('>').map_or(html.len(), |g| cp + g + 1)
        })
    }
}

/// Whether an open tag is skipped whole by the article scans: it names an
/// `INTRO_SKIP_TAG` or carries hatnote attributes (`is_hatnote_attrs`). Only
/// real open tags qualify (a comment could quote `class="hatnote"`).
fn is_skip_tag(tag_name: &str, attrs: &str, opens: bool) -> bool {
    opens
        && (INTRO_SKIP_TAGS.iter().any(|t| tag_name.eq_ignore_ascii_case(t))
            || is_hatnote_attrs(attrs))
}

/// The region of a MediaWiki page that holds the article: cut the page at
/// the content div (`id="mw-content-text"`) so browser chrome (title bar,
/// navigation menus) stays out. The article preview is only a prefix of the
/// page, so the content div's close tag usually lies beyond it; the
/// fallback must stay INSIDE the div (everything from its open tag onward),
/// never widen to the whole document - that leaks head chrome
/// (`<title>`, `<h1 id="firstHeading">`) into the intro. When the prefix
/// ends inside the div's own open tag, nothing of the article has arrived
/// and the body is empty.
fn article_body(html: &str) -> &str {
    const MARKER: &str = "id=\"mw-content-text\"";
    let Some(i) = html.find(MARKER) else { return html };
    let after = &html[i + MARKER.len()..];
    // Skip the rest of the content div's own open tag, then find its close
    // by counting nested <div>/</div> tags.
    let Some(gt) = after.find('>') else { return "" };
    let after = &after[gt + 1..];
    let mut depth = 1usize; // inside the content div
    let mut pos = 0usize;
    while let Some(lt) = after[pos..].find('<') {
        let rest = &after[pos + lt + 1..];
        let (is_close, name) = match rest.strip_prefix('/') {
            Some(r) => (true, r),
            None => (false, rest),
        };
        let name = name
            .find(|c: char| !c.is_ascii_alphanumeric())
            .map_or(name, |e| &name[..e]);
        if name.eq_ignore_ascii_case("div") {
            if is_close {
                depth -= 1;
                if depth == 0 {
                    return &after[..pos + lt];
                }
            } else {
                depth += 1;
            }
        }
        pos += lt + 1;
    }
    // Never closed within the scanned text: we are inside the div, so the
    // rest of the scanned text is the article body.
    after
}

/// Strip HTML tags, decode common entities, and collapse whitespace into
/// single spaces. Hatnotes and the page title's own elements (`<title>`,
/// `<h1>` - see `INTRO_SKIP_TAGS`) are skipped whole, so intros start with
/// the article lead, never the article name.
pub fn intro_from_html(html: &str) -> String {
    let html = article_body(html);
    let bytes = html.as_bytes();
    let mut out = String::new();
    // Whitespace or markup seen since the last emitted text: emits a single
    // separating space when the next text arrives.
    let mut sep = false;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => {
                let Some((tag_name, attrs, opens, after)) = tag_at(html, i) else {
                    break;
                };
                if is_skip_tag(tag_name, attrs, opens) {
                    i = skip_end(html, i, tag_name);
                    sep = true;
                    continue;
                }
                sep = true;
                i = after;
            }
            b'&' => {
                let (text, next) = match decode_entity(html, i) {
                    Some((text, next)) => (text, next),
                    None => ("&".to_string(), i + 1),
                };
                if sep && !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&text);
                sep = false;
                i = next;
            }
            c if c.is_ascii_whitespace() => {
                sep = true;
                i += 1;
            }
            _ => {
                if sep && !out.is_empty() {
                    out.push(' ');
                }
                // `i` is always on a char boundary: we advance by
                // len_utf8() or by single-byte ASCII steps.
                let ch = html[i..].chars().next().unwrap();
                out.push(ch);
                sep = false;
                i += ch.len_utf8();
            }
        }
    }
    out.trim().to_string()
}

/// The name the intro region carries everywhere it is reported: as the
/// first entry of [`sections`] and in a search hit's `sections` list (the
/// markdown splitter mirrors both). A leading underscore marks it as a
/// reserved name, distinct from every heading text.
pub(crate) const INTRO_SECTION: &str = "_intro";

/// Tags that implicitly close an open `<p>`: browsers close a `<p>` before
/// any block-level element. Well-formed wiki HTML closes it explicitly;
/// this only guards the odd page that leaves it open.
const P_CLOSERS: [&str; 14] = [
    "div", "ul", "ol", "dl", "li", "table", "blockquote", "pre", "hr", "h2", "h3", "h4", "h5",
    "h6",
];

/// Finish the paragraph under construction: collapse its whitespace and
/// keep it when it holds any text.
fn push_para(paras: &mut Vec<String>, cur: &mut String) {
    let text: String = cur.split_whitespace().collect::<Vec<_>>().join(" ");
    cur.clear();
    if !text.is_empty() {
        paras.push(text);
    }
}

/// The cleaned paragraph texts of an HTML region: one per `<p>` element,
/// extracted with the same skip machinery as `intro_from_html` (skipped
/// elements and hatnotes yield nothing), tags stripped, entities decoded,
/// whitespace collapsed. Text outside `<p>` elements - heading text, list
/// items, navigation blocks - stays out.
fn paragraphs(html: &str) -> Vec<String> {
    let bytes = html.as_bytes();
    let mut paras: Vec<String> = Vec::new();
    // The `<p>` currently open and its text so far (uncollapsed).
    let mut cur = String::new();
    let mut in_p = false;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => {
                let Some((tag_name, attrs, opens, after)) = tag_at(html, i) else {
                    break;
                };
                if is_skip_tag(tag_name, attrs, opens) {
                    i = skip_end(html, i, tag_name);
                    continue;
                }
                if tag_name.eq_ignore_ascii_case("p") {
                    push_para(&mut paras, &mut cur);
                    in_p = opens;
                } else if opens && P_CLOSERS.iter().any(|t| tag_name.eq_ignore_ascii_case(t)) {
                    push_para(&mut paras, &mut cur);
                    in_p = false;
                }
                i = after;
            }
            b'&' => {
                let (text, next) = match decode_entity(html, i) {
                    Some((text, next)) => (text, next),
                    None => ("&".to_string(), i + 1),
                };
                if in_p {
                    cur.push_str(&text);
                }
                i = next;
            }
            c if c.is_ascii_whitespace() => {
                if in_p {
                    cur.push(' ');
                }
                i += 1;
            }
            _ => {
                // A run of plain text: push it whole instead of char by
                // char. The run ends at the next `<`, `&` or whitespace -
                // all ASCII bytes, so the slice end is a char boundary.
                let run = bytes[i..]
                    .iter()
                    .position(|&b| b == b'<' || b == b'&' || b.is_ascii_whitespace())
                    .map_or(html.len(), |p| i + p);
                if in_p {
                    cur.push_str(&html[i..run]);
                }
                i = run;
            }
        }
    }
    // A `<p>` left open (e.g. the scan was truncated inside it) still counts.
    push_para(&mut paras, &mut cur);
    paras
}

/// Split an article into its intro region and one region per heading, with
/// the cleaned paragraph texts of each region. Returns `(name, paragraphs)`
/// pairs: the first entry carries the intro region (everything before the
/// first heading) under the [`INTRO_SECTION`] name, the later entries carry
/// each heading's section under the heading text as written. A heading's
/// section spans what [`section_content`] would return for it, so a nested
/// `<h3>`'s paragraphs belong to its own entry and to the enclosing
/// `<h2>`'s. `<h1>` carries the page title, not a section: it bounds no
/// entry and is skipped like in `intro_from_html`.
pub fn sections(html: &str) -> Vec<(String, Vec<String>)> {
    let body = article_body(html);
    let headings = collect_headings(body);
    let mut out = Vec::with_capacity(headings.len() + 1);
    out.push((
        INTRO_SECTION.to_string(),
        paragraphs(&body[..intro_region_end(body, &headings)]),
    ));
    for h in headings.iter().filter(|h| h.level >= 2) {
        out.push((
            h.name.clone(),
            paragraphs(&body[h.content_start..h.content_end]),
        ));
    }
    out
}

/// The intro region's paragraphs: the paragraphs [`sections`] reports for
/// its `INTRO_SECTION` entry, without extracting any body section's
/// paragraphs (a search hit's lead fast path needs only these).
pub fn intro_paragraphs(html: &str) -> Vec<String> {
    let body = article_body(html);
    let headings = collect_headings(body);
    paragraphs(&body[..intro_region_end(body, &headings)])
}

/// Where the intro region ends: at the first heading of level >= 2 (`<h1>`
/// carries the page title, not a section), else at the end of the body.
/// Both [`sections`] and [`intro_paragraphs`] bound their intro region
/// with this, so the two agree on what the intro holds.
fn intro_region_end(body: &str, headings: &[Heading]) -> usize {
    headings
        .iter()
        .find(|h| h.level >= 2)
        .map_or(body.len(), |h| h.start)
}

/// Remove tags from a short string (used for heading text).
fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        if in_tag {
            if c == '>' {
                in_tag = false;
            }
        } else if c == '<' {
            in_tag = true;
        } else {
            out.push(c);
        }
    }
    out
}

/// Normalize a heading name/title for matching: trim, lowercase, collapse
/// whitespace.
pub(crate) fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Find all `<h1>`..`<h6>` headings and the text range each one governs
/// (up to the next heading of the same or higher prominence, else end of
/// document). The range starts after the heading's own `</hN>` and ends
/// before the next heading's markup, so it never spills heading tags into
/// the section content.
fn collect_headings(html: &str) -> Vec<Heading> {
    let bytes = html.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let lt = match bytes[i..].iter().position(|&c| c == b'<') {
            Some(p) => i + p,
            None => break,
        };
        // A heading tag: <hN followed by whitespace, '>', or '/'.
        let level = (matches!(bytes.get(lt + 1), Some(b'h' | b'H'))
            && matches!(bytes.get(lt + 2), Some(b'1'..=b'6'))
            && matches!(bytes.get(lt + 3), Some(b' ' | b'>' | b'/')))
        .then(|| bytes[lt + 2] - b'0');
        let Some(level) = level else {
            i = lt + 1;
            continue;
        };
        let gt = match bytes[lt..].iter().position(|&c| c == b'>') {
            Some(p) => lt + p,
            None => break,
        };
        let close_needle = format!("</h{level}");
        let close = match find_ci(bytes, gt + 1, close_needle.as_bytes()) {
            Some(c) => c,
            None => break,
        };
        let close_gt = match bytes[close..].iter().position(|&c| c == b'>') {
            Some(p) => close + p,
            None => break,
        };
        let inner = &html[gt + 1..close];
        let name = strip_tags(inner).trim().to_string();
        let content_start = close_gt + 1;
        out.push(Heading {
            level,
            name_norm: normalize(&name),
            name,
            start: lt,
            content_start,
            content_end: html.len(),
        });
        i = content_start;
    }
    // Each heading governs up to the next heading of the same or higher
    // prominence (smaller or equal level number). End the range at that
    // heading's own start (its `<hN>` tag), not at its content, so the
    // next section's heading markup stays out of this section's content.
    for i in 0..out.len() {
        for j in (i + 1)..out.len() {
            if out[j].level <= out[i].level {
                out[i].content_end = out[j].start;
                break;
            }
        }
    }
    out
}

struct Heading {
    level: u8,
    name_norm: String,
    name: String,
    /// Offset of the heading tag's opening `<`.
    start: usize,
    content_start: usize,
    content_end: usize,
}

/// Remove `<style>`/`<script>` elements (including their contents) from an
/// HTML fragment. Wikipedia sections open with TemplateStyles CSS blocks
/// that are not part of the visible section text.
pub(crate) fn strip_style_script(html: &str) -> String {
    let mut out = String::new();
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        out.push_str(&rest[..lt]);
        let after = &rest[lt + 1..];
        let name_end = after
            .find(|c: char| !c.is_ascii_alphanumeric())
            .unwrap_or(after.len());
        let name = after[..name_end].to_ascii_lowercase();
        let opens_element = !after.starts_with('/') && !after.starts_with('!');
        if opens_element && matches!(name.as_str(), "style" | "script") {
            // Skip to the matching close tag, or drop the rest if unclosed.
            let close = format!("</{}", name);
            match find_ci(after.as_bytes(), name_end, close.as_bytes()) {
                Some(cp) => {
                    let end = after[cp..].find('>').map_or(after.len(), |g| cp + g + 1);
                    rest = &after[end..];
                }
                None => return out,
            }
        } else {
            out.push('<');
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Trim trailing markup that belongs to what follows the section, e.g. the
/// `<div class="mw-heading">` wrapper MediaWiki opens right before the next
/// `<h2>`. Leaves closing tags (a normal section ends with `</p>` or `</ul>`)
/// in place.
fn trim_trailing_open_tag(s: &str) -> &str {
    let mut s = s.trim_end();
    while let Some(lt) = s.rfind('<') {
        let tail = &s[lt..];
        let single_tag = tail.len() > 1
            && tail[1..].ends_with('>')
            && !tail[1..].contains('<');
        if single_tag && tail.as_bytes()[1].is_ascii_alphabetic() {
            s = s[..lt].trim_end();
        } else {
            break;
        }
    }
    s
}

/// Find the content of the named section in an HTML document.
///
/// Matches the section name (case-insensitively, whitespace-normalized)
/// against all heading texts; among matches the most prominent (lowest
/// level), then earliest, heading wins. Returns the heading's actual text
/// plus the raw HTML fragment between that heading and the next
/// same-or-higher-level heading, trimmed.
///
/// The reserved name [`INTRO_SECTION`] selects the article's introduction
/// instead: the raw HTML of the intro region bounded by
/// [`intro_region_end`], exactly the region [`sections`] reports first. It
/// always exists (an article opening with a heading has an empty intro),
/// so an empty region is returned as empty content, never `None`.
pub fn section_content(html: &str, name: &str) -> Option<(String, String)> {
    let target = normalize(name);
    if target.is_empty() {
        return None;
    }
    // The reserved intro name can never match a heading text, so it is
    // special-cased before the heading search.
    if target == normalize(INTRO_SECTION) {
        let body = article_body(html);
        let headings = collect_headings(body);
        let content = strip_style_script(&body[..intro_region_end(body, &headings)]);
        let content = trim_trailing_open_tag(&content);
        return Some((INTRO_SECTION.to_string(), content.trim().to_string()));
    }
    let headings = collect_headings(html);
    let best = headings
        .iter()
        .enumerate()
        .filter(|(_, h)| h.name_norm == target)
        .min_by_key(|(i, h)| (h.level, *i))?;
    let h = best.1;
    let content = strip_style_script(&html[h.content_start..h.content_end]);
    let content = trim_trailing_open_tag(&content);
    Some((h.name.clone(), content.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIKI: &str = "<html><head><title>Apple</title><style>p{}</style></head>\
        <body><script>var x=1;</script><h1>Apple</h1>\
        <div role=\"note\" class=\"hatnote\">This article is about the fruit. For other uses, see <a href=\"Apple_(disambiguation)\" title=\"Apple (disambiguation)\">Apple (disambiguation)</a>.</div>\
        <p>An <b>apple</b> is the fruit of &lt;rosaceae&gt; trees.</p>\
        <div class=\"mw-heading mw-heading2\"><h2 id=\"History\">History</h2></div>\
        <p>Apples have been grown for 10,000 years.</p>\
        <div class=\"mw-heading mw-heading3\"><h3>Domestication</h3></div><p>Wild apples grew in Kazakhstan.</p>\
        <div class=\"mw-heading mw-heading2\"><h2 id=\"Uses\">Uses</h2></div>\
        <style data-mw-deduplicate=\"x\">.portalbox{padding:0}</style>\
        <p>Eaten fresh, cooked, or pressed into juice.</p>\
        </body></html>";

    #[test]
    fn intro_strips_tags_scripts_and_entities() {
        let intro = intro_from_html(WIKI);
        // The <title> and <h1> carry the page title, which every search hit
        // already reports as its own field: their text must not open the
        // intro, so it starts at the lead paragraph.
        assert!(intro.starts_with("An apple is the fruit of <rosaceae> trees"), "{intro:?}");
        assert!(!intro.starts_with("Apple"), "{intro:?}");
        assert!(!intro.contains("var x"), "{intro:?}");
        assert!(!intro.contains("p{}"), "{intro:?}");
        // The hatnote before the lead (note text and its inner link) is
        // skipped; the intro starts with the article text.
        assert!(!intro.contains("disambiguation"), "{intro:?}");
    }
    #[test]
    fn intro_skips_markup_between_texts() {
        // Skipped markup between two texts neither leaks into the intro nor
        // disturbs the join: hundreds of empty spans leave both texts,
        // separated by one space.
        let page = format!("<p>x</p>{}<p>real text here</p>", "<span></span>".repeat(400));
        let intro = intro_from_html(&page);
        assert_eq!(intro, "x real text here", "{intro:?}");
    }

    #[test]
    fn intro_scopes_to_article_body_and_skips_infobox() {
        let page = "<html><head><title>Chemistry - Wikipedia</title></head>\
            <body><h1>Chemistry</h1>\
            <div id=\"mw-content-text\"><div class=\"mw-parser-output\">\
            <style>.hatnote{font-style:italic}</style>\
            <table><tbody><tr><td>Standard atomic weight 1.008</td></tr></tbody></table>\
            <figure>Aromatic hydrocarbon rings</figure>\
            <p><b>Chemistry</b> is the study of &amp; matter.</p>\
            </div></div><footer>Navigation menu</footer></body></html>";
        let intro = intro_from_html(page);
        assert!(intro.starts_with("Chemistry is the study of & matter"), "{intro:?}");
        assert!(!intro.contains("atomic weight"), "{intro:?}");
        assert!(!intro.contains("hydrocarbon"), "{intro:?}");
        assert!(!intro.contains("Navigation"), "{intro:?}");
        assert!(!intro.contains("Wikipedia"), "{intro:?}");
    }
    #[test]
    fn intro_skips_title_and_first_heading_without_marker() {
        // Pages lacking the mw-content-text marker (older scrapers put the
        // <h1 id="firstHeading"> inside the body): the title's text must
        // not open the intro even there, so <title> and <h1> contents are
        // skipped like hatnotes; the intro starts at the lead paragraph.
        let page = "<html><head><title>Beryllium - Wikipedia</title></head>\
            <body><h1 id=\"firstHeading\">Beryllium</h1>\
            <p>It is a lightweight metal.</p></body></html>";
        let intro = intro_from_html(page);
        assert_eq!(intro, "It is a lightweight metal.");
    }

    #[test]
    fn article_body_fallbacks_when_div_never_closes() {
        // A real page's close tag lies beyond the read prefix: the body is
        // everything from the div's open tag onward, not the whole document
        // (whose head chrome once leaked "Salt Salt Salt ..." intros).
        let page = "<html><title>Salt</title><h1>Salt</h1>\
            <div id=\"mw-content-text\" class=\"x\"><p>Lead.</p>";
        assert_eq!(article_body(page), "<p>Lead.</p>");
        // Prefix ends inside the div's own open tag (no `>` after the
        // marker): nothing of the article has arrived, so the body is
        // empty rather than the whole document.
        let page = "<html><title>Salt</title><div id=\"mw-content-text\" cla";
        assert_eq!(article_body(page), "");
    }
    #[test]
    fn intro_skips_nested_infobox_tables() {
        // An infobox <table> holding a nested <table>: the skip must not end
        // at the inner table's close, or the rest of the infobox leaks into
        // the intro.
        let page = "<div id=\"mw-content-text\"><div class=\"mw-parser-output\">\
            <table><tbody><tr><td><table><tbody><tr><td>inner</td></tr></tbody></table></td></tr>\
            <tr><td>Written in Objective-C</td></tr></tbody></table>\
            <p><b>Apple Books</b> is an e-book reader.</p></div></div>";
        let intro = intro_from_html(page);
        assert_eq!(intro, "Apple Books is an e-book reader.");

        // A scan truncated inside the table skips to the end of the input:
        // everything after is the table's content.
        let page = "<div id=\"mw-content-text\"><table><tr><td>infobox";
        assert_eq!(intro_from_html(page), "");
    }

    #[test]
    fn hatnote_detection_by_class_token_or_role() {
        // `hatnote` anywhere in the class list, in any order, quote style
        // and case, on any element; or a bare role="note". Each element is
        // skipped whole, down to its close tag.
        let hatnotes = [
            "<div class=\"hatnote navigation-not-searchable\">For other uses, see X.</div>",
            "<div class='navigation-not-searchable hatnote'>For other uses, see X.</div>",
            "<div class=hatnote>For other uses, see X.</div>",
            "<span CLASS=\"navigation-not-searchable hatnote\">For other uses, see X.</span>",
            "<P class=\"Hatnote\">For other uses, see X.</P>",
            "<div role=\"note\" class=\"navigation-not-searchable\">Main article: X.</div>",
            "<div role=note>Not to be confused with X.</div>",
        ];
        for el in hatnotes {
            let page = format!(
                "<div id=\"mw-content-text\">{el}<p>Lead text here.</p></div>"
            );
            let intro = intro_from_html(&page);
            assert_eq!(intro, "Lead text here.", "{el}");
        }
    }

    #[test]
    fn intro_keeps_non_hatnotes() {
        // `navigation-not-searchable` alone is not a hatnote; neither are
        // other roles, lookalike attributes (`data-class`, `data-role`) or
        // "hatnote" inside another attribute's value. They stay in the intro.
        let keepers = [
            "<div class=\"navigation-not-searchable\">For other uses, see X.</div>",
            "<div role=\"presentation\" class=\"navbox\">For other uses, see X.</div>",
            "<div data-class=\"hatnote\">For other uses, see X.</div>",
            "<div data-role=\"note\">For other uses, see X.</div>",
            "<div title=\"hatnote\">For other uses, see X.</div>",
        ];
        for el in keepers {
            let page = format!(
                "<div id=\"mw-content-text\">{el}<p>Lead text here.</p></div>"
            );
            let intro = intro_from_html(&page);
            assert_eq!(intro, "For other uses, see X. Lead text here.", "{el}");
        }
        // The word "hatnote" in plain text changes nothing, and close tags
        // (`</div>`) are never mistaken for hatnote open tags.
        let page = "<div id=\"mw-content-text\"><p>The hatnote template renders notes.</p><p>More.</p></div>";
        assert_eq!(
            intro_from_html(page),
            "The hatnote template renders notes. More."
        );
    }

    #[test]
    fn attribute_shorter_than_probed_name_is_not_found() {
        // A `<br />` open tag splits into name "br" and attribute text " /";
        // probing `class` in that 2-byte haystack once underflowed
        // `hay.len() - needle.len()` in find_ci and panicked, aborting every
        // search on scraped ZIMs where `<br />` is ubiquitous.
        assert_eq!(attr_value(" /", "class"), None);
        assert_eq!(attr_value(" /", "role"), None);
        assert!(!is_hatnote_attrs(" /"));
        assert_eq!(intro_from_html("<p>a<br />b</p>"), "a b");
    }

    #[test]
    fn malformed_h_prefixes_are_not_headings() {
        // A non-digit byte after `<h` (in `c - b'0'`) once underflowed and
        // panicked the heading scan in debug builds.
        let headings =
            collect_headings("<p><h/ junk <h x><h- <H8></p><h2>Title</h2><p>Body.</p>");
        assert_eq!(headings.len(), 1);
        assert_eq!(headings[0].level, 2);
        assert_eq!(headings[0].name, "Title");
    }

    #[test]
    fn section_extraction() {
        let (hist_name, hist) = section_content(WIKI, "History").unwrap();
        assert_eq!(hist_name, "History");
        assert!(hist.contains("grown for 10,000 years"), "{hist:?}");
        // Includes the h3 subsection, stops at the next h2.
        assert!(hist.contains("Kazakhstan"), "{hist:?}");
        assert!(!hist.contains("pressed into juice"), "{hist:?}");
        // The next section's heading markup (including its mw-heading
        // wrapper div) stays out of the content.
        assert!(!hist.contains("<h2"), "{hist:?}");
        assert!(!hist.contains("Uses"), "{hist:?}");
        assert!(!hist.contains("mw-heading2"), "{hist:?}");

        let (uses_name, uses) = section_content(WIKI, "uses").unwrap();
        assert_eq!(uses_name, "Uses");
        assert!(uses.contains("juice"), "{uses:?}");
        assert!(!uses.contains("Kazakhstan"), "{uses:?}");
        // TemplateStyles CSS blocks are not part of the visible section.
        assert!(!uses.contains("portalbox"), "{uses:?}");
        assert!(!uses.contains("<style"), "{uses:?}");

        // Subsection on its own.
        let (dom_name, dom) = section_content(WIKI, "Domestication").unwrap();
        assert_eq!(dom_name, "Domestication");
        assert!(dom.contains("Kazakhstan"), "{dom:?}");
        assert!(!dom.contains("10,000 years"), "{dom:?}");
        assert!(dom.ends_with("</p>"), "{dom:?}");

        assert!(section_content(WIKI, "Nope").is_none());
        assert!(section_content(WIKI, "").is_none());
    }

    #[test]
    fn section_content_intro_region() {
        // The reserved intro name returns the intro region: everything
        // before the first h2, including the hatnote div (it is the
        // region's HTML), with style/script stripped like any section.
        let (name, intro) = section_content(WIKI, "_intro").unwrap();
        assert_eq!(name, "_intro");
        assert!(intro.contains("An <b>apple</b> is the fruit of &lt;rosaceae&gt; trees."), "{intro:?}");
        assert!(intro.contains("This article is about the fruit"), "{intro:?}");
        assert!(!intro.contains("var x"), "{intro:?}");
        assert!(!intro.contains("p{}"), "{intro:?}");
        // Bounded by the first heading, and the next section's heading
        // wrapper is trimmed off the end like in a named section.
        assert!(!intro.contains("grown for 10,000 years"), "{intro:?}");
        assert!(!intro.contains("<h2"), "{intro:?}");
        assert!(!intro.contains("mw-heading2"), "{intro:?}");

        // Matched case-insensitively, like heading names.
        assert!(section_content(WIKI, "_Intro").is_some());

        // The introduction always exists: an article opening with a heading
        // has an empty intro, not an error.
        let (name, intro) = section_content("<h2>Only</h2><p>Body.</p>", "_intro").unwrap();
        assert_eq!(name, "_intro");
        assert_eq!(intro, "");
    }

    #[test]
    fn sections_split_intro_and_headings_with_clean_paragraphs() {
        // Hatnote, an infobox table holding a nested table, a lead <p>, and
        // two <h2> sections with paragraphs - plus an <h1> page title inside
        // the body (older scrapers): the title, the hatnote and the infobox
        // must yield no paragraphs anywhere.
        let page = "<html><head><title>Salt - Wikipedia</title></head>\
            <body><div id=\"mw-content-text\"><div class=\"mw-parser-output\">\
            <h1 id=\"firstHeading\">Salt</h1>\
            <div role=\"note\" class=\"hatnote\">This article is about the mineral. For the seasoning, see Pepper.</div>\
            <table><tbody><tr><td><table><tbody><tr><td>inner</td></tr></tbody></table></td></tr>\
            <tr><td>Halite crystals</td></tr></tbody></table>\
            <p><b>Salt</b> is a mineral composed of sodium chloride.</p>\
            <p>It tastes salty.</p>\
            <div class=\"mw-heading mw-heading2\"><h2 id=\"History\">History</h2></div>\
            <p>Salt has been mined for millennia.</p>\
            <div class=\"mw-heading mw-heading3\"><h3>Trade</h3></div><p>Salt roads crossed continents.</p>\
            <div class=\"mw-heading mw-heading2\"><h2 id=\"Uses\">Uses</h2></div>\
            <style>.portalbox{padding:0}</style>\
            <p>Salt seasons food and preserves it.</p>\
            </div></div><footer>Navigation menu</footer></body></html>";

        let secs = sections(page);
        // The intro region: the lead paragraphs only (title, hatnote and
        // infobox skipped).
        assert_eq!(secs[0].0, "_intro");
        assert_eq!(
            secs[0].1,
            vec![
                "Salt is a mineral composed of sodium chloride.".to_string(),
                "It tastes salty.".to_string(),
            ]
        );
        // One entry per heading (nested ones included), named as written;
        // the nested h3's paragraphs belong to the enclosing h2 as well.
        assert_eq!(secs[1].0, "History");
        assert_eq!(
            secs[1].1,
            vec!["Salt has been mined for millennia.", "Salt roads crossed continents."]
        );
        assert_eq!(secs[2].0, "Trade");
        assert_eq!(secs[2].1, vec!["Salt roads crossed continents."]);
        assert_eq!(secs[3].0, "Uses");
        assert_eq!(secs[3].1, vec!["Salt seasons food and preserves it."]);
        assert_eq!(secs.len(), 4);
        // Infobox, hatnote and title text never becomes a paragraph.
        for (name, paras) in &secs {
            assert!(!paras.iter().any(|p| p.contains("Halite")), "{name}: {paras:?}");
            assert!(!paras.iter().any(|p| p.contains("seasoning, see")), "{name}: {paras:?}");
            assert!(!paras.iter().any(|p| p.contains("Salt Salt")), "{name}: {paras:?}");
            assert!(!paras.iter().any(|p| p.contains("Navigation")), "{name}: {paras:?}");
        }

        // No headings: a single intro entry.
        let flat = "<p>Just a lead.</p><p>And more.</p>";
        assert_eq!(sections(flat), vec![("_intro".to_string(), vec!["Just a lead.".to_string(), "And more.".to_string()])]);
        // No <p> anywhere: empty regions (callers fall back gracefully).
        assert_eq!(sections("<div>no paragraphs here</div>"), vec![("_intro".to_string(), Vec::<String>::new())]);
    }
}
