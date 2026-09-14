//! Tiny, dependency-free HTML helpers: text extraction for article intros and
//! heading-based section extraction. Not a full HTML parser; it is a byte
//! scanner good enough for the well-formed pages ZIMs (mostly Wikipedia) hold.

/// Case-insensitive byte substring search from index `from`.
fn find_ci(hay: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || from >= hay.len() {
        return None;
    }
    let limit = hay.len() - needle.len();
    for i in from..=limit {
        if hay[i..i + needle.len()].eq_ignore_ascii_case(needle) {
            return Some(i);
        }
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

/// Elements whose whole content is invisible or not part of an article's
/// lead: CSS/JS, tables (an infobox precedes the lead in MediaWiki output),
/// figures (image captions) and reference markers. `title` and `h1` are
/// skipped too: they carry the page title (`<title>` in the head;
/// `<h1 id="firstHeading">` inside the body on pages from older scrapers,
/// which lack the `mw-content-text` marker) and the title is already a
/// separate field of every search hit, so it must not open the intro as
/// well. Other heading levels are content and pass through.
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

/// Whether an open tag's attributes mark a MediaWiki hatnote. `{{about}}`,
/// `{{other uses}}` and `{{main}}` render as elements like
/// `<div role="note" class="hatnote navigation-not-searchable">`: the class
/// list contains the token `hatnote` (order, extra classes and quote style
/// vary) or the role is `note`.
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
/// `open_i`. Tables nest (an infobox holds nested tables), so their close
/// tag is matched by depth - ending the skip at an inner table's close
/// would spill the rest of the infobox into the text. Other skipped
/// elements never nest; their first close tag ends the skip (a script's
/// text may itself contain "<script", so depth counting could overshoot
/// there). An element never closed within the scanned text ends the skip
/// at the end of it.
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

/// The region of a MediaWiki page that holds the article. Scanning the whole
/// HTML would pick up browser-chrome text (title bar, navigation menus), so
/// cut the page at the content div (`id="mw-content-text"`) and its matching
/// close when the page has one. The article preview is only a prefix of the
/// page, and on real MediaWiki pages the content div's close tag lies far
/// beyond that prefix: the div then never closes within the scanned text,
/// and the fallback must stay inside the div - everything from its open tag
/// onward is the article body. Widening back to the whole document there
/// (as this once did) silently defeats the scoping and leaks the head
/// chrome (`<title>`, `<h1 id="firstHeading">`) into the intro. Likewise,
/// when the prefix ends inside the div's own open tag (no `>` after the
/// marker), nothing of the article has arrived and the body is empty
/// rather than the whole document.
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
/// single spaces. Stops after producing `max_chars` characters of text;
/// markup and whitespace never consume the budget. MediaWiki hatnotes
/// (`{{about}}` and friends, marked by `is_hatnote_attrs`) and the page
/// title's own elements (`<title>`, `<h1>` - see `INTRO_SKIP_TAGS`) are
/// skipped whole, so intros start with the article lead and never open
/// with the article name, which every search hit already reports as its
/// own field - as the Markdown path already does.
pub fn intro_from_html(html: &str, max_chars: usize) -> String {
    let html = article_body(html);
    let bytes = html.as_bytes();
    let mut out = String::new();
    // Whitespace or markup seen since the last emitted text: emits a single
    // separating space when the next text arrives.
    let mut sep = false;
    let mut i = 0usize;
    while i < bytes.len() && out.chars().count() < max_chars {
        match bytes[i] {
            b'<' => {
                let Some((tag_name, attrs, opens, after)) = tag_at(html, i) else {
                    break;
                };
                // Elements whose whole content stays out of the intro: the
                // INTRO_SKIP_TAGS set plus hatnotes (`is_hatnote_attrs`).
                // Only a real open tag qualifies: a close tag (`</div>`)
                // splits into an empty tag name and a comment (`<!-- .. -->`)
                // starts with `!`, and neither must read as a hatnote (a
                // comment could quote `class="hatnote"` in its text). Jump
                // to the element's close tag (hatnote divs hold no nested
                // same-name element); failing that, drop just the open tag.
                let skip = opens
                    && (INTRO_SKIP_TAGS.iter().any(|t| tag_name.eq_ignore_ascii_case(t))
                        || is_hatnote_attrs(attrs));
                if skip {
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

/// Tags that implicitly close an open `<p>`: browsers close a `<p>` before
/// any block-level element. Well-formed wiki HTML closes it explicitly;
/// this only guards the odd page that leaves it open.
const P_CLOSERS: [&str; 14] = [
    "div", "ul", "ol", "dl", "li", "table", "blockquote", "pre", "hr", "h2", "h3", "h4", "h5",
    "h6",
];

/// Finish the paragraph under construction: collapse its whitespace, cap it
/// at `max_para_chars` characters, and keep it when it holds any text.
fn push_para(paras: &mut Vec<String>, cur: &mut String, max_para_chars: usize) {
    let text: String = cur.split_whitespace().collect::<Vec<_>>().join(" ");
    cur.clear();
    if !text.is_empty() {
        paras.push(text.chars().take(max_para_chars).collect());
    }
}

/// The cleaned paragraph texts of an HTML region: the text of each `<p>`
/// element, extracted with the same skip machinery as `intro_from_html`
/// (skipped elements - style/script/table/figure/sup/title/h1 - and
/// hatnotes yield nothing, so infobox or figure text never becomes a
/// paragraph), tags stripped, entities decoded, whitespace collapsed, each
/// paragraph capped at `max_para_chars`. Text outside `<p>` elements -
/// heading text, list items, navigation blocks - is not prose and stays
/// out.
fn paragraphs(html: &str, max_para_chars: usize) -> Vec<String> {
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
                let skip = opens
                    && (INTRO_SKIP_TAGS.iter().any(|t| tag_name.eq_ignore_ascii_case(t))
                        || is_hatnote_attrs(attrs));
                if skip {
                    i = skip_end(html, i, tag_name);
                    continue;
                }
                if opens && tag_name.eq_ignore_ascii_case("p") {
                    push_para(&mut paras, &mut cur, max_para_chars);
                    in_p = true;
                } else if !opens && tag_name.eq_ignore_ascii_case("p") {
                    push_para(&mut paras, &mut cur, max_para_chars);
                    in_p = false;
                } else if opens && P_CLOSERS.iter().any(|t| tag_name.eq_ignore_ascii_case(t)) {
                    push_para(&mut paras, &mut cur, max_para_chars);
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
                // `i` is always on a char boundary: we advance by
                // len_utf8() or by single-byte ASCII steps.
                let ch = html[i..].chars().next().unwrap();
                if in_p {
                    cur.push(ch);
                }
                i += ch.len_utf8();
            }
        }
    }
    // A `<p>` left open (e.g. the scan was truncated inside it) still counts.
    push_para(&mut paras, &mut cur, max_para_chars);
    paras
}

/// Split an article into its intro region and one region per heading, with
/// the cleaned paragraph texts of each region. Returns `(name, paragraphs)`
/// pairs: the first entry carries the intro region (everything before the
/// first heading) under the empty name, the later entries carry each
/// heading's section under the heading text as written. A heading's section
/// spans what [`section_content`] would return for it, so a nested
/// `<h3>`'s paragraphs belong to its own entry and to the enclosing
/// `<h2>`'s. `<h1>` carries the page title (see `INTRO_SKIP_TAGS`), not a
/// section: it bounds no entry, and its element is skipped like in
/// `intro_from_html`, so its text stays out of the intro paragraphs.
pub fn sections(html: &str, max_para_chars: usize) -> Vec<(String, Vec<String>)> {
    let body = article_body(html);
    let headings = collect_headings(body);
    let secs: Vec<&Heading> = headings.iter().filter(|h| h.level >= 2).collect();
    let mut out = Vec::with_capacity(secs.len() + 1);
    let intro_end = secs.first().map_or(body.len(), |h| h.start);
    out.push((String::new(), paragraphs(&body[..intro_end], max_para_chars)));
    for h in &secs {
        out.push((
            h.name.clone(),
            paragraphs(&body[h.content_start..h.content_end], max_para_chars),
        ));
    }
    out
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
        let level = (lt + 2 < bytes.len()
            && (bytes[lt + 1] == b'h' || bytes[lt + 1] == b'H')
            && bytes.get(lt + 2).copied().map(|c| c - b'0')
            .is_some_and(|d| (1..=6).contains(&d))
            && matches!(bytes.get(lt + 3), Some(b' ') | Some(b'>') | Some(b'/')))
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
pub fn section_content(html: &str, name: &str) -> Option<(String, String)> {
    let target = normalize(name);
    if target.is_empty() {
        return None;
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
        let intro = intro_from_html(WIKI, 100);
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
    fn intro_respects_char_limit() {
        let intro = intro_from_html(WIKI, 10);
        assert!(intro.chars().count() <= 10, "{intro:?}");
    }

    #[test]
    fn intro_budget_ignores_markup() {
        // Markup between texts must not eat the character budget: even
        // hundreds of tags cannot crowd later text out of the intro.
        let page = format!("<p>x</p>{}<p>real text here</p>", "<span></span>".repeat(400));
        let intro = intro_from_html(&page, 20);
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
        let intro = intro_from_html(page, 100);
        assert!(intro.starts_with("Chemistry is the study of & matter"), "{intro:?}");
        assert!(!intro.contains("atomic weight"), "{intro:?}");
        assert!(!intro.contains("hydrocarbon"), "{intro:?}");
        assert!(!intro.contains("Navigation"), "{intro:?}");
        assert!(!intro.contains("Wikipedia"), "{intro:?}");
    }

    #[test]
    fn intro_when_content_div_never_closes_in_prefix() {
        // A real page is read as a prefix (64 KiB) that ends inside the
        // content div - its close tag lies far beyond it. Before the fix
        // the never-closed fallback widened back to the whole document,
        // so the head chrome (<title>, <h1>) opened the intro ("Salt Salt
        // Salt is a mineral ..."); the intro must hold only lead text.
        let page = "<html><head><title>Salt - Wikipedia</title></head>\
            <body><h1 id=\"firstHeading\" class=\"firstHeading mw-first-heading\">Salt</h1>\
            <div id=\"mw-content-text\"><div class=\"mw-parser-output\">\
            <p><b>Salt</b> is a mineral composed primarily of sodium chloride.</p>\
            <p>It is an ionic compound.</p>";
        let intro = intro_from_html(page, 200);
        assert_eq!(
            intro,
            "Salt is a mineral composed primarily of sodium chloride. It is an ionic compound.",
            "{intro:?}"
        );
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
        let intro = intro_from_html(page, 100);
        assert_eq!(intro, "It is a lightweight metal.");
    }

    #[test]
    fn article_body_fallbacks_when_div_never_closes() {
        // Prefix ends inside the content div (a real page's close tag lies
        // beyond the prefix): everything from the div's open tag onward is
        // the article body - not the whole document, whose head chrome
        // would leak into the intro ("Salt Salt Salt is ...").
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
    fn intro_skips_hatnotes() {
        // Vector 2022 pages open the article body with hatnote divs
        // ({{about}} and friends) before the lead paragraph: the note, its
        // inner link and the deduplicated-style <link> MediaWiki puts next
        // to it must not reach the intro.
        let page = "<html><head><title>Solid oxygen - Wikipedia</title></head><body><div id=\"mw-content-text\"><div class=\"mw-parser-output\"><div role=\"note\" class=\"hatnote navigation-not-searchable\">This article is about the solid phase of elemental oxygen. For other uses, see <a href=\"Oxygen\" title=\"Oxygen\">Oxygen</a>.</div><link rel=\"mw-deduplicated-inline-style\" href=\"mw-data:TemplateStyles:r128\"/><p><b>Solid oxygen</b> forms below 54.36 K at normal pressure.</p></div></div><footer>Navigation menu</footer></body></html>";
        let intro = intro_from_html(page, 200);
        assert_eq!(intro, "Solid oxygen forms below 54.36 K at normal pressure.");
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
        let intro = intro_from_html(page, 100);
        assert_eq!(intro, "Apple Books is an e-book reader.");

        // A scan truncated inside the table skips to the end of the input:
        // everything after is the table's content.
        let page = "<div id=\"mw-content-text\"><table><tr><td>infobox";
        assert_eq!(intro_from_html(page, 100), "");
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
            let intro = intro_from_html(&page, 100);
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
            let intro = intro_from_html(&page, 100);
            assert_eq!(intro, "For other uses, see X. Lead text here.", "{el}");
        }
        // The word "hatnote" in plain text changes nothing, and close tags
        // (`</div>`) are never mistaken for hatnote open tags.
        let page = "<div id=\"mw-content-text\"><p>The hatnote template renders notes.</p><p>More.</p></div>";
        assert_eq!(
            intro_from_html(page, 100),
            "The hatnote template renders notes. More."
        );
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

        let secs = sections(page, 300);
        // The intro region: the lead paragraphs only (title, hatnote and
        // infobox skipped), under the empty name.
        assert_eq!(secs[0].0, "");
        assert_eq!(
            secs[0].1,
            vec![
                "Salt is a mineral composed of sodium chloride.".to_string(),
                "It tastes salty.".to_string(),
            ]
        );
        // One entry per heading (nested ones included), named as written,
        // each with its own paragraphs.
        assert_eq!(secs[1].0, "History");
        // A heading's section spans what `section_content` would return for
        // it: the nested h3's paragraphs belong to it as well.
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
        assert_eq!(sections(flat, 100), vec![("".to_string(), vec!["Just a lead.".to_string(), "And more.".to_string()])]);
        // No <p> anywhere: empty regions (callers fall back gracefully).
        assert_eq!(sections("<div>no paragraphs here</div>", 100), vec![("".to_string(), Vec::<String>::new())]);
        // A paragraph is capped at `max_para_chars` characters.
        let long = sections("<p>0123456789 0123456789 0123456789</p>", 12);
        assert_eq!(long[0].1, vec!["0123456789 0"]);
    }
}
