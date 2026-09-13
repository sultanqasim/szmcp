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

/// Push a separating single space, collapsing consecutive whitespace.
fn push_sep(out: &mut String) {
    if !out.is_empty() && out.chars().next_back().unwrap_or(' ') != ' ' {
        out.push(' ');
    }
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

/// Strip HTML tags (and the text of `<script>`/`<style>` elements), decode
/// common entities, and collapse whitespace. Stops after producing
/// `max_chars` characters of text.
pub fn intro_from_html(html: &str, max_chars: usize) -> String {
    let bytes = html.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if out.chars().count() >= max_chars {
            break;
        }
        match bytes[i] {
            b'<' => {
                let gt = match bytes[i..].iter().position(|&c| c == b'>') {
                    Some(p) => i + p,
                    None => {
                        out.push('<');
                        i += 1;
                        continue;
                    }
                };
                let tag = &html[i + 1..gt];
                let tag_name = tag
                    .split(|c: char| c.is_whitespace())
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if tag_name == "script" || tag_name == "style" {
                    // Skip the whole element, including its text.
                    let close = format!("</{tag_name}");
                    if let Some(cp) = find_ci(bytes, i + 1, close.as_bytes()) {
                        if let Some(g) = bytes[cp..].iter().position(|&c| c == b'>') {
                            i = cp + g + 1;
                            continue;
                        }
                    }
                }
                push_sep(&mut out);
                i = gt + 1;
            }
            b'&' => {
                if let Some((text, next)) = decode_entity(html, i) {
                    out.push_str(&text);
                    i = next;
                } else {
                    out.push('&');
                    i += 1;
                }
            }
            c if c.is_ascii_whitespace() => {
                push_sep(&mut out);
                i += 1;
            }
            _ => {
                // `i` is always on a char boundary: we advance by
                // len_utf8() or by single-byte ASCII steps.
                let ch = html[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    out.trim().to_string()
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
        assert!(intro.contains("apple is the fruit of <rosaceae> trees"), "{intro:?}");
        assert!(!intro.contains("var x"), "{intro:?}");
        assert!(!intro.contains("p{}"), "{intro:?}");
    }

    #[test]
    fn intro_respects_char_limit() {
        let intro = intro_from_html(WIKI, 10);
        assert!(intro.chars().count() <= 10, "{intro:?}");
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
}
