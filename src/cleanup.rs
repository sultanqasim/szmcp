//! The final cleanup pass and section assembly of the converter (ports
//! wikizim_parser/html2md.py's `_cleanup` / `_assemble` / `_repair_line`).

use crate::util::{collapse_ws, fre, fre_sub, html_unescape, re};

/// Section headings whose entire section is dropped (English Wikipedia
/// names; wikil10n adds the per-language ones).
const DROP_SECTIONS: &[&str] = &[
    "references",
    "notes",
    "footnotes",
    "notes and references",
    "references and notes",
    "works cited",
    "external links",
    "further reading",
    "family tree",
];

pub(crate) fn is_dropped_section(heading: &str, lang: Option<&str>) -> bool {
    let norm = collapse_ws(heading).trim().to_lowercase();
    DROP_SECTIONS.contains(&norm.as_str())
        || crate::wikil10n::extra_drop_sections(lang).contains(&norm.as_str())
}

/// Split the flat block stream at heading lines and drop dropped-section
/// bodies (the heading + everything under deeper headings).
fn drop_section_blocks(body_md: &str, lang: Option<&str>) -> (Vec<String>, Vec<(u32, String, Vec<String>)>) {
    let mut intro: Vec<String> = Vec::new();
    struct Sec {
        level: u32,
        heading: String,
        body: Vec<String>,
    }
    let mut sections: Vec<Sec> = Vec::new();
    let mut skip_until = 0u32;
    for blk in body_md.split("\n\n") {
        let mut is_heading = None;
        if let Some(caps) = re(r"(?s)^(#{2,6}) (.+)$").captures(blk) {
            is_heading = Some((caps.get(1).unwrap().len() as u32, caps.get(2).unwrap().as_str()));
        }
        match is_heading {
            Some((level, heading_raw)) => {
                let heading = collapse_ws(heading_raw).trim().to_string();
                if skip_until > 0 {
                    if level <= skip_until {
                        skip_until = 0;
                    } else {
                        continue;
                    }
                }
                if is_dropped_section(&heading, lang) {
                    skip_until = level;
                    continue;
                }
                sections.push(Sec { level, heading, body: Vec::new() });
            }
            None => {
                if skip_until > 0 {
                    continue;
                }
                match sections.last_mut() {
                    Some(sec) => sec.body.push(blk.to_string()),
                    None => intro.push(blk.to_string()),
                }
            }
        }
    }
    (intro, sections.into_iter().map(|s| (s.level, s.heading, s.body)).collect())
}

/// Split rendered blocks into sections, drop dropped sections and apply
/// bottom-up empty-section pruning; return the final markdown body.
pub(crate) fn assemble(body_md: &str, lang: Option<&str>) -> String {
    let (intro, sections) = drop_section_blocks(body_md, lang);
    let mut alive: Vec<bool> = sections.iter().map(|_| false).collect();

    // Bottom-up empty-section pruning: a section with content marks its
    // nearest open ancestor alive.
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..sections.len() {
        while let Some(&top) = stack.last() {
            if sections[top].0 >= sections[i].0 {
                let popped = stack.pop().unwrap();
                if alive[popped] {
                    if let Some(&parent) = stack.last() {
                        alive[parent] = true;
                    }
                }
            } else {
                break;
            }
        }
        alive[i] = !sections[i].2.join("\n\n").trim().is_empty();
        stack.push(i);
    }
    while let Some(popped) = stack.pop() {
        if alive[popped] {
            if let Some(&parent) = stack.last() {
                alive[parent] = true;
            }
        }
    }

    let mut parts: Vec<String> = Vec::new();
    let intro_md = intro.join("\n\n");
    if !intro_md.trim().is_empty() {
        parts.push(intro_md);
    }
    for ((level, heading, body), alive) in sections.iter().zip(alive.iter()) {
        if !alive {
            continue;
        }
        parts.push(format!("{} {}", "#".repeat(*level as usize), heading));
        let rendered = body.join("\n\n");
        if !rendered.trim().is_empty() {
            parts.push(rendered);
        }
    }
    parts.join("\n\n")
}

/// Boolean per-line flags: fenced code blocks and HTML table blocks are
/// protected from the whitespace/punctuation repairs.
fn protected_lines(md: &str) -> Vec<bool> {
    let mut prot = Vec::new();
    let mut in_fence = false;
    let mut in_table = false;
    for ln in md.split('\n') {
        if ln.trim_start().starts_with("```") {
            prot.push(true);
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            prot.push(true);
            continue;
        }
        let low = ln.to_lowercase();
        if !in_table && low.trim_start().starts_with("<table") {
            in_table = true;
        }
        if in_table {
            prot.push(true);
            if low.contains("</table") {
                in_table = false;
            }
            continue;
        }
        prot.push(false);
    }
    prot
}

/// Mask URLs (`\S*://\S*`) as `\x00N\x00` placeholders so the punctuation
/// repairs cannot touch them; returns the masked text plus the extracted
/// URLs in order of appearance.  (Without a literal `"://"` nothing can
/// match, so the text is returned unchanged.)
fn mask_urls(s: &str) -> (String, Vec<String>) {
    if !s.contains("://") {
        return (s.to_string(), Vec::new());
    }
    let mut urls: Vec<String> = Vec::new();
    let masked = fre_sub(fre(r"\S*://\S*"), s, |c| {
        urls.push(c.get(0).map(|m| m.as_str()).unwrap_or("").to_string());
        format!("\x00{}\x00", urls.len() - 1)
    });
    (masked, urls)
}

/// Restore the `\x00N\x00` placeholders of [`mask_urls`] /
/// [`mask_code_dests`]; repeated until no placeholder is left, so a mask
/// that swallowed an earlier placeholder (a code span inside a link
/// destination) unwinds correctly.
fn unmask(s: &str, items: &[String]) -> String {
    if !s.contains('\x00') {
        return s.to_string();
    }
    let mut s = s.to_string();
    while s.contains('\x00') {
        let next = fre_sub(fre(r"\x00(\d+)\x00"), &s, |c| {
            let idx: usize = c.get(1).map(|m| m.as_str().parse().unwrap_or(0)).unwrap_or(0);
            items.get(idx).cloned().unwrap_or_default()
        });
        if next == s {
            break; // dangling placeholder: leave it alone
        }
        s = next;
    }
    s
}

/// Mask inline code spans (`` `…` ``) and Markdown link destinations (the
/// `(…)` of `[label](…)`) as `\x00N\x00` placeholders (the [`mask_urls`]
/// shape) so the entity decode cannot rewrite their verbatim `&…;` text.
/// Destinations run from `](` to the matching close paren (balanced parens
/// are part of the URL) or, if unbalanced, to end of line; code spans are
/// masked first, so a `](` inside one cannot start a destination.
fn mask_code_dests(ln: &str) -> (String, Vec<String>) {
    let mut items: Vec<String> = Vec::new();
    // Inline code spans: a backtick run, its content, the matching run.
    let mut masked = fre_sub(fre(r"(`+)(.*?)\1"), ln, |c| {
        items.push(c.get(0).map(|m| m.as_str()).unwrap_or("").to_string());
        format!("\x00{}\x00", items.len() - 1)
    });
    if masked.contains("](") {
        let b = masked.as_bytes();
        let mut out = String::with_capacity(masked.len());
        let mut last = 0usize;
        let mut i = 0usize;
        while i + 1 < b.len() {
            if b[i] == b']' && b[i + 1] == b'(' {
                let (mut depth, mut j) = (0usize, i + 1);
                let end = loop {
                    if j == b.len() {
                        break b.len(); // unbalanced parens: mask to end of line
                    }
                    if b[j] == b'(' {
                        depth += 1;
                    } else if b[j] == b')' {
                        depth -= 1;
                        if depth == 0 {
                            break j + 1;
                        }
                    }
                    j += 1;
                };
                out.push_str(&masked[last..i]);
                items.push(masked[i..end].to_string());
                out.push_str(&format!("\x00{}\x00", items.len() - 1));
                i = end;
                last = end;
            } else {
                i += 1;
            }
        }
        out.push_str(&masked[last..]);
        masked = out;
    }
    (masked, items)
}

/// Generic punctuation/whitespace residue repair for one output line.
/// URLs are masked off for the duration (see [`mask_urls`]).
fn repair_line(ln: &str) -> String {
    let (mut s, urls) = mask_urls(ln);
    if s.contains('(') {
        s = re(r"\(\s*\)").replace_all(&s, "").into_owned();
        s = fre_sub(fre(r"\(\s+(?=\S)"), &s, |_| "(".to_string());
    }
    if s.contains(')') {
        s = fre_sub(fre(r#"(?<=\S)\s+\)"#), &s, |_| ")".to_string());
    }
    if s.contains("..") {
        s = fre_sub(fre(r#"(?<=[A-Za-z0-9])\.\.(?![.\w/])"#), &s, |_| ".".to_string());
    }
    if s.contains("  ") {
        s = fre_sub(fre(r"(?<=\S) {2,}"), &s, |_| " ".to_string());
    }
    if s.contains(',') {
        s = re(r"^(\s*(?:[-*+] |\d+[.)] ))\s*,+\s*")
            .replace_all(&s, "$1")
            .into_owned();
        if s.contains('.') {
            s = fre_sub(fre(r",\s*\.(?![.\w])"), &s, |_| ".".to_string());
        }
        if s.matches(',').count() > 1 {
            s = fre_sub(fre(r",\s*,(?!\w)"), &s, |_| ",".to_string());
        }
    }
    if s.contains(" .") {
        s = fre_sub(fre(r#"(?<=[A-Za-z0-9\)\]"'*]) +\.(?![.\w])"#), &s, |_| ".".to_string());
    }
    unmask(&s, &urls)
}

/// Remove blank lines sitting strictly between two consecutive list-item
/// lines.
fn rejoin_split_lists(lines: Vec<String>, prot: &[bool]) -> Vec<String> {
    let list_item = re(r"^\s*(?:[-*+]|\d+[.)])\s");
    let mut out: Vec<String> = Vec::new();
    for (i, ln) in lines.iter().enumerate() {
        if ln.trim().is_empty()
            && !prot[i]
            && !out.is_empty()
            && i + 1 < lines.len()
            && !prot[i + 1]
            && list_item.is_match(out.last().unwrap())
            && list_item.is_match(&lines[i + 1])
        {
            continue;
        }
        out.push(ln.clone());
    }
    out
}

/// The entity-decode step of [`cleanup`]: `html_unescape` over the whole
/// document, except that fenced code blocks are left untouched entirely
/// and, on the other lines, inline code spans and Markdown link
/// destinations are masked off for the duration ([`mask_code_dests`]) —
/// their `&…;` text is verbatim content the HTML parser deliberately kept
/// literal, not residue to clean up.
fn unescape_residue(md: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for ln in md.split('\n') {
        if ln.trim_start().starts_with("```") {
            in_fence = !in_fence;
            out.push(ln.to_string());
            continue;
        }
        if in_fence {
            out.push(ln.to_string());
            continue;
        }
        let (masked, spans) = mask_code_dests(ln);
        out.push(unmask(&html_unescape(&masked), &spans));
    }
    out.join("\n")
}

/// The final cleanup pass: entity decode (leaving fenced code blocks,
/// inline code spans and link destinations literal — see
/// [`unescape_residue`]), nbsp/feff collapse, edit-link
/// residue removal, blank normalization, per-line repairs and list
/// rejoining, then one last `end.He` -> `end. He` sentence-period repair
/// on the joined text — with URLs masked off for that repair, since a
/// period inside a URL (`example.NET`, `Page.Us`) is not a sentence end.
pub(crate) fn cleanup(md: &str) -> String {
    let md = unescape_residue(md);
    let md = md
        .replace('\u{00a0}', " ")
        .replace('\u{feff}', "")
        .replace('\u{200b}', "");
    let md = md.replace("[edit]", "").replace("[modifier]", "");

    // No doubled blank lines; rstrip line ends.
    let mut lines: Vec<String> = Vec::new();
    let mut prev_blank = false;
    for ln in md.split('\n') {
        let ln = ln.trim_end();
        let blank = ln.trim().is_empty();
        if blank && prev_blank {
            continue;
        }
        lines.push(ln.to_string());
        prev_blank = blank;
    }

    let joined = lines.join("\n");
    let prot = protected_lines(&joined);
    let lines: Vec<&str> = joined.split('\n').collect();

    // Per-line transforms on unprotected lines, in order: `$$` display-math
    // lines are left untouched, everything else gets the repairs.
    let mut out: Vec<String> = Vec::new();
    let mut in_math = false;
    for (ln, p) in lines.iter().zip(prot.iter()) {
        if *p {
            out.push((*ln).to_string());
            continue;
        }
        if ln.contains("$$") {
            out.push((*ln).to_string());
            if ln.matches("$$").count() % 2 == 1 {
                in_math = !in_math;
            }
            continue;
        }
        if in_math {
            out.push((*ln).to_string());
            continue;
        }
        out.push(repair_line(ln));
    }

    let out = rejoin_split_lists(out, &prot);
    // Restore the space after a sentence period before an uppercase letter,
    // with URLs masked so the repair stays out of them (`example.NET` keeps
    // its dot; `end.He` still gains the space).
    let (joined, urls) = mask_urls(&out.join("\n"));
    let joined = fre_sub(fre(r"(?<=[a-z0-9\)])\.([A-Z])"), &joined, |c| {
        format!(". {}", c.get(1).map(|m| m.as_str()).unwrap_or(""))
    });
    let md = unmask(&joined, &urls);
    format!("{}\n", md.trim_matches('\n'))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The final `end.He` -> `end. He` repair gains the space while the
    /// URL on the same line is left untouched.
    #[test]
    fn period_upper_repair_skips_urls() {
        let md = "See [x](https://example.com/About) now end.He said.";
        assert_eq!(
            cleanup(md),
            "See [x](https://example.com/About) now end. He said.\n"
        );
    }

    /// Same for URLs whose own dot+uppercase would be hit by the repair
    /// (`Page.Us` / `example.NET` used to become `Page. Us` /
    /// `example. NET`).
    #[test]
    fn period_upper_repair_keeps_url_dots() {
        let md = "See [x](https://example.com/Page.Us) and <https://example.NET>. Then end.He said.";
        assert_eq!(
            cleanup(md),
            "See [x](https://example.com/Page.Us) and <https://example.NET>. Then end. He said.\n"
        );
    }

    /// Fenced code blocks display entity text verbatim (`<pre>&lt;?php</pre>`
    /// parses to a literal `&lt;?php` inside the fence), so the entity
    /// decode must leave them untouched while the surrounding prose still
    /// decodes (`&notit;` -> `¬it;`).
    #[test]
    fn fenced_code_keeps_entities() {
        let md = "Prose &notit; here.\n\n```html\n&lt;?php\n$a = &amp;lt;b&gt;;\n```\n\nAfter &notit;.";
        assert_eq!(
            cleanup(md),
            "Prose ¬it; here.\n\n```html\n&lt;?php\n$a = &amp;lt;b&gt;;\n```\n\nAfter ¬it;.\n"
        );
    }

    /// Same for inline code spans: the entity text between backticks is
    /// content, the entity in the surrounding prose is residue.
    #[test]
    fn inline_code_span_keeps_entities() {
        let md = "Use `&lt;` for < and `&amp;lt;` here, not `&gt;`-shaped.";
        assert_eq!(
            cleanup(md),
            "Use `&lt;` for < and `&amp;lt;` here, not `&gt;`-shaped.\n"
        );
    }

    /// A link destination keeps its literal `&...;` query text while the
    /// same residue in prose still decodes.
    #[test]
    fn link_destination_keeps_entities() {
        let md = "See [l](https://x.example/a?x=&notit&amp;y=2) now.\n\nElsewhere &notit; ends.";
        assert_eq!(
            cleanup(md),
            "See [l](https://x.example/a?x=&notit&amp;y=2) now.\n\nElsewhere ¬it; ends.\n"
        );
    }

    /// Balanced parens are part of the destination and stay inside the
    /// mask; the prose after the link still decodes.
    #[test]
    fn link_destination_with_parens_keeps_entities() {
        let md = "[l](https://x.example/a_(b)&amp;c) tail &notit;.";
        assert_eq!(
            cleanup(md),
            "[l](https://x.example/a_(b)&amp;c) tail ¬it;.\n"
        );
    }
}
