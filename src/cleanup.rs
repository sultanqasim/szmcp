//! The final cleanup pass and section assembly of the converter (ports
//! wikizim_parser/html2md.py's `_cleanup` / `_assemble` / `_repair_line`).

use crate::util::{
    collapse_ws, html_unescape, prev_char, re, run_end,
};

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
fn drop_section_blocks(
    body_md: &str,
    lang: Option<&str>,
) -> (Vec<String>, Vec<(u32, String, Vec<String>)>) {
    let mut intro: Vec<String> = Vec::new();
    let mut sections: Vec<(u32, String, Vec<String>)> = Vec::new(); // (level, heading, body)
    let mut skip_until = 0u32;
    for blk in body_md.split("\n\n") {
        let heading = re(r"(?s)^(#{2,6}) (.+)$")
            .captures(blk)
            .map(|c| (c.get(1).unwrap().len() as u32, c.get(2).unwrap().as_str()));
        match heading {
            Some((level, raw)) => {
                let heading = collapse_ws(raw).trim().to_string();
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
                sections.push((level, heading, Vec::new()));
            }
            None => {
                if skip_until > 0 {
                    continue;
                }
                match sections.last_mut() {
                    Some((_, _, body)) => body.push(blk.to_string()),
                    None => intro.push(blk.to_string()),
                }
            }
        }
    }
    (intro, sections)
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

/// Record a masked-off span, return its `\x00N\x00` placeholder.
fn stash(items: &mut Vec<String>, span: &str) -> String {
    items.push(span.to_string());
    format!("\x00{}\x00", items.len() - 1)
}

/// Mask URLs (`\S*://\S*`) as `\x00N\x00` placeholders so the punctuation
/// repairs cannot touch them; returns the masked text plus the extracted
/// URLs in order of appearance.  (Without a literal `"://"` nothing can
/// match, so the text is returned unchanged.)
fn mask_urls(s: &str) -> (String, Vec<String>) {
    let mut urls: Vec<String> = Vec::new();
    let masked = if s.contains("://") {
        re(r"\S*://\S*")
            .replace_all(s, |c: &regex::Captures| {
                stash(&mut urls, c.get(0).map(|m| m.as_str()).unwrap_or(""))
            })
            .into_owned()
    } else {
        s.to_string()
    };
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
        let next = re(r"\x00(\d+)\x00")
            .replace_all(&s, |c: &regex::Captures| {
                let idx: usize = c
                    .get(1)
                    .map(|m| m.as_str().parse().unwrap_or(0))
                    .unwrap_or(0);
                items.get(idx).cloned().unwrap_or_default()
            })
            .into_owned();
        if next == s {
            break; // dangling placeholder: leave it alone
        }
        s = next;
    }
    s
}

/// End offset of a `` (`+)(.*?)\1 `` match starting exactly at byte
/// offset `p`, where `kmax` is the length of the maximal backtick run at
/// `p`.  Python `re`-style backtracking: the greedy `` `+ `` runs down
/// from `kmax`, and for each run the `(.*?)` span grows lazily one char
/// at a time (`.` never matches a newline) until the same run of
/// backticks closes it; the first closure wins.
fn code_span_end(ln: &str, p: usize, kmax: usize) -> Option<usize> {
    let b = ln.as_bytes();
    for k in (1..=kmax).rev() {
        let mut l = p + k; // group 2 spans [p+k, l), lazily growing
        loop {
            // The backreference: k backticks right after the span.
            if l + k <= ln.len() && b[l..l + k].iter().all(|&x| x == b'`') {
                return Some(l + k);
            }
            if l >= ln.len() {
                break;
            }
            let c = ln[l..].chars().next().unwrap();
            if c == '\n' {
                break; // `.` cannot cross a newline
            }
            l += c.len_utf8();
        }
    }
    None
}

/// Mask inline code spans (`` `…` ``) as `\x00N\x00` placeholders (the
/// [`mask_urls`] shape), stashing the span text into `items`.  Scanned
/// with Python `re.sub` semantics: try each backtick position, replace
/// on a match and continue after it, else advance one character.
/// (Without a backtick nothing can match.)
fn mask_code_spans(ln: &str, items: &mut Vec<String>) -> String {
    if !ln.contains('`') {
        return ln.to_string();
    }
    let b = ln.as_bytes();
    let mut out = String::with_capacity(ln.len());
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some(k) = ln[from..].find('`') {
        let p = from + k;
        let mut kmax = 0usize;
        while p + kmax < ln.len() && b[p + kmax] == b'`' {
            kmax += 1;
        }
        match code_span_end(ln, p, kmax) {
            Some(end) => {
                out.push_str(&ln[last..p]);
                out.push_str(&stash(items, &ln[p..end]));
                last = end;
                from = end;
            }
            None => {
                from = p + 1;
            }
        }
    }
    out.push_str(&ln[last..]);
    out
}

/// Mask inline code spans (`` `…` ``) and Markdown link destinations (the
/// `(…)` of `[label](…)`) as `\x00N\x00` placeholders (the [`mask_urls`]
/// shape) so the entity decode cannot rewrite their verbatim `&…;` text.
/// Destinations run from `](` to the matching close paren (balanced parens
/// are part of the URL) or, if unbalanced, to end of line; code spans are
/// masked first, so a `](` inside one cannot start a destination.
fn mask_code_dests(ln: &str) -> (String, Vec<String>) {
    let mut items: Vec<String> = Vec::new();
    let mut masked = mask_code_spans(ln, &mut items);
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
                out.push_str(&stash(&mut items, &masked[i..end]));
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

/// `\(\s+(?=\S)` -> `(`: "( (" becomes "((", but "(  " at end of line
/// stays.  Candidates are the `(` positions; the greedy `\s+` takes the
/// maximal whitespace run, and the lookahead only holds when a non-space
/// char follows it — a run to end of line fails, and so does every
/// shorter prefix of the run (it would end in whitespace), so the
/// maximal-run test is the whole rule.  Python `re.sub` scan semantics:
/// replace and continue after the match.
fn sub_paren_space(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some(k) = s[from..].find('(') {
        let p = from + k;
        let run = run_end(s, p + 1, char::is_whitespace);
        if run > p + 1 && s[run..].chars().next().is_some_and(|c| !c.is_whitespace()) {
            out.push_str(&s[last..p]);
            out.push('(');
            last = run;
            from = run;
        } else {
            from = p + 1;
        }
    }
    out.push_str(&s[last..]);
    out
}

/// `(?<=\S)\s+\)` -> `)`: "( )" becomes "()", "(  )" too, while "( ) "
/// at start of line stays.  Candidates are the maximal whitespace runs
/// (greedy `\s+` can only reach a `)` at the run's end — a shorter run
/// would end in whitespace); the lookbehind wants a non-space char
/// before the run.
fn sub_ws_close_paren(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    let mut i = 0usize;
    while i < s.len() {
        let start = match s[i..].find(char::is_whitespace) {
            Some(k) => i + k,
            None => break,
        };
        let end = run_end(s, start, char::is_whitespace);
        if end < s.len()
            && s.as_bytes()[end] == b')'
            && prev_char(s, start).is_some_and(|c| !c.is_whitespace())
        {
            out.push_str(&s[last..start]);
            out.push(')');
            last = end + 1;
            i = end + 1;
        } else {
            i = end;
        }
    }
    out.push_str(&s[last..]);
    out
}

/// `(?<=[A-Za-z0-9])\.\.(?![.\w/])` -> `.`: "3.." becomes "3.", while
/// "a..." keeps its dots (the second pair is preceded by a dot) and
/// "a..b" keeps both dots (the lookahead sees a word char).  Candidates
/// are the ".." positions; a rejected pair is followed up one char later
/// because overlapping pairs share their middle dot.
fn sub_dot_pair(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some(k) = s[from..].find("..") {
        let p = from + k;
        // [A-Za-z0-9] is pure ASCII, so the char before the pair can be
        // tested on its last byte (a non-ASCII char ends in >= 0x80).
        let lookbehind = p > 0 && s.as_bytes()[p - 1].is_ascii_alphanumeric();
        let lookahead = !re("^[.\\w/]").is_match(&s[p + 2..]);
        if lookbehind && lookahead {
            out.push_str(&s[last..p]);
            out.push('.');
            last = p + 2;
            from = p + 2;
        } else {
            from = p + 1;
        }
    }
    out.push_str(&s[last..]);
    out
}

/// `(?<=\S) {2,}` -> ` `: "x   " becomes "x ", runs after whitespace or
/// at start of line stay.  Candidates are the maximal runs of spaces
/// (the core matches only `' '`), needing two or more.
fn sub_double_space(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    let mut i = 0usize;
    while i < s.len() {
        let start = match s[i..].find(' ') {
            Some(k) => i + k,
            None => break,
        };
        let end = run_end(s, start, |c| c == ' ');
        if end - start >= 2 && prev_char(s, start).is_some_and(|c| !c.is_whitespace()) {
            out.push_str(&s[last..start]);
            out.push(' ');
            last = end;
        }
        i = end;
    }
    out.push_str(&s[last..]);
    out
}

/// `,\s*\.(?![.\w])` -> `.`: ", ." becomes ".", ", .x" stays.  The
/// greedy `\s*` can only reach a `.` at the run's end, so the candidate
/// dot is uniquely placed; the lookahead holds past end of line too.
fn sub_comma_dot(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some(k) = s[from..].find(',') {
        let p = from + k;
        let run = run_end(s, p + 1, char::is_whitespace);
        if s[run..].starts_with('.') && !re("^[.\\w]").is_match(&s[run + 1..]) {
            out.push_str(&s[last..p]);
            out.push('.');
            last = run + 1;
            from = run + 1;
        } else {
            from = p + 1;
        }
    }
    out.push_str(&s[last..]);
    out
}

/// `,\s*,(?!\w)` -> `,`: ", ," becomes ",", ", ,a" stays.
fn sub_comma_comma(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some(k) = s[from..].find(',') {
        let p = from + k;
        let run = run_end(s, p + 1, char::is_whitespace);
        if s[run..].starts_with(',') && !re("^\\w").is_match(&s[run + 1..]) {
            out.push_str(&s[last..p]);
            out.push(',');
            last = run + 1;
            from = run + 1;
        } else {
            from = p + 1;
        }
    }
    out.push_str(&s[last..]);
    out
}

/// `(?<=[A-Za-z0-9\)\]"'*]) +\.(?![.\w])` -> `.`: "x ." becomes "x.",
/// "( ." and " ." after a word too, while " ) ." (preceded by a space)
/// and "x .y" stay.  Candidates are the maximal runs of spaces ending at
/// a dot; the lookbehind class is pure ASCII, so it is tested on the
/// last byte of the char before the run.
fn sub_cls_space_dot(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    let mut i = 0usize;
    while i < s.len() {
        let start = match s[i..].find(' ') {
            Some(k) => i + k,
            None => break,
        };
        let end = run_end(s, start, |c| c == ' ');
        let lookbehind = start > 0
            && matches!(
                s.as_bytes()[start - 1],
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b')' | b']' | b'"' | b'\'' | b'*'
            );
        if s[end..].starts_with('.') && lookbehind && !re("^[.\\w]").is_match(&s[end + 1..]) {
            out.push_str(&s[last..start]);
            out.push('.');
            last = end + 1;
            i = end + 1;
        } else {
            i = end;
        }
    }
    out.push_str(&s[last..]);
    out
}

/// `(?<=[a-z0-9\)])\.([A-Z])` -> `. X`: "end.He" becomes "end. He" (the
/// uppercase letter is kept), while "H.He" (H not in the class) stays.
/// The lookbehind class is pure ASCII, tested on the last byte of the
/// char before the dot; the `[A-Z]` is a single ASCII char.
fn sub_period_upper(joined: &str) -> String {
    let mut out = String::with_capacity(joined.len());
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some(k) = joined[from..].find('.') {
        let p = from + k;
        let upper = joined[p + 1..]
            .chars()
            .next()
            .filter(|c| c.is_ascii_uppercase());
        if let Some(c) = upper {
            if p > 0 && matches!(joined.as_bytes()[p - 1], b'a'..=b'z' | b'0'..=b'9' | b')') {
                out.push_str(&joined[last..p]);
                out.push_str(". ");
                out.push(c);
                last = p + 1 + c.len_utf8();
                from = last;
                continue;
            }
        }
        from = p + 1;
    }
    out.push_str(&joined[last..]);
    out
}

/// Generic punctuation/whitespace residue repair for one output line.
/// URLs are masked off for the duration.
fn repair_line(ln: &str) -> String {
    let (mut s, urls) = mask_urls(ln);
    // The `contains` checks are pure shortcuts — each rule fails without
    // its trigger characters — but skipping the scans keeps large
    // articles fast.
    if s.contains('(') {
        s = re(r"\(\s*\)").replace_all(&s, "").into_owned();
        s = sub_paren_space(&s);
    }
    if s.contains(')') {
        s = sub_ws_close_paren(&s);
    }
    if s.contains("..") {
        s = sub_dot_pair(&s);
    }
    if s.contains("  ") {
        s = sub_double_space(&s);
    }
    if s.contains(',') {
        s = re(r"^(\s*(?:[-*+] |\d+[.)] ))\s*,+\s*")
            .replace_all(&s, "$1")
            .into_owned();
        if s.contains('.') {
            s = sub_comma_dot(&s);
        }
        if s.matches(',').count() > 1 {
            s = sub_comma_comma(&s);
        }
    }
    if s.contains(" .") {
        s = sub_cls_space_dot(&s);
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
    let joined = sub_period_upper(&joined);
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
