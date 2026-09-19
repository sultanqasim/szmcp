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

/// Generic punctuation/whitespace residue repair for one output line.
fn repair_line(ln: &str) -> String {
    let mut urls: Vec<String> = Vec::new();
    let mut s = ln.to_string();
    if s.contains("://") {
        s = fre_sub(fre(r"\S*://\S*"), &s, |c| {
            urls.push(c.get(0).map(|m| m.as_str()).unwrap_or("").to_string());
            format!("\x00{}\x00", urls.len() - 1)
        });
    }
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
    if s.contains('\x00') {
        return fre_sub(fre(r"\x00(\d+)\x00"), &s, |c| {
            let idx: usize = c.get(1).map(|m| m.as_str().parse().unwrap_or(0)).unwrap_or(0);
            urls.get(idx).cloned().unwrap_or_default()
        });
    }
    s
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

/// The final cleanup pass: entity decode, nbsp/feff collapse, edit-link
/// residue removal, blank normalization, per-line repairs and list
/// rejoining.
pub(crate) fn cleanup(md: &str) -> String {
    let md = html_unescape(md);
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
    let md = fre_sub(fre(r"(?<=[a-z0-9\)])\.([A-Z])"), &out.join("\n"), |c| {
        format!(". {}", c.get(1).map(|m| m.as_str()).unwrap_or(""))
    });
    format!("{}\n", md.trim_matches('\n'))
}
