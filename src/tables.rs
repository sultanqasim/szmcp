//! Table rendering: the port of wikizim_parser/html2md.py's table section
//! (pipe/HTML tables, wrapper tables with nested tables, and the multicol
//! layout-table flattening).

use crate::htmldom::{NodeKind, NodeRef};
use crate::html2md::{
    block_md, escape_plain_asterisks, inline_raw, inline_text, is_dropped, is_infobox_table,
    is_infobox_wrapper, list_item_lines, render_dl, renderable_children,
};
use crate::util::{collapse_ws, re};

pub(crate) fn is_infobox_container(el: NodeRef) -> bool {
    is_infobox_table(el) || is_infobox_wrapper(el)
}

/// Nearest enclosing <table> ancestor of `el`, or None.
fn nearest_table(el: NodeRef) -> Option<NodeRef> {
    let mut anc = el.parent();
    while let Some(a) = anc {
        if a.tag() == Some("table") {
            return Some(a);
        }
        anc = a.parent();
    }
    None
}

/// Rows of `tbl` itself (rows belonging to nested tables excluded).
pub(crate) fn own_table_rows(tbl: NodeRef) -> Vec<NodeRef> {
    tbl.find_all("tr")
        .into_iter()
        .filter(|tr| nearest_table(*tr).map_or(false, |t| t.id() == tbl.id()))
        .collect()
}

/// Direct <th>/<td> children of a row (in document order).
pub(crate) fn row_cells(tr: NodeRef) -> Vec<NodeRef> {
    tr.element_children()
        .filter(|c| matches!(c.tag(), Some("th") | Some("td")))
        .collect()
}

/// Integer value of a rowspan/colspan attribute, or None.
pub(crate) fn span_attr(el: NodeRef, name: &str) -> Option<u32> {
    let v = el.attr(name)?;
    let start = v.find(|c: char| c.is_ascii_digit())?;
    let digits: &str = &v[start..];
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    digits[..end].parse().ok()
}

/// Top-level nested tables of `tbl`: descendant <table> elements whose
/// nearest enclosing table is `tbl` and whose ancestor chain up to `tbl`
/// survives the drop rules.
fn nested_tables(tbl: NodeRef) -> Vec<NodeRef> {
    let mut out = Vec::new();
    'outer: for t in tbl.find_all("table") {
        if is_dropped(t) {
            continue;
        }
        let mut anc = t.parent();
        while let Some(a) = anc {
            if a.id() == tbl.id() {
                out.push(t);
                continue 'outer;
            }
            if !a.is_element() || is_dropped(a) {
                continue 'outer;
            }
            if a.tag() == Some("table") {
                continue 'outer; // one nesting level deeper
            }
            anc = a.parent();
        }
    }
    out
}

/// Rendered text of the table's <caption> element ('' when none).
/// `keep_br=true` — the HTML-table path, consistent with its <br>-keeping
/// cells — renders the caption's <br> verbatim; `keep_br=false` — the
/// single-line pipe-table path — degrades it to a space. (The Python
/// original passed this bool as `_inline_raw`'s `br_mode` string, where
/// neither True nor False equals "keep", so both paths always degraded
/// <br> to a space.)
fn table_caption_text(tbl: NodeRef, keep_br: bool) -> String {
    for ch in tbl.element_children() {
        if ch.tag() == Some("caption") {
            return collapse_ws(&inline_raw(ch, keep_br)).trim().to_string();
        }
    }
    String::new()
}

/// One cell's content for the HTML-table output: content rendered inline
/// with <br> kept verbatim; list content becomes newline-separated
/// `* item` / `1. item` lines — a lone list item with no other content
/// renders as plain text, leading prose is joined with a single space.
fn cell_html_text(cell: NodeRef) -> String {
    let mut prose: Vec<String> = Vec::new();
    let mut items: Vec<String> = Vec::new();
    for ch in cell.children() {
        match ch.kind() {
            // the leading text or a tail consumed at its owner
            NodeKind::Text(t) => {
                if !t.is_empty() {
                    prose.push(t.to_string());
                }
            }
            NodeKind::Element { .. } => {
                if is_dropped(ch) {
                    continue;
                }
                match ch.tag() {
                    Some("ul") | Some("ol") => {
                        items.extend(list_item_lines(ch, 0, true));
                    }
                    Some("dl") => items.extend(
                        render_dl(ch, 0)
                            .split('\n')
                            .filter(|l| !l.trim().is_empty())
                            .map(String::from),
                    ),
                    _ => prose.push(inline_raw(ch, true)),
                }
            }
            _ => {}
        }
    }
    let lead = collapse_ws(&prose.join("")).trim().to_string();
    if items.is_empty() {
        return lead;
    }
    if items.len() == 1 && lead.is_empty() {
        // a lone list item renders as plain text
        return re(r"^\s*(?:\* |1\. )")
            .replace_all(&items[0], "")
            .trim()
            .to_string();
    }
    let mut lines: Vec<String> = Vec::new();
    if !lead.is_empty() {
        lines.push(lead);
    }
    lines.extend(items);
    lines.join("\n")
}

/// A table is complex (HTML output) when any cell carries a multi-cell
/// rowspan/colspan or contains a list. lxml's `find("ul")` matches direct
/// children only — a nested list inside a span does not count.
fn is_complex_table(rows: &[NodeRef]) -> bool {
    for tr in rows {
        for c in row_cells(*tr) {
            if span_attr(c, "rowspan").unwrap_or(1) > 1 || span_attr(c, "colspan").unwrap_or(1) > 1 {
                return true;
            }
            if c.element_children()
                .any(|ch| matches!(ch.tag(), Some("ul") | Some("ol") | Some("dl")))
            {
                return true;
            }
        }
    }
    false
}

/// Render parsed rows as an HTML table: spans preserved as rowspan=/
/// colspan= attributes, per-cell <th>/<td>, optional caption right after
/// <table>; all attribute debris stripped.
fn table_to_html(rows: &[NodeRef], caption: &str, texts: &[Vec<String>]) -> String {
    let mut out = vec!["<table>".to_string()];
    if !caption.is_empty() {
        out.push(format!("<caption>{}</caption>", caption));
    }
    for (row_texts, tr) in texts.iter().zip(rows.iter()) {
        let mut cells: Vec<String> = Vec::new();
        for (c, text) in row_cells(*tr).into_iter().zip(row_texts.iter()) {
            let tag = if c.tag() == Some("th") { "th" } else { "td" };
            let mut attrs = String::new();
            if let Some(rs) = span_attr(c, "rowspan").filter(|&v| v > 1) {
                attrs.push_str(&format!(" rowspan=\"{}\"", rs));
            }
            if let Some(cs) = span_attr(c, "colspan").filter(|&v| v > 1) {
                attrs.push_str(&format!(" colspan=\"{}\"", cs));
            }
            cells.push(format!("<{}{}>{}</{}>", tag, attrs, text, tag));
        }
        out.push(format!("<tr>{}</tr>", cells.join("")));
    }
    out.push("</table>".to_string());
    out.join("\n")
}

/// Render a simple table as a Markdown pipe table: header from the first
/// row containing header cells (else the first row), body rows padded to
/// the same column count; a non-empty caption becomes an italic line
/// above.
fn table_to_pipe(rows: &[NodeRef], caption: &str, texts: &[Vec<String>]) -> String {
    let header_idx = rows
        .iter()
        .enumerate()
        .find(|(_, tr)| row_cells(**tr).iter().any(|c| c.tag() == Some("th")))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let header = &texts[header_idx];
    let body = &texts[header_idx + 1..];
    let num_cols = header
        .len()
        .max(body.iter().map(|r| r.len()).max().unwrap_or(0))
        .max(1);
    let pad = |cells: &[String]| -> String {
        let mut padded: Vec<String> = cells.to_vec();
        padded.resize(num_cols, String::new());
        format!("| {} |", padded.join(" | "))
    };
    let mut lines = vec![pad(header), format!("|{}", " --- |".repeat(num_cols))];
    for r in body {
        lines.push(pad(r));
    }
    let mut md = lines.join("\n");
    let mut caption = caption.to_string();
    if !caption.is_empty() && !caption.contains('*') {
        caption = format!("*{}*", caption); // plain caption -> italic
    }
    if !caption.is_empty() {
        md = format!("{}\n\n{}", caption, md);
    }
    md
}

/// Rendered Markdown of a wrapper's nested tables (blank-line joined).
fn nested_tables_md(nested: &[NodeRef]) -> String {
    let mut parts = Vec::new();
    for t in nested {
        let md = render_table(*t);
        if !md.trim().is_empty() {
            parts.push(md);
        }
    }
    parts.join("\n\n")
}

// ---------------------------------------------------------------------------
// Multicol layout tables ({{col-begin}} / {{col-break}} template output)
// ---------------------------------------------------------------------------

fn is_multicol_col_class(cls: &str) -> bool {
    cls == "col-break" || cls.starts_with("col-break-")
}

/// True for a MediaWiki multicol layout table ({{col-begin}} output).
fn is_multicol_table(tbl: NodeRef) -> bool {
    if tbl.has_any_class(&["col-begin", "col-begin-small"]) {
        return true;
    }
    // Defensive fallback for parsers that emit the cells' col-break classes
    // without the template's role.
    if tbl.attr("role") == Some("presentation") {
        for c in tbl.find_all("td") {
            if c.class_tokens().any(is_multicol_col_class) {
                return true;
            }
        }
    }
    false
}

/// Bold lines for a <dl> that carries ONLY bare terms (the ';Term'
/// pseudo-heading convention); None when any <dd> carries content.
fn dl_bare_term_lines(dl: NodeRef) -> Option<Vec<String>> {
    let mut had_term = false;
    for ch in dl.children() {
        if !ch.is_element() || is_dropped(ch) {
            continue;
        }
        match ch.tag() {
            Some("dt") => had_term = true,
            Some("dd") => {
                if !inline_text(ch).trim().is_empty() {
                    return None; // a real definition: standard dl rendering
                }
            }
            _ => return None, // nested list/spans between terms: standard
        }
    }
    if !had_term {
        return None;
    }
    let mut lines = Vec::new();
    for ch in dl.children() {
        if ch.tag() != Some("dt") {
            continue;
        }
        let term = inline_text(ch).trim().to_string();
        if !term.is_empty() {
            lines.push(format!("**{}**", term));
        }
    }
    if lines.is_empty() { None } else { Some(lines) }
}

/// One multicol table cell as block Markdown: the cell is a plain block
/// container rendered through the ordinary block dispatch, with the one
/// multicol-specific rule that bare-term <dl>s render as bold standalone
/// lines.
fn cell_blocks_md(cell: NodeRef) -> String {
    let mut out: Vec<String> = Vec::new();
    let add_prose = |txt: Option<&str>, out: &mut Vec<String>| {
        let t = txt.map(|t| collapse_ws(&escape_plain_asterisks(t))).unwrap_or_default();
        let t = t.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    };
    add_prose(cell.text(), &mut out);
    for ch in renderable_children(cell) {
        if ch.tag() == Some("dl") {
            if let Some(lines) = dl_bare_term_lines(ch) {
                out.extend(lines);
                add_prose(ch.tail(), &mut out);
                continue;
            }
        }
        let md = block_md(ch, false);
        if !md.is_empty() {
            out.push(md);
        }
        add_prose(ch.tail(), &mut out);
    }
    out.join("\n\n")
}

/// Flatten a multicol layout table into the block flow: each cell's
/// content rendered as ordinary block Markdown, cells in document order.
fn render_multicol(tbl: NodeRef) -> String {
    let mut blocks = Vec::new();
    for tr in own_table_rows(tbl) {
        for c in row_cells(tr) {
            let md = cell_blocks_md(c);
            if !md.trim().is_empty() {
                blocks.push(md);
            }
        }
    }
    blocks.join("\n\n")
}

/// Render one <table> element to Markdown. Multicol layout tables flatten
/// to ordinary block Markdown; complex tables (rowspan/colspan > 1 or
/// lists in cells) become HTML tables, simple ones pipe tables. Only a
/// table with no renderable content at all is dropped (every cell empty,
/// or a bare <br> in the HTML path — the residue of stripped images); a
/// simple table also drops blank rows, while a complex one keeps them
/// (blank rowspan/colspan rows are the skeleton of large layout grids).
/// A table left without any row yields just its nested tables'
/// renderings; everything else renders, and a wrapper table renders its
/// own rows followed by its nested tables.
pub(crate) fn render_table(tbl: NodeRef) -> String {
    if tbl.tag() != Some("table") {
        return String::new();
    }
    if is_multicol_table(tbl) {
        return render_multicol(tbl);
    }
    let rows: Vec<NodeRef> = own_table_rows(tbl)
        .into_iter()
        .filter(|tr| !row_cells(*tr).is_empty())
        .collect();
    let nested_md = nested_tables_md(&nested_tables(tbl));
    let as_html = is_complex_table(&rows);
    // each row's rendered cell text, computed once for the renderers and
    // the blank-row rule
    let texts: Vec<Vec<String>> = rows
        .iter()
        .map(|tr| {
            row_cells(*tr)
                .into_iter()
                .map(|c| {
                    if as_html {
                        cell_html_text(c)
                    } else {
                        collapse_ws(&inline_text(c)).trim().to_string()
                    }
                })
                .collect()
        })
        .collect();
    // A simple table drops blank rows (the residue of stripped images);
    // a complex one keeps them. Either way a table left with no rows —
    // or an HTML table whose every cell is blank — has no renderable
    // content and yields just its nested tables' renderings.
    let (rows, texts): (Vec<NodeRef>, Vec<Vec<String>>) = rows
        .into_iter()
        .zip(texts)
        .filter(|(_, row)| as_html || row.iter().any(|t| !t.is_empty()))
        .unzip();
    if rows.is_empty()
        || (as_html
            && texts
                .iter()
                .all(|row| row.iter().all(|t| t.replace("<br>", "").trim().is_empty())))
    {
        return nested_md;
    }
    let caption = table_caption_text(tbl, as_html);
    let md = if as_html {
        table_to_html(&rows, &caption, &texts)
    } else {
        table_to_pipe(&rows, &caption, &texts)
    };
    if nested_md.is_empty() {
        md
    } else {
        format!("{}\n\n{}", md, nested_md)
    }
}