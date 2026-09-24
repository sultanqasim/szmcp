//! Infobox extraction for ZIM article HTML -> '## Key facts' Markdown
//! (the Rust port of wikizim_parser/infobox_html.py).

use std::collections::HashSet;

use crate::cleanup;
use crate::htmldom::{Dom, NodeId, NodeRef};
use crate::html2md::{escape_plain_asterisks, is_dropped, render_inline_default, render_list};
use crate::tables::{own_table_rows, render_table, row_cells, top_level_tables};
use crate::util::{collapse_ws, re};
use crate::wikil10n;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const CAPTION_CLASSES: &[&str] = &[
    "infobox-caption",
    "ib-settlement-caption",
    "ib-settlement-caption-link",
    "thumbcaption",
];

/// Class tokens marking map/media container structures.
const MAP_CLASSES: &[&str] = &[
    "maptable", "locmap", "switcher-container", "ib-settlement-cols",
    "multiimageinner", "thumbinner", "legend", "legend-color",
];

/// Class tokens of MediaWiki's collapse-widget markup.
const TOGGLE_CLASS_TOKENS: &[&str] = &["mw-collapsible-toggle", "navbox-toggle"];

const EMPH_TAGS: &[&str] = &["b", "strong", "i", "em"];

/// Class tokens marking an embedded table as office-group structured.
const INFOBOX_ROW_CLASSES: &[&str] = &[
    "infobox-above", "infobox-subheader", "infobox-header", "infobox-label",
    "infobox-full-data", "infobox-below",
];

fn is_media_el(el: NodeRef) -> bool {
    if el
        .attr("typeof")
        .unwrap_or("")
        .split_whitespace()
        .any(|t| t == "mw:File")
    {
        return true;
    }
    matches!(el.tag(), Some("img") | Some("figure") | Some("figcaption") | Some("audio") | Some("video"))
}

/// Content excluded from visible-text walks: dropped elements, media
/// elements and media captions.
fn is_ignored(el: NodeRef) -> bool {
    is_dropped(el) || is_media_el(el) || el.has_any_class(CAPTION_CLASSES)
}

/// Whitespace-collapsed visible text with dropped elements, media elements
/// and media captions removed. Used for labels and emptiness tests.
fn visible_text(el: NodeRef) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(t) = el.text() {
        parts.push(t.to_string());
    }
    for ch in el.children() {
        if ch.is_text() {
            continue; // the leading text or a tail consumed at its owner
        }
        if !ch.is_element() || is_ignored(ch) {
            // comments contribute only their tail
            if let Some(t) = ch.tail() {
                parts.push(t.to_string());
            }
            continue;
        }
        parts.push(visible_text(ch));
        if let Some(t) = ch.tail() {
            parts.push(t.to_string());
        }
    }
    collapse_ws(&parts.join("")).trim().to_string()
}

/// A collapse-widget link: an <a> whose whole label is 'more...' /
/// 'less...' (optionally bracket- or parenthesis-wrapped).
fn is_toggle_anchor(el: NodeRef) -> bool {
    if el.tag() != Some("a") {
        return false;
    }
    let txt = collapse_ws(&el.text_content()).trim().to_string();
    !txt.is_empty()
        && re(r"(?i)^\(?\[?(?:more|less)(?:\.\.\.|…)?\)?\]?$").is_match(&txt)
}

/// A collapse-widget element: a classed toggle span or a
/// 'more...'/'less...' label link.
fn is_toggle_el(el: NodeRef) -> bool {
    if el.tag() == Some("a") {
        return is_toggle_anchor(el);
    }
    el.has_any_class(TOGGLE_CLASS_TOKENS)
}

/// What may be left once toggle content is gone: brackets/parentheses and
/// stray zero-width characters.
fn is_residue_only(s: &str) -> bool {
    s.chars().all(|c| {
        c.is_whitespace()
            || matches!(c, '(' | ')' | '[' | ']' | '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}')
    })
}

/// Remove collapse-widget elements from the standalone `dom` copy (the
/// copy of a label/value cell), pruning wrappers left holding nothing
/// visible.
fn strip_toggles(dom: &mut Dom) {
    let root = dom.root().id();
    for tgl in element_ids(dom, is_toggle_el) {
        let Some(par) = dom.parent_of(tgl) else { continue };
        dom.detach(tgl);
        dom.merge_text(par);
        let mut par = par;
        while par != root {
            let Some(nxt) = dom.parent_of(par) else { break };
            if !is_residue_only(&visible_text(dom.ref_(par))) {
                break; // the wrapper still carries real content: keep it
            }
            dom.detach(par);
            dom.merge_text(nxt);
            par = nxt;
        }
    }
}

/// Ids of the `dom`'s elements (document order) matching `pred`.
fn element_ids(dom: &Dom, pred: impl Fn(NodeRef) -> bool) -> Vec<NodeId> {
    dom.ref_(dom.root().id())
        .self_and_descendants()
        .filter(|e| e.is_element() && pred(*e))
        .map(|e| e.id())
        .collect()
}

/// True when `el` carries inline emphasis or a collapse-widget element.
fn needs_plain_copy(el: NodeRef) -> bool {
    el.self_and_descendants()
        .any(|e| {
            if !e.is_element() {
                return false;
            }
            let tag = e.tag().unwrap_or("");
            if EMPH_TAGS.contains(&tag) || e.has_any_class(TOGGLE_CLASS_TOKENS) {
                return true;
            }
            tag == "a" && is_toggle_anchor(e)
        })
}

/// Unwrap all <b>/<strong>/<i>/<em> emphasis in the standalone copy.
fn unwrap_emphasis(dom: &mut Dom) {
    let emph = element_ids(dom, |e| EMPH_TAGS.contains(&e.tag().unwrap_or("")));
    for id in emph {
        dom.drop_tag(id); // no-op when detached (removed with a toggle)
    }
}

/// A standalone copy of `el` with collapse widgets removed and emphasis
/// unwrapped (the Key facts no-emphasis rule).
fn plain_copy(el: NodeRef) -> Dom {
    let mut copy = el.dom.copy_subtree(el.id());
    strip_toggles(&mut copy);
    unwrap_emphasis(&mut copy);
    copy
}

/// Render a Key facts label/value cell with ALL inline emphasis stripped
/// and collapse-widget text filtered.
fn plain_inline(el: NodeRef) -> String {
    if !needs_plain_copy(el) {
        return render_inline_default(el);
    }
    let copy = plain_copy(el);
    render_inline_default(copy.ref_(copy.root().id()))
}

/// <ul>/<ol> elements of a value cell whose parent chain carries no other
/// list element (the top-level lists of a plainlist value).
fn top_lists<'a>(cell: NodeRef<'a>) -> Vec<NodeRef<'a>> {
    let mut out = Vec::new();
    for el in cell.descendants() {
        if !matches!(el.tag(), Some("ul") | Some("ol")) {
            continue;
        }
        // nested when an ancestor below `cell` is itself list-ish
        let nested = el
            .ancestors()
            .take_while(|p| p.id() != cell.id())
            .any(|p| {
                p.is_element()
                    && matches!(
                        p.tag(),
                        Some("ul") | Some("ol") | Some("li") | Some("dl") | Some("dd") | Some("dt")
                    )
            });
        if !nested {
            out.push(el);
        }
    }
    out
}

/// A full-data row containing a nested table or a map/media structure.
fn is_map_row(cell: NodeRef) -> bool {
    cell.descendants()
        .any(|el| el.tag() == Some("table") || el.has_any_class(MAP_CLASSES))
}

/// A nested table built from ordinary infobox rows (>= 2 rows whose cells
/// carry infobox row classes) — the collapsible office-group sub-tables.
fn is_office_table(tbl: NodeRef) -> bool {
    let mut hits = 0;
    for tr in own_table_rows(tbl) {
        if row_cells(tr).iter().any(|c| c.has_any_class(INFOBOX_ROW_CLASSES)) {
            hits += 1;
            if hits >= 2 {
                return true;
            }
        }
    }
    false
}

/// True when `tbl` and every element between it and the document root
/// survives the drop rules.
fn drop_free(tbl: NodeRef) -> bool {
    !is_dropped(tbl) && tbl.ancestors().all(|a| !is_dropped(a))
}

/// Visible text of `cell` that precedes the embedded table `stop`.
fn leading_text_before(cell: NodeRef, stop: NodeId) -> String {
    fn walk(el: NodeRef, stop: NodeId, parts: &mut Vec<String>) -> bool {
        if let Some(t) = el.text() {
            parts.push(t.to_string());
        }
        for ch in el.children() {
            if ch.is_text() {
                continue; // tails handled at their owners
            }
            if ch.id() == stop {
                return false;
            }
            if !ch.is_element() || is_ignored(ch) {
                if let Some(t) = ch.tail() {
                    parts.push(t.to_string());
                }
                continue;
            }
            if !walk(ch, stop, parts) {
                return false;
            }
            if let Some(t) = ch.tail() {
                parts.push(t.to_string());
            }
        }
        true
    }
    let mut parts = Vec::new();
    walk(cell, stop, &mut parts);
    collapse_ws(&parts.join("")).trim().to_string()
}

/// Document-order visible text of `cell` as (text, in_bold, after_first_br)
/// triples plus whether the cell contains a <br>.
fn bold_parts(cell: NodeRef) -> (Vec<(String, bool, bool)>, bool) {
    let mut parts: Vec<(String, bool, bool)> = Vec::new();
    let mut saw_br = false;
    fn walk(el: NodeRef, in_bold: bool, saw_br: &mut bool, parts: &mut Vec<(String, bool, bool)>) {
        let Some(tag) = el.tag() else { return };
        if tag == "br" {
            *saw_br = true;
            return;
        }
        if is_ignored(el) {
            return;
        }
        let bold = in_bold || tag == "b";
        let after = *saw_br;
        if let Some(t) = el.text() {
            if !t.trim().is_empty() {
                parts.push((t.to_string(), bold, after));
            }
        }
        for ch in el.children() {
            if ch.is_text() {
                continue;
            }
            if ch.is_element() {
                walk(ch, bold, saw_br, parts);
            }
            if let Some(t) = ch.tail() {
                if !t.trim().is_empty() {
                    parts.push((t.to_string(), bold, *saw_br));
                }
            }
        }
    }
    if cell.is_element() {
        walk(cell, false, &mut saw_br, &mut parts);
    }
    (parts, saw_br)
}

/// A full-width term/status row of an infobox-full-data cell: a non-empty
/// bold lead before a <br> or a wholly bold cell.
fn is_context_row(cell: NodeRef) -> bool {
    if !cell.has_class("infobox-full-data") {
        return false;
    }
    let (parts, saw_br) = bold_parts(cell);
    if parts.is_empty() {
        return false;
    }
    if parts.iter().all(|(_, b, _)| *b) {
        return true;
    }
    let lead: Vec<&(String, bool, bool)> =
        parts.iter().filter(|(_, _, after)| !*after).collect();
    if saw_br && !lead.is_empty() && lead.iter().all(|(_, b, _)| *b) {
        let lead_txt = collapse_ws(
            &lead.iter().map(|(t, _, _)| t.as_str()).collect::<String>(),
        )
        .trim()
        .to_string();
        return !lead_txt.is_empty();
    }
    false
}

// ---------------------------------------------------------------------------
// Parsed structure
// ---------------------------------------------------------------------------

/// A group heading: its level and eagerly computed heading text (element
/// heading cells render through the inline path; scaffold titles are
/// asterisk-escaped plain text). The tree is never mutated between
/// extraction and rendering, so the text can be computed upfront.
#[derive(Clone)]
pub(crate) struct Header {
    level: u32,
    text: String,
}

#[derive(Clone, Default)]
pub(crate) struct Row {
    label_cell: Option<NodeId>,
    value_cells: Vec<NodeId>,
    is_sub: bool,
    split_label: bool,
    is_context: bool,
    list_els: Vec<NodeId>,
}

#[derive(Clone)]
pub(crate) enum Item {
    Header(Header),
    Row(Row),
    /// A pre-rendered Markdown table block: an embedded non-infobox data
    /// table.
    RawTable(String),
    /// A full-width header row with no text: the separator idiom that
    /// closes a section.
    GroupEnd,
}

pub(crate) struct Infobox {
    title: String,
    items: Vec<Item>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Heading for a box in a multi-box page: the above/entete text when
/// present, else a class name (ib-settlement -> Settlement), else the
/// fallback.
fn subbox_title(tbl: NodeRef, above_text: &str, fallback: &str) -> String {
    if !above_text.is_empty() {
        return above_text.to_string();
    }
    for tok in tbl.class_tokens() {
        if let Some(rest) = tok.strip_prefix("ib-") {
            if !rest.is_empty() {
                let rest = rest.replace('_', "-");
                return rest
                    .split('-')
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                            None => String::new(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
            }
        }
    }
    fallback.to_string()
}

/// Build a Row from a label/data row, or None when it has no label.
fn fact_row(label_cell: NodeRef, data_cells: &[NodeRef]) -> Option<Row> {
    let mut label_txt = visible_text(label_cell);
    if label_txt.is_empty() {
        return None;
    }
    let mut is_sub = false;
    if let Some(rest) = label_txt.strip_prefix('•') {
        is_sub = true;
        label_txt = rest.trim().to_string();
    }
    if label_txt.trim().is_empty() {
        return None;
    }
    let mut lists = Vec::new();
    for td in data_cells {
        lists.extend(top_lists(*td).iter().map(|l| l.id()));
    }
    Some(Row {
        label_cell: Some(label_cell.id()),
        value_cells: data_cells.iter().map(|c| c.id()).collect(),
        is_sub,
        list_els: lists,
        ..Default::default()
    })
}

/// Interpret one full-width cell, or None when it is media/caption-only.
fn full_data_row(cell: NodeRef) -> Option<Row> {
    if is_map_row(cell) {
        return None;
    }
    if cell
        .descendants()
        .any(|el| el.is_element() && is_media_el(el))
    {
        return None;
    }
    if visible_text(cell).is_empty() {
        return None;
    }
    Some(Row {
        value_cells: vec![cell.id()],
        split_label: true,
        ..Default::default()
    })
}

/// Rows from the frwiki taxobox_v3 idiom of label/value content outside
/// the box's classification table (<p class="bloc">Label</p> + value
/// siblings).
fn v3_bloc_rows(wrapper: NodeRef) -> Vec<Row> {
    let kids: Vec<NodeRef> = wrapper.element_children().collect();
    let mut rows = Vec::new();
    let mut i = 0;
    while i < kids.len() {
        let el = kids[i];
        if el.tag() == Some("p") && el.has_class("bloc") && !visible_text(el).is_empty() {
            let mut value_cells: Vec<NodeId> = Vec::new();
            let mut j = i + 1;
            while j < kids.len() {
                let nxt = kids[j];
                if nxt.tag() == Some("p") && nxt.has_class("bloc") {
                    break; // the next label: this one has no more value
                }
                if !visible_text(nxt).is_empty() {
                    value_cells.push(nxt.id());
                }
                j += 1;
            }
            if !value_cells.is_empty() {
                let mut lists = Vec::new();
                for vc in &value_cells {
                    lists.extend(top_lists(el.dom.ref_(*vc)).iter().map(|l| l.id()));
                }
                rows.push(Row {
                    label_cell: Some(el.id()),
                    value_cells,
                    list_els: lists,
                    ..Default::default()
                });
            }
            i = j;
            continue;
        }
        i += 1;
    }
    rows
}

/// Cells of `rows[i + 1]`, or None (used for header/value lookahead).
fn next_row_cells<'a>(rows: &'a [NodeRef<'a>], i: usize) -> Option<Vec<NodeRef<'a>>> {
    let cells = row_cells(*rows.get(i + 1)?);
    if cells.is_empty() { None } else { Some(cells) }
}

/// True when `nxt` is ONE non-empty td value cell (not a context row):
/// a header followed by such a cell labels that value.
fn is_single_value(nc: &Option<Vec<NodeRef>>) -> bool {
    matches!(
        nc,
        Some(nc)
            if nc.len() == 1
                && nc[0].tag() == Some("td")
                && !visible_text(nc[0]).is_empty()
                && !is_context_row(nc[0])
    )
}

/// The title signal of an embedded office-group sub-table: a leading bare
/// <th> row or a <caption>. Returns (skip_row, title text).
fn subtable_title(tbl: NodeRef) -> (Option<NodeId>, Option<String>) {
    let rows = own_table_rows(tbl);
    if let Some(first) = rows.first() {
        let cells = row_cells(*first);
        if cells.len() == 1
            && cells[0].tag() == Some("th")
            && cells[0].class_tokens().next().is_none()
            && !visible_text(cells[0]).is_empty()
        {
            let nxt = next_row_cells(&rows, 0);
            if !is_single_value(&nxt) {
                return (Some(first.id()), Some(header_text(cells[0])));
            }
        }
    }
    for ch in tbl.element_children() {
        if ch.tag() == Some("caption") && !visible_text(ch).is_empty() {
            return (None, Some(header_text(ch)));
        }
    }
    (None, None)
}

/// Collapsed inline text of a group heading cell.
fn header_text(el: NodeRef) -> String {
    collapse_ws(&render_inline_default(el)).trim().to_string()
}

/// Dispatch the rows of one infobox-structured table into `items`.
/// Returns the box's above-row text ('' when absent).
fn collect_rows(
    rows: &[NodeRef],
    items: &mut Vec<Item>,
    fallback_title: &str,
    top: bool,
    header_level: u32,
) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let mut above_text = String::new();
    let mut skip_next = false;
    for (i, tr) in rows.iter().enumerate() {
        let cells = row_cells(*tr);
        if std::mem::take(&mut skip_next) || cells.is_empty() {
            continue;
        }
        let c0 = cells[0];
        if c0.has_class("infobox-above") {
            if top && above_text.is_empty() {
                above_text = visible_text(c0);
            }
            continue;
        }
        if c0.has_any_class(&["infobox-subheader", "infobox-below", "infobox-caption"]) {
            continue;
        }
        if c0.tag() == Some("th") && cells.len() == 1 {
            // Single full-width header cell: a group heading, except in
            // two cases.
            let t = visible_text(c0);
            if t.is_empty() {
                // An empty full-width header row closes the section.
                items.push(Item::GroupEnd);
                continue;
            }
            if !fallback_title.is_empty() && t.starts_with(fallback_title) {
                if top && above_text.is_empty() {
                    above_text = t;
                }
                continue;
            }
            // Data tables embedded in the header cell render as tables.
            let mut embedded_md: Vec<String> = Vec::new();
            for st in top_level_tables(c0, |_| true) {
                if !is_office_table(st) && drop_free(st) {
                    let md = render_table(st);
                    if !md.is_empty() {
                        embedded_md.push(md);
                    }
                }
            }
            let nxt = next_row_cells(rows, i);
            if is_single_value(&nxt) && embedded_md.is_empty() {
                // A header immediately followed by ONE non-empty value cell
                // labels that value rather than grouping sub-rows.
                let value = nxt.unwrap()[0];
                items.push(Item::Row(Row {
                    label_cell: Some(c0.id()),
                    value_cells: vec![value.id()],
                    list_els: top_lists(value).iter().map(|l| l.id()).collect(),
                    ..Default::default()
                }));
                skip_next = true;
                continue;
            }
            items.push(Item::Header(Header { level: header_level, text: header_text(c0) }));
            for md in embedded_md {
                items.push(Item::RawTable(md));
            }
            continue;
        }
        if c0.has_class("infobox-label") || (c0.tag() == Some("th") && cells.len() >= 2) {
            if let Some(row) = fact_row(c0, &cells[1..]) {
                items.push(Item::Row(row));
            }
            continue;
        }
        if c0.has_class("infobox-full-data") || (c0.tag() == Some("td") && cells.len() == 1) {
            // Full-width cell. An embedded office-group table flattens into
            // the same item stream; its title (a leading bare <th>, a
            // <caption>, or the cell text ahead of it) becomes a parent
            // heading with the group headers demoted one level.
            let embedded: Vec<NodeRef> = top_level_tables(c0, |_| true)
                .into_iter()
                .filter(|st| is_office_table(*st))
                .collect();
            if !embedded.is_empty() {
                for (k, st) in embedded.iter().enumerate() {
                    let (skip_tr, mut title) = subtable_title(*st);
                    if title.is_none() && k == 0 {
                        // the cell text ahead of the first sub-table
                        let lead = leading_text_before(c0, st.id());
                        if !lead.is_empty() {
                            title = Some(collapse_ws(&escape_plain_asterisks(&lead)).trim().to_string());
                        }
                    }
                    match title {
                        None => {
                            let own = own_table_rows(*st);
                            collect_rows(&own, items, fallback_title, false, header_level);
                        }
                        Some(text) => {
                            items.push(Item::Header(Header { level: header_level, text }));
                            let own: Vec<NodeRef> = own_table_rows(*st)
                                .into_iter()
                                .filter(|tr| Some(tr.id()) != skip_tr)
                                .collect();
                            collect_rows(&own, items, fallback_title, false, header_level + 1);
                        }
                    }
                }
                continue;
            }
            if let Some(mut row) = full_data_row(c0) {
                row.is_context = is_context_row(c0);
                items.push(Item::Row(row));
            }
            continue;
        }
        // unknown row layout: try label/data, then full-data, else skip
        let row = if cells.len() >= 2 { fact_row(c0, &cells[1..]) } else { full_data_row(c0) };
        if let Some(row) = row {
            items.push(Item::Row(row));
        }
    }
    above_text
}

/// Parse one infobox table. `above_text` stands in for a missing
/// infobox-above row as the multi-box sub-heading.
fn parse_infobox_table(tbl: NodeRef, fallback_title: &str, above_text: &str) -> Infobox {
    let mut items = Vec::new();
    let rows = own_table_rows(tbl);
    let row_above_text = collect_rows(&rows, &mut items, fallback_title, true, 3);
    Infobox {
        title: subbox_title(
            tbl,
            if row_above_text.is_empty() { above_text } else { &row_above_text },
            fallback_title,
        ),
        items,
    }
}

/// Return the infobox objects of a full ZIM page, in document order.
/// Containers nested inside another container are excluded.
pub(crate) fn extract_infoboxes(dom: &Dom) -> Vec<Infobox> {
    let root = dom.root();
    // One pre-order pass gathers the article title (the h1#firstHeading
    // text, else the first h1's — a last-resort subbox title) and the
    // infobox containers, in document order: classed <table class="infobox">
    // and the frwiki wrapper <div class="infobox_v2|v3 infobox">.
    let mut fallback_title = String::new();
    let mut title_set = false;
    let mut containers: Vec<NodeRef> = Vec::new();
    for el in root.self_and_descendants() {
        if !title_set && el.tag() == Some("h1") {
            if el.attr("id") == Some("firstHeading") {
                title_set = true;
            }
            if title_set || fallback_title.is_empty() {
                fallback_title = collapse_ws(&el.text_content()).trim().to_string();
            }
        }
        if el.is_element()
            && matches!(el.tag(), Some("table") | Some("div"))
            && el.has_class("infobox")
        {
            containers.push(el);
        }
    }
    if containers.is_empty() {
        return Vec::new();
    }
    let container_ids: HashSet<NodeId> = containers.iter().map(|c| c.id()).collect();
    let mut boxes = Vec::new();
    for el in containers {
        // Containers nested inside another container are excluded.
        if el.ancestors().any(|a| container_ids.contains(&a.id())) {
            continue;
        }
        if el.tag() == Some("table") {
            boxes.push(parse_infobox_table(el, &fallback_title, ""));
            continue;
        }
        // frwiki wrapper div: parse each of its top-level tables as a box;
        // the wrapper's 'entete' title div stands in for the above row,
        // and taxobox_v3 bloc label/value pairs join the first box.
        let above = el
            .descendants()
            .filter(|d| d.is_element() && d.has_class("entete"))
            .map(|d| visible_text(d))
            .find(|t| !t.is_empty())
            .unwrap_or_default();
        let bloc_rows = v3_bloc_rows(el);
        let mut top_tables = Vec::new();
        for tbl in el.find_all("table") {
            // a table nested inside a deeper table is not a box table
            let deeper = tbl
                .ancestors()
                .take_while(|p| p.id() != el.id())
                .any(|p| p.tag() == Some("table"));
            if !deeper {
                top_tables.push(tbl);
            }
        }
        if !top_tables.is_empty() {
            for (n, tbl) in top_tables.into_iter().enumerate() {
                let mut b = parse_infobox_table(tbl, &fallback_title, if n == 0 { &above } else { "" });
                if n == 0 {
                    b.items.extend(bloc_rows.iter().cloned().map(Item::Row));
                }
                boxes.push(b);
            }
        } else if !bloc_rows.is_empty() {
            boxes.push(Infobox {
                title: if above.is_empty() { fallback_title.clone() } else { above.clone() },
                items: bloc_rows.into_iter().map(Item::Row).collect(),
            });
        }
    }
    boxes
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Whitespace-collapsed inline text of a row's value cells, with the Key
/// facts no-emphasis rule applied.
fn join_cells(dom: &Dom, cells: &[NodeId]) -> String {
    let joined: Vec<String> = cells.iter().map(|&c| plain_inline(dom.ref_(c))).collect();
    collapse_ws(&joined.join(" ")).trim().to_string()
}

/// Append one '- **label**: value' bullet, nested when indent > 0.
fn append_row(dom: &Dom, lines: &mut Vec<String>, row: &Row, indent: usize) {
    let pad = "  ".repeat(indent);
    if row.is_context {
        // Full-width term/status row: an unlabelled bullet whose bold
        // marker is template styling, not content.
        let rendered = join_cells(dom, &row.value_cells);
        if rendered.is_empty() {
            return;
        }
        let rendered = re(r"\*{2}(.+?)\*{2}")
            .replace_all(&rendered, "$1")
            .trim()
            .to_string();
        if !rendered.is_empty() {
            lines.push(format!("{}- {}", pad, rendered));
        }
        return;
    }
    if row.split_label {
        let rendered = join_cells(dom, &row.value_cells);
        let Some(caps) = re(r"(?s)^([^:\n]{1,40}):\s*(.+)$").captures(&rendered) else {
            return;
        };
        let label = caps.get(1).unwrap().as_str().trim().to_string();
        let value = caps.get(2).unwrap().as_str().trim().to_string();
        if label.is_empty() || value.is_empty() {
            return;
        }
        if label.contains("://") || label.contains("www.") || label.contains("[[") {
            return;
        }
        lines.push(format!("{}- **{}**: {}", pad, label, value));
        return;
    }
    let mut label = row
        .label_cell
        .map(|id| collapse_ws(&plain_inline(dom.ref_(id))).trim().to_string())
        .unwrap_or_default();
    if row.is_sub {
        // the visible bullet marker is layout, not content
        label = label
            .trim_start_matches(|c: char| c.is_whitespace() || c == '•')
            .to_string();
    }
    // labels may carry their own colon; the ': ' separator is added below
    let label = label.trim_end_matches(':').trim().to_string();
    if label.is_empty() {
        return;
    }
    // A plainlist value becomes a nested list.
    if !row.list_els.is_empty() {
        let mut items: Vec<String> = Vec::new();
        for le in &row.list_els {
            let md = if needs_plain_copy(dom.ref_(*le)) {
                let copy = plain_copy(dom.ref_(*le));
                render_list(copy.ref_(copy.root().id()), 0)
            } else {
                render_list(dom.ref_(*le), 0)
            };
            if !md.is_empty() {
                items.extend(md.split('\n').filter(|l| !l.trim().is_empty()).map(String::from));
            }
        }
        if items.is_empty() {
            return;
        }
        lines.push(format!("{}- **{}**:", pad, label));
        for it in items {
            lines.push(format!("{}  {}", pad, it));
        }
        return;
    }
    let rendered = join_cells(dom, &row.value_cells);
    if rendered.is_empty() {
        return;
    }
    lines.push(format!("{}- **{}**: {}", pad, label, rendered));
}

/// Push a blank line unless `lines` is empty or already ends blank.
fn sep_blank_line(lines: &mut Vec<String>) {
    if !lines.is_empty() && lines.last().map(String::as_str) != Some("") {
        lines.push(String::new());
    }
}

/// Append a sub-heading with a blank line before and after.
fn emit_heading(lines: &mut Vec<String>, text: String) {
    sep_blank_line(lines);
    lines.push(text);
    lines.push(String::new());
}

/// Split an item stream into the segments to render, in output order: the
/// top-level facts first, then each named group.
fn group_segments(items: &[Item]) -> Vec<Vec<usize>> {
    let mut segs: Vec<Vec<usize>> = Vec::new();
    let mut top: Option<usize> = None;
    let mut group: Option<usize> = None;
    let mut level = 0u32;
    for (i, item) in items.iter().enumerate() {
        match item {
            Item::GroupEnd => {
                top = None;
                group = None;
            }
            Item::Header(h) => {
                if group.is_none() || h.level <= level {
                    // a new group (or a group after a closed one)
                    segs.push(vec![i]);
                    group = Some(segs.len() - 1);
                    top = None;
                    level = h.level;
                } else {
                    // deeper heading: stays inside the open group
                    segs[group.unwrap()].push(i);
                }
            }
            _ => match group {
                Some(g) => segs[g].push(i),
                None => match top {
                    Some(t) => segs[t].push(i),
                    None => {
                        segs.push(vec![i]);
                        top = Some(segs.len() - 1);
                    }
                },
            },
        }
    }
    segs
}

/// Render one segment of items into `lines`, with infobox_parser's nesting
/// semantics.
fn emit_items(dom: &Dom, lines: &mut Vec<String>, items: &[&Item]) {
    let mut anchor_open = false; // a fact row was emitted; no header since
    let mut context_open = false; // a context bullet is current
    for item in items {
        match item {
            Item::Header(h) => {
                if !h.text.is_empty() {
                    emit_heading(lines, format!("{} {}", "#".repeat(h.level as usize), &h.text));
                }
                anchor_open = false;
                context_open = false;
            }
            Item::RawTable(md) => {
                // an embedded real table renders verbatim between blank
                // lines (never indented: an indented table would be code)
                sep_blank_line(lines);
                lines.extend(md.split('\n').map(String::from));
                lines.push(String::new());
                context_open = false;
            }
            Item::Row(row) => {
                if row.is_context {
                    // A new context stint starts a fresh bullet at group
                    // level; with facts already emitted a blank line
                    // separates the stints.
                    sep_blank_line(lines);
                    let before = lines.len();
                    append_row(dom, lines, row, 0);
                    if lines.len() > before {
                        anchor_open = true;
                        context_open = true;
                    }
                    continue;
                }
                // A '•' sub-row nests under the previous fact row; facts
                // after a context bullet nest one level beneath it.
                let indent = (if context_open { 1 } else { 0 })
                    + (if row.is_sub && anchor_open { 1 } else { 0 });
                let before = lines.len();
                append_row(dom, lines, row, indent);
                if lines.len() > before {
                    anchor_open = true;
                }
            }
            Item::GroupEnd => {}
        }
    }
}

/// Rendered fact lines of one box (headings included, no '## Key facts').
fn box_lines(dom: &Dom, items: &[Item]) -> Vec<String> {
    let mut lines = Vec::new();
    for seg in group_segments(items) {
        let seg_items: Vec<&Item> = seg.iter().map(|&i| &items[i]).collect();
        emit_items(dom, &mut lines, &seg_items);
    }
    lines
}

/// Render parsed infoboxes under a combined '## Key facts' section;
/// returns '' when nothing is renderable.
pub(crate) fn infoboxes_to_markdown(dom: &Dom, boxes: &[Infobox], lang: Option<&str>) -> String {
    let rendered: Vec<(&Infobox, Vec<String>)> = boxes
        .iter()
        .map(|b| (b, box_lines(dom, &b.items)))
        .filter(|(_, ln)| ln.iter().any(|l| !l.trim().is_empty() && !l.starts_with('#')))
        .collect();
    if rendered.is_empty() {
        return String::new();
    }
    let mut lines = vec![format!("## {}", wikil10n::key_facts_title(lang))];
    if rendered.len() == 1 {
        lines.push(String::new());
        lines.extend(rendered[0].1.iter().cloned());
    } else {
        for (b, b_lines) in &rendered {
            emit_heading(
                &mut lines,
                format!(
                    "### {}",
                    if b.title.is_empty() { wikil10n::details_title(lang) } else { &b.title }
                ),
            );
            lines.extend(b_lines.iter().cloned());
        }
    }
    cleanup::cleanup(&lines.join("\n"))
}
