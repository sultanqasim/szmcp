//! Tiny, dependency-free Markdown helpers: text extraction for article
//! intros and heading-based section extraction. They serve the
//! `text/markdown` articles wikizim_parser produces, which use a
//! straight-line subset of Markdown: ATX headings, `**bold**`/`*italic*`,
//! `[[Target|label]]` wikilinks, lists, pipe tables, fenced code blocks and
//! the occasional raw HTML block (complex tables).

use crate::html::{self, normalize, INTRO_SECTION};

/// Append plain text to the intro under construction: a single space
/// separates it from any text already emitted, whitespace inside the text
/// collapses the same way, and whitespace never consumes the character
/// budget (same rule as `intro_from_html`).
fn push_text(out: &mut String, text: &str, max_chars: usize) {
    // A space is due before the first character unless the intro is empty.
    let mut sep = !out.is_empty();
    // The character count of `out`, tracked as we go: recounting per
    // character would make long paragraphs quadratic.
    let mut len = out.chars().count();
    for c in text.chars() {
        if c.is_whitespace() {
            sep = true;
            continue;
        }
        // Room for the pending separator space (if any) and the char itself.
        if len + sep as usize >= max_chars {
            return;
        }
        if sep {
            out.push(' ');
            len += 1;
        }
        out.push(c);
        len += 1;
        sep = false;
    }
}

/// Resolve `[[Target|label]]` wikilinks to their text and drop emphasis and
/// inline-code markers (`*`, `` ` ``). The label wins over the target; a
/// bare target loses its `#anchor` and reads its underscores as spaces, so
/// `[[Mean_anomaly#Mean_anomaly_at_epoch]]` becomes "Mean anomaly".
fn strip_inline(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(close) = rest.find("]]") {
        // Anchor on the close and take the nearest opener before it, so a
        // literal bracket next to a link (`[[[Helium|He]]]`, the electron
        // configuration notation) stays text.
        let Some(open) = rest[..close].rfind("[[") else {
            out.push_str(&rest[..close + 2]); // stray close: keep as text
            rest = &rest[close + 2..];
            continue;
        };
        out.push_str(&rest[..open]);
        let inner = &rest[open + 2..close];
        let text = match inner.split_once('|') {
            Some((_, label)) if !label.is_empty() => label.to_string(),
            _ => {
                let target = inner.split('|').next().unwrap_or(inner);
                target.split('#').next().unwrap_or(target).replace('_', " ")
            }
        };
        out.push_str(&text);
        rest = &rest[close + 2..];
    }
    out.push_str(rest);
    // Emphasis and inline-code markers are never content.
    out.chars().filter(|&c| c != '*' && c != '`').collect()
}

/// Whether the (leading-trimmed) line opens or closes a fenced code block
/// (``` or ~~~, with an optional info string).
fn is_fence(t: &str) -> bool {
    t.starts_with("```") || t.starts_with("~~~")
}

/// The ATX heading level of a (leading-trimmed) line: `#{1..6}` followed by
/// whitespace or the end of the line; 0 when the line is not a heading.
fn heading_level(t: &str) -> u8 {
    let hashes = t.bytes().take_while(|&b| b == b'#').count();
    if (1..=6).contains(&hashes)
        && t.as_bytes().get(hashes).map_or(true, |b| b.is_ascii_whitespace())
    {
        hashes as u8
    } else {
        0
    }
}

/// Length of the list bullet at the start of a (leading-trimmed) line:
/// `- `, `* `, `+ ` or an ordered `12. `.
fn list_marker_len(t: &str) -> Option<usize> {
    if t.starts_with("- ") || t.starts_with("* ") || t.starts_with("+ ") {
        return Some(2);
    }
    let digits = t.bytes().take_while(|b| b.is_ascii_digit()).count();
    (digits > 0 && t[digits..].starts_with(". ")).then_some(digits + 2)
}

/// Whether a paragraph is a hatnote, as wikizim_parser emits them: text
/// entirely wrapped in single asterisks (`*For other uses, see [[X]].*`),
/// possibly wrapped over several lines. Bold (`**x**`) does not match: the
/// character after the first `*` is another `*`.
fn is_hatnote(para: &str) -> bool {
    let t = para.trim();
    t.len() > 2
        && t.starts_with('*')
        && t.ends_with('*')
        && !t[1..].starts_with('*')
        && !t[1..].starts_with(' ')
        && !t[1..t.len() - 1].contains('*')
}

/// Append one cleaned paragraph to the list, capped at `max_para_chars`
/// characters (whitespace collapses the way [`push_text`] collapses it).
fn push_paragraph(paras: &mut Vec<String>, text: &str, max_para_chars: usize) {
    let mut out = String::new();
    push_text(&mut out, text, max_para_chars);
    if !out.is_empty() {
        paras.push(out);
    }
}

/// The cleaned paragraph texts of one Markdown region: one per
/// blank-line-separated block (wikilinks resolved, emphasis and inline-code
/// markers stripped, hatnotes dropped, fenced blocks skipped, raw HTML
/// blocks handed to `intro_from_html`), each capped at `max_para_chars`.
/// Deeper headings inside the region are dropped like the HTML path drops
/// heading text: they name subregions whose paragraphs follow right after.
fn md_paragraphs(lines: &[&str], max_para_chars: usize) -> Vec<String> {
    let mut paras: Vec<String> = Vec::new();
    let mut in_fence = false;
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim_start();
        if is_fence(t) {
            in_fence = !in_fence;
            i += 1;
            continue;
        }
        if in_fence || t.is_empty() || heading_level(t) > 0 {
            i += 1;
            continue;
        }
        if t.starts_with('|') {
            // A pipe table: one paragraph of its cells' text; a dash-only
            // separator row is pure markup. Wikilinks are resolved first -
            // their `|` is a label separator, not a cell boundary.
            let mut cells = String::new();
            while i < lines.len() {
                let n = lines[i].trim_start();
                if !n.starts_with('|') {
                    break;
                }
                i += 1;
                let row = strip_inline(n).replace('|', " ");
                let row = row.trim();
                if !row.is_empty() && !row.chars().all(|c| matches!(c, '-' | ':' | ' ')) {
                    if !cells.is_empty() {
                        cells.push(' ');
                    }
                    cells.push_str(row);
                }
            }
            push_paragraph(&mut paras, &cells, max_para_chars);
            continue;
        }
        if t.starts_with('<') {
            // Raw HTML block (a complex table, its styles): it runs to the
            // next blank line, and the HTML helpers strip tags and drop
            // table/style content.
            let mut block = String::from(lines[i]);
            i += 1;
            while i < lines.len() && !lines[i].trim().is_empty() {
                block.push('\n');
                block.push_str(lines[i]);
                i += 1;
            }
            let text = html::intro_from_html(&block, max_para_chars);
            push_paragraph(&mut paras, &text, max_para_chars);
            continue;
        }
        if list_marker_len(t).is_some() {
            // A run of list items becomes one paragraph: the items of a
            // "Key Facts" or "See also" list belong together. Dash-only
            // item bodies carry no text.
            let mut items = String::new();
            while let Some(n) = lines.get(i).map(|l| l.trim_start()) {
                let Some(m) = list_marker_len(n) else { break };
                i += 1;
                let body = &n[m..];
                if !body.chars().all(|c| matches!(c, '-' | ':' | ' ')) {
                    if !items.is_empty() {
                        items.push(' ');
                    }
                    items.push_str(&strip_inline(body));
                }
            }
            push_paragraph(&mut paras, &items, max_para_chars);
            continue;
        }
        // A plain paragraph: gather its remaining lines so a hatnote (see
        // `is_hatnote`) can be skipped whole. Any special line - heading,
        // list, table, fence, HTML, a textless dash row - ends the
        // paragraph and is handled on its own turn through the loop.
        let mut para = String::from(t);
        i += 1;
        while i < lines.len() {
            let n = lines[i].trim_start();
            if n.is_empty()
                || is_fence(n)
                || heading_level(n) > 0
                || n.starts_with('|')
                || n.starts_with('<')
                || list_marker_len(n).is_some()
                || n.chars().all(|c| matches!(c, '-' | ':' | ' '))
            {
                break;
            }
            para.push('\n');
            para.push_str(lines[i]);
            i += 1;
        }
        if para.chars().all(|c| matches!(c, '-' | ':' | ' ')) {
            continue; // a thematic break carries no text
        }
        if is_hatnote(&para) {
            continue;
        }
        push_paragraph(&mut paras, &strip_inline(&para), max_para_chars);
    }
    paras
}

/// Split a Markdown article into its intro region and one region per
/// heading, with the cleaned paragraph texts of each region (the shape
/// [`html::sections`] mirrors for HTML). Returns `(name, paragraphs)` pairs:
/// the first entry carries the intro region under the `INTRO_SECTION`
/// name, the later entries carry each heading's section under the heading
/// text as written, spanning what [`section_content`] would return for it,
/// so a nested `###`'s paragraphs belong to its own entry and to the
/// enclosing `##`'s. The leading `# Title` heading is the article title - a
/// separate field of every search hit - not a section: it is dropped and
/// the intro region runs from after it (see [`title_split`]).
pub fn sections(md: &str, max_para_chars: usize) -> Vec<(String, Vec<String>)> {
    let lines: Vec<&str> = md.lines().collect();
    let headings = collect_headings(&lines);
    let (intro_start, first_section) = title_split(&lines, &headings);
    let intro_end = headings
        .get(first_section)
        .map_or(lines.len(), |h| h.line);
    let mut out = vec![(INTRO_SECTION.to_string(), md_paragraphs(&lines[intro_start..intro_end], max_para_chars))];
    for (i, h) in headings.iter().enumerate().skip(first_section) {
        let end = headings[i + 1..]
            .iter()
            .find(|n| n.level <= h.level)
            .map_or(lines.len(), |n| n.line);
        out.push((
            h.name.clone(),
            md_paragraphs(&lines[h.line + 1..end], max_para_chars),
        ));
    }
    out
}

/// The intro region's paragraphs: the paragraphs [`sections`] reports for
/// its `INTRO_SECTION` entry, computed without extracting any body
/// section's paragraphs (a search hit's lead fast path needs only these).
pub fn intro_paragraphs(md: &str, max_para_chars: usize) -> Vec<String> {
    let lines: Vec<&str> = md.lines().collect();
    let headings = collect_headings(&lines);
    let (intro_start, first_section) = title_split(&lines, &headings);
    let intro_end = headings
        .get(first_section)
        .map_or(lines.len(), |h| h.line);
    md_paragraphs(&lines[intro_start..intro_end], max_para_chars)
}

/// Where the intro region starts and where the section entries begin: a
/// document opening with a heading (however deep) drops it as the article
/// title. Both [`sections`] and [`intro_paragraphs`] split on this rule.
fn title_split(lines: &[&str], headings: &[Heading]) -> (usize, usize) {
    let title = headings
        .first()
        .filter(|h| lines[..h.line].iter().all(|l| l.trim().is_empty()));
    match title {
        Some(t) => (t.line + 1, 1),
        None => (0, 0),
    }
}

struct Heading {
    level: u8,
    /// Heading text as written (minus the `#` markers).
    name: String,
    /// Heading text normalized for matching (inline markdown stripped).
    name_norm: String,
    /// Index of the heading's line in the document's line list.
    line: usize,
}

/// Strip inline markdown from heading text for matching: `*` and backtick
/// markers go, and `_` goes only in pairs (`_like this_`), since a lone
/// underscore is subscript-style math that stays (`*A*_r°(O)` keeps its
/// `_r`).
fn heading_key(s: &str) -> String {
    let no_markers: String = s.chars().filter(|&c| c != '*' && c != '`').collect();
    let parts: Vec<&str> = no_markers.split('_').collect();
    // An even number of `_` markers reads as paired emphasis (`_like this_`)
    // and is dropped; a lone one is subscript math and stays.
    let text = if parts.len() % 2 == 1 { parts.concat() } else { no_markers.clone() };
    normalize(&text)
}

/// Find all ATX headings (outside fenced code blocks), with the line index
/// each one's section starts after.
fn collect_headings(lines: &[&str]) -> Vec<Heading> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for (i, raw) in lines.iter().enumerate() {
        let t = raw.trim_start();
        if is_fence(t) {
            in_fence = !in_fence;
            continue;
        }
        // Heading-like lines inside a fenced block are code, not markup.
        if in_fence {
            continue;
        }
        let level = heading_level(t);
        if level > 0 {
            let name = t[level as usize..].trim().to_string();
            let name_norm = heading_key(&name);
            out.push(Heading { level, name, name_norm, line: i });
        }
    }
    out
}

/// Find the content of the named section in a Markdown document.
///
/// Matches the section name (case-insensitively, whitespace-normalized)
/// against heading texts with their inline markdown stripped; among matches
/// the most prominent (lowest level), then earliest, heading wins. Returns
/// the heading text as written plus the raw Markdown between that heading
/// and the next heading of the same or higher level, trimmed.
///
/// The reserved name [`INTRO_SECTION`] selects the article's introduction
/// instead: the raw Markdown from after the leading `# Title` line (per
/// [`title_split`], as in [`sections`]) to the first heading. The
/// introduction always exists, so an empty region is empty content, never
/// `None`.
pub fn section_content(md: &str, name: &str) -> Option<(String, String)> {
    let target = normalize(name);
    if target.is_empty() {
        return None;
    }
    let lines: Vec<&str> = md.lines().collect();
    let headings = collect_headings(&lines);
    // The reserved intro name can never match a heading text, so it is
    // special-cased before the heading search.
    if target == normalize(INTRO_SECTION) {
        let (intro_start, first_section) = title_split(&lines, &headings);
        let end = headings
            .get(first_section)
            .map_or(lines.len(), |h| h.line);
        let content = lines[intro_start..end].join("\n");
        return Some((INTRO_SECTION.to_string(), content.trim().to_string()));
    }
    let best = headings
        .iter()
        .enumerate()
        .filter(|(_, h)| h.name_norm == target)
        .min_by_key(|(i, h)| (h.level, *i))?;
    let end = headings
        .iter()
        .filter(|h| h.line > best.1.line && h.level <= best.1.level)
        .map(|h| h.line)
        .min()
        .unwrap_or(lines.len());
    let content = lines[best.1.line + 1..end].join("\n");
    Some((best.1.name.clone(), content.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An article in the shape wikizim_parser emits: title, a wrapped
    /// hatnote, a lead paragraph, sections with inline markup, a Key Facts
    /// infobox section with nested lists, a pipe table and a fenced block.
    const CHEMISTRY: &str = "\
# Chemistry

*For other uses, see [[Chemistry (disambiguation)]]. \"Chemical science\"
redirects here.*

**Chemistry** is the scientific study of the [[Matter|matter]] and its
properties. It is a [[Physical science|physical science]].

## Etymology

The word *chemistry* comes from the [[Renaissance]].

### Alchemy

Alchemy preceded chemistry as a discipline.

## Key Facts

### Standard atomic weight *A*_r°(O)

- **[[Atomic number]] (Z)**: 8
- **[[Electron configuration]]**: [[[Helium|He]]] 2s^2 2p^4
  - shells fill in order
- **[[Block (periodic table)#p-block|p-block]]**: yes

## Uses

| Element | Use |
| --- | --- |
| Oxygen | breathing |

```
code line with ## inside
```

That is all.
";

    #[test]
    fn section_extraction() {
        let (name, etym) = section_content(CHEMISTRY, "etymology").unwrap();
        assert_eq!(name, "Etymology");
        assert!(etym.contains("comes from the [[Renaissance]]"), "{etym:?}");
        // Includes the h3 subsection, stops at the next h2.
        assert!(etym.contains("Alchemy preceded chemistry"), "{etym:?}");
        assert!(!etym.contains("Atomic number"), "{etym:?}");
        assert!(!etym.contains("breathing"), "{etym:?}");

        // The Key Facts infobox section, raw Markdown content preserved.
        let (name, kf) = section_content(CHEMISTRY, "Key Facts").unwrap();
        assert_eq!(name, "Key Facts");
        assert!(kf.contains("- **[[Atomic number]] (Z)**: 8"), "{kf:?}");
        assert!(kf.contains("shells fill in order"), "{kf:?}");
        assert!(!kf.contains("breathing"), "{kf:?}");

        // A subsection on its own.
        let (name, alchemy) = section_content(CHEMISTRY, "Alchemy").unwrap();
        assert_eq!(name, "Alchemy");
        assert!(alchemy.contains("preceded chemistry"), "{alchemy:?}");
        assert!(!alchemy.contains("Renaissance"), "{alchemy:?}");

        assert!(section_content(CHEMISTRY, "Nope").is_none());
        assert!(section_content(CHEMISTRY, "").is_none());
    }

    #[test]
    fn section_content_intro_region() {
        // The reserved intro name returns the region between the leading
        // `# Title` line and the first heading: hatnote and lead paragraph,
        // raw, under the reserved name.
        let (name, intro) = section_content(CHEMISTRY, "_intro").unwrap();
        assert_eq!(name, "_intro");
        assert!(
            intro.contains("*For other uses, see [[Chemistry (disambiguation)]]."),
            "{intro:?}"
        );
        assert!(
            intro.contains("**Chemistry** is the scientific study of the [[Matter|matter]]"),
            "{intro:?}"
        );
        // From after the title line, to (not including) the first heading.
        assert!(!intro.starts_with('#'), "{intro:?}");
        assert!(!intro.contains("Etymology"), "{intro:?}");

        // Matched case-insensitively, like heading names.
        assert!(section_content(CHEMISTRY, "_Intro").is_some());

        // Without a leading title heading, the intro starts at the first
        // line.
        let (_, intro) =
            section_content("Plain lead.\n\n## Section\n\nBody.\n", "_intro").unwrap();
        assert_eq!(intro, "Plain lead.");

        // The introduction always exists: nothing between the title line
        // and the first heading is an empty intro, not an error.
        let (name, intro) = section_content("# Doc\n\n## Section\n\nBody.\n", "_intro").unwrap();
        assert_eq!(name, "_intro");
        assert_eq!(intro, "");
    }

    #[test]
    fn section_matching_strips_inline_markdown() {
        // The emphasis markers go; the lone underscore (subscript math)
        // stays, so the query mirrors what the heading reads like.
        let (name, _) = section_content(CHEMISTRY, "standard atomic weight a_r°(o)").unwrap();
        assert_eq!(name, "Standard atomic weight *A*_r°(O)");
        assert!(section_content(CHEMISTRY, "Standard atomic weight A_r°(O)").is_some());
        // Paired `_` markers read as emphasis and are stripped, too.
        let md = "# Doc\n\n## _Nested_ lists\n\n- a\n";
        assert!(section_content(md, "nested lists").is_some());
    }

    #[test]
    fn section_takes_most_prominent_then_earliest() {
        let md = "# D\n\n## Step\n\nfirst\n\n### Step\n\ndetail\n\n## Step\n\nlast\n";
        let (name, content) = section_content(md, "step").unwrap();
        assert_eq!(name, "Step");
        assert!(content.starts_with("first"), "{content:?}");
        // The later same-level heading bounds the first one's content...
        assert!(!content.contains("last"), "{content:?}");
        // ...but the deeper one belongs to it.
        assert!(content.contains("detail"), "{content:?}");
    }

    #[test]
    fn section_ignores_headings_inside_fenced_blocks() {
        let md = "# Doc\n\n## Real\n\n```\n## Fake\n```\n\nMore.\n";
        assert!(section_content(md, "Fake").is_none());
        let (_, real) = section_content(md, "Real").unwrap();
        // The fence (with its heading-looking line) stays in the section.
        assert!(real.contains("## Fake"), "{real:?}");
        assert!(real.contains("More."), "{real:?}");
    }

    /// Title, hatnote, lead paragraphs, a section with a nested heading, a
    /// section with a table, a fence, a list and an HTML block - the block
    /// kinds `md_paragraphs` has to tell apart.
    const SECTIONS_MD: &str = "# Salt\n\n*This article is about the mineral. For the seasoning, see [[Pepper]].*\n\n**Salt** is a mineral composed of sodium chloride.\n\nIt is an ionic compound.\n\n## History\n\nSalt has been mined for millennia.\n\n### China\n\nChinese salt lakes fed ancient trade.\n\n## Uses\n\n| Use | Detail |\n| --- | --- |\n| seasoning | keeps food edible |\n\n```\nfenced code, not prose\n```\n\n- [[Preservation|preserving]] meat\n- tanning\n\n<p>plain &amp; <b>bold</b></p>\n\nFinal paragraph.\n";

    #[test]
    fn sections_split_intro_and_headings_with_clean_paragraphs() {
        let secs = sections(SECTIONS_MD, 400);
        // The intro region: the paragraphs after the dropped title line,
        // hatnote dropped, under the reserved intro name.
        assert_eq!(secs[0].0, "_intro");
        assert_eq!(
            secs[0].1,
            vec![
                "Salt is a mineral composed of sodium chloride.".to_string(),
                "It is an ionic compound.".to_string(),
            ]
        );
        // One entry per heading (nested ones included), named as written.
        assert_eq!(secs[1].0, "History");
        assert_eq!(
            secs[1].1,
            vec![
                "Salt has been mined for millennia.".to_string(),
                "Chinese salt lakes fed ancient trade.".to_string(),
            ]
        );
        assert_eq!(secs[2].0, "China");
        assert_eq!(secs[2].1, vec!["Chinese salt lakes fed ancient trade."]);
        assert_eq!(secs[3].0, "Uses");
        assert_eq!(
            secs[3].1,
            vec![
                // The table's cells run together into one paragraph.
                "Use Detail seasoning keeps food edible".to_string(),
                // The list items run together into one paragraph.
                "preserving meat tanning".to_string(),
                // The HTML block goes through the HTML helpers.
                "plain & bold".to_string(),
                "Final paragraph.".to_string(),
            ]
        );
        assert_eq!(secs.len(), 4);
        for (name, paras) in &secs {
            // Hatnote, title line and fence content never become paragraphs.
            assert!(!paras.iter().any(|p| p.contains("seasoning, see")), "{name}: {paras:?}");
            assert!(!paras.iter().any(|p| p.contains("fenced code")), "{name}: {paras:?}");
            assert!(!paras.iter().any(|p| p.contains('#')), "{name}: {paras:?}");
            assert!(!paras.iter().any(|p| p.contains("**") || p.contains("[[")), "{name}: {paras:?}");
        }
    }

    #[test]
    fn sections_resolve_wikilink_forms() {
        // Labels win over targets; bare targets lose their `#anchor` and
        // read underscores as spaces; a literal bracket next to a link
        // (`[[[Helium|He]]]`, electron-configuration notation) stays text.
        let md = "# Doc\n\nSee [[Earth's age]], [[Mean anomaly#Mean anomaly at epoch]], \
                  [[Chemical_element]], [[Block (periodic table)#p-block|p-block]] \
                  and [[[Helium|He]]] atoms.";
        let secs = sections(md, 300);
        assert_eq!(
            secs[0].1,
            vec!["See Earth's age, Mean anomaly, Chemical element, p-block and [He] atoms.".to_string()]
        );
    }

    #[test]
    fn sections_without_a_leading_title_heading() {
        // A document that does not open with `# Title`: nothing is dropped,
        // the plain lead is the intro region.
        let md = "Plain lead text.\n\n## Section\n\nBody.\n";
        let secs = sections(md, 100);
        assert_eq!(secs[0].0, "_intro");
        assert_eq!(secs[0].1, vec!["Plain lead text."]);
        assert_eq!(secs[1].0, "Section");
        assert_eq!(secs[1].1, vec!["Body."]);

        // No headings at all: a single intro entry.
        assert_eq!(
            sections("One.\n\nTwo.\n", 100),
            vec![("_intro".to_string(), vec!["One.".to_string(), "Two.".to_string()])]
        );
    }
    #[test]
    fn sections_cap_paragraph_characters() {
        let secs = sections(SECTIONS_MD, 12);
        assert_eq!(secs[0].1[0], "Salt is a mi");
        assert!(secs.iter().all(|(_, paras)| paras.iter().all(|p| p.chars().count() <= 12)));
    }
}
