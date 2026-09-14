//! Tiny, dependency-free Markdown helpers: text extraction for article
//! intros and heading-based section extraction. They serve the
//! `text/markdown` articles wikizim_parser produces, which use a
//! straight-line subset of Markdown: ATX headings, `**bold**`/`*italic*`,
//! `[[Target|label]]` wikilinks, lists, pipe tables, fenced code blocks and
//! the occasional raw HTML block (complex tables).

use crate::html::{self, normalize};

/// Append plain text to the intro under construction: a single space
/// separates it from any text already emitted (a skipped line - blank line,
/// fence, separator row - contributes nothing but its successor still gets
/// its space), and whitespace inside the text collapses the same way.
/// Markup and whitespace never consume the character budget (same rule as
/// `intro_from_html`).
fn push_text(out: &mut String, text: &str, max_chars: usize) {
    // A space is due before the first character unless the intro is empty.
    let mut sep = !out.is_empty();
    for c in text.chars() {
        if c.is_whitespace() {
            sep = true;
            continue;
        }
        // Room for the pending separator space (if any) and the char itself.
        if out.chars().count() + sep as usize >= max_chars {
            return;
        }
        if sep {
            out.push(' ');
        }
        out.push(c);
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

/// Plain-text preview of the start of a Markdown article: the paragraphs
/// after the leading title line, until `max_chars` characters of text have
/// been produced. The leading ATX heading is the article title - a separate
/// field of every search hit - and is not emitted; emitting it as well made
/// intros read "Photosynthesis Photosynthesis (/ˌfoʊtə... ) is ...". The
/// rule is positional: a heading reached before any text has been emitted
/// is the title and is skipped, a heading reached after that is content and
/// keeps its text (only the leading title line is special). Headings lose
/// their `#` markers, emphasis and inline-code markers are stripped,
/// wikilinks become their text, list bullets and pipe-table cell separators
/// vanish, fenced code blocks are skipped entirely, and raw HTML lines are
/// handed to `intro_from_html` for tag stripping. Hatnote paragraphs
/// (standalone paragraphs wrapped in single asterisks, how wikizim_parser
/// renders "For other uses..." notes) are dropped, so the intro starts with
/// the actual lead text. Whitespace collapses to single spaces; markup and
/// whitespace never consume the character budget.
pub fn intro_from_markdown(md: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    let mut lines = md.lines().peekable();
    while let Some(line) = lines.next() {
        if out.chars().count() >= max_chars {
            break;
        }
        let t = line.trim_start();
        if is_fence(t) {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if t.is_empty() {
            continue;
        }
        // Raw HTML block (a complex table, its styles): the HTML helpers
        // strip tags and drop table/style content. Such blocks run to the
        // next blank line, the way wikizim_parser emits them.
        if t.starts_with('<') {
            let mut block = String::from(line);
            while let Some(next) = lines.peek() {
                if next.trim().is_empty() {
                    break;
                }
                block.push('\n');
                block.push_str(next);
                lines.next();
            }
            let text = html::intro_from_html(&block, max_chars.saturating_sub(out.chars().count()));
            push_text(&mut out, &text, max_chars);
            continue;
        }
        let level = heading_level(t);
        if level > 0 {
            // ATX heading: the `#` markers are markup, the text is content -
            // except while no text has been emitted yet, when the heading is
            // the article's leading title line (wikizim_parser opens every
            // article with `# Title`, already reported as its own field of
            // every search hit) and is skipped. Once text flows, headings
            // are content, even another level-1 one.
            if !out.is_empty() {
                let text = strip_inline(t[level as usize..].trim_start());
                push_text(&mut out, &text, max_chars);
            }
            continue;
        }
        if t.starts_with('|') {
            // Pipe-table row: cells become text; a dash-only separator row
            // is pure markup. Wikilinks are resolved first - their `|` is a
            // label separator, not a cell boundary.
            let row = strip_inline(t).replace('|', " ");
            let row = row.trim();
            if !row.is_empty() && !row.chars().all(|c| matches!(c, '-' | ':' | ' ')) {
                push_text(&mut out, row, max_chars);
            }
            continue;
        }
        // List items: drop the bullet marker, keep the text. A body made
        // only of dashes/colons carries no text.
        if let Some(n) = list_marker_len(t) {
            let body = &t[n..];
            if !body.chars().all(|c| matches!(c, '-' | ':' | ' ')) {
                let text = strip_inline(body);
                push_text(&mut out, &text, max_chars);
            }
            continue;
        }
        // A plain paragraph: gather its remaining lines so a hatnote (see
        // `is_hatnote`) can be skipped whole. Any special line - heading,
        // list, table, fence, HTML, a textless dash row - ends the
        // paragraph and is handled on its own turn through the loop.
        let mut para = String::from(t);
        while let Some(next) = lines.peek() {
            let n = next.trim_start();
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
            para.push_str(next);
            lines.next();
        }
        // A line made only of dashes/colons (a thematic break) carries no text.
        if para.chars().all(|c| matches!(c, '-' | ':' | ' ')) {
            continue;
        }
        if is_hatnote(&para) {
            continue;
        }
        let text = strip_inline(&para);
        push_text(&mut out, &text, max_chars);
    }
    out.trim().to_string()
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
/// underscore is subscript-style math that stays - `*A*_r°(O)` keeps its
/// `_r` so it matches a query for "standard atomic weight a_r°(O)".
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
pub fn section_content(md: &str, name: &str) -> Option<(String, String)> {
    let target = normalize(name);
    if target.is_empty() {
        return None;
    }
    let lines: Vec<&str> = md.lines().collect();
    let headings = collect_headings(&lines);
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
    fn intro_strips_markup_and_resolves_wikilinks() {
        let intro = intro_from_markdown(CHEMISTRY, 400);
        // The leading `# Chemistry` title line is not emitted (the title is
        // a separate field of every search hit): the hatnote paragraph is
        // dropped, so the intro starts with the actual article text - the
        // bold article name inside the lead sentence is sentence text and
        // stays.
        assert!(
            intro.starts_with("Chemistry is the scientific study of the matter and its properties"),
            "{intro:?}"
        );
        assert!(!intro.contains("For other uses"), "{intro:?}");
        // No markdown markup survives into the intro.
        assert!(!intro.contains("**") && !intro.contains("[["), "{intro:?}");
        assert!(!intro.contains('#'), "{intro:?}");
        assert!(!intro.contains('*'), "{intro:?}");
    }

    #[test]
    fn intro_excludes_the_leading_title_line() {
        // The leading `# Title` line must not reach the intro: the title is
        // a separate field of every search hit, and emitting it as well made
        // intros read "Kyoto Kyoto is ...". A lead that does not start with
        // the title word makes the rule observable.
        let md = "# Kyoto\n\n*Not to be confused with [[Kyoto Prefecture]].*\n\n\
                  It was the imperial capital of Japan for over a thousand years.\n";
        let intro = intro_from_markdown(md, 200);
        assert_eq!(intro, "It was the imperial capital of Japan for over a thousand years.");
        // The hatnote following the title is still dropped.
        assert!(!intro.contains("confused with"), "{intro:?}");
    }

    #[test]
    fn intro_emits_mid_document_h1_headings() {
        // Only the leading title line is special: a heading reached after
        // text has been emitted is content and keeps its text, even a
        // level-1 one.
        let md = "# Doc\n\nLead text first.\n\n# Appendix\n\nTrailing note.\n";
        let intro = intro_from_markdown(md, 200);
        assert_eq!(intro, "Lead text first. Appendix Trailing note.");
    }

    #[test]
    fn intro_skips_hatnotes() {
        // Hatnotes are standalone paragraphs wrapped in single asterisks,
        // possibly wrapped over several lines. Bold paragraphs do not match.
        let md = "# X\n\n*For the village, see [[X (village)]].*\n\n\
                  *A second hatnote\nwrapped over two lines.*\n\n\
                  **X** is a thing of great importance.\n";
        let intro = intro_from_markdown(md, 200);
        assert_eq!(intro, "X is a thing of great importance.");

        // Hatnotes further in (before a section) are dropped as well.
        let md = "# Y\n\n**Y** leads here.\n\n## Section\n\n\
                  *Main article: [[Something else]]*\n\nSection text follows.\n";
        let intro = intro_from_markdown(md, 200);
        assert_eq!(intro, "Y leads here. Section Section text follows.");
    }

    #[test]
    fn intro_resolves_wikilink_forms() {
        let md = "# T\n\nSee [[Earth's age]], [[Mean anomaly#Mean anomaly at epoch]], \
                  [[Chemical_element]], [[Block (periodic table)#p-block|p-block]] \
                  and [[[Helium|He]]] atoms.";
        let intro = intro_from_markdown(md, 300);
        assert_eq!(
            intro,
            "See Earth's age, Mean anomaly, Chemical element, p-block and [He] atoms."
        );
    }

    #[test]
    fn intro_skips_fences_and_separator_rows() {
        let md = "# Code\n\n```rust\nlet x = 1;\n## not a heading\n```\n\n\
                  | a | b |\n| --- | --- |\n| x | y |\n\n~~~\nfence two\n~~~\n\nAfter.";
        let intro = intro_from_markdown(md, 200);
        assert_eq!(intro, "a b x y After.");
    }

    #[test]
    fn intro_strips_list_bullets() {
        let md = "# KF\n\n- **[[Atomic number]] (Z)**: 8\n  - shells fill in order\n\
                  1. ordered item\n";
        let intro = intro_from_markdown(md, 200);
        assert_eq!(intro, "Atomic number (Z): 8 shells fill in order ordered item");
    }

    #[test]
    fn intro_feeds_html_blocks_through_the_html_helpers() {
        // Tags and entities are handled by intro_from_html; a raw <table>
        // block (what complex tables stay as) is skipped like in HTML pages.
        let md = "# H\n\n<p>plain &amp; <b>bold</b></p>\n\n\
                  <table>\n<tr><td>cells</td></tr>\n</table>\n\nAfter.";
        let intro = intro_from_markdown(md, 100);
        assert_eq!(intro, "plain & bold After.");
    }

    #[test]
    fn intro_respects_char_limit() {
        let intro = intro_from_markdown(CHEMISTRY, 10);
        assert!(intro.chars().count() <= 10, "{intro:?}");
    }

    #[test]
    fn intro_budget_ignores_markup() {
        // Hundreds of separator rows (pure markup) must not crowd the real
        // text out of the character budget.
        let md = format!("# T\n\n{}\nreal text here", "| --- |\n".repeat(400));
        assert_eq!(intro_from_markdown(&md, 20), "real text here");
    }

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
}
