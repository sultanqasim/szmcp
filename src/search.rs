//! The search pipeline behind the `zim_search` tool: query parsing and
//! ranking against the ZIM full-text indexes, cross-archive merging, and
//! the per-hit preview/section reporting.

use crate::html;
use crate::markdown;
use crate::tools::ToolError;
use crate::zim::{Archive, ZimLibrary};
use schemars::JsonSchema;
use serde::Serialize;
use std::sync::Arc;
use xapian2::{Database, Enquire, Operator, Query, QueryParser, Stem, StemStrategy};

/// Number of results `zim_search` returns in total (across all archives).
const SEARCH_LIMIT: u32 = 20;
/// Maximum characters of the `preview` reported per search hit.
const INTRO_CHARS: usize = 300;
/// How many raw bytes of an article are read to locate the query's matches
/// in it (region and paragraph level). Matching needs the whole article, not
/// just the lead the old 64 KiB intro preview covered; for compressed
/// clusters the whole cluster decompresses anyway.
const HIT_READ_BYTES: u64 = 1024 * 1024;
/// Cap on one paragraph's characters while scanning it for matches: long
/// paragraphs keep matching far into their text. A paragraph chosen for
/// reporting is truncated to `INTRO_CHARS` separately.
const PARA_MATCH_CHARS: usize = 2000;

/// Common English words dropped when collecting the query's stemmed terms
/// (they drive the all-words branch, the title-index band, and the paragraph
/// matching). They are stopped at index time (libzim's TermGenerator), so
/// they are absent from the index terms and an AND over them would match
/// nothing.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in",
    "is", "it", "its", "not", "of", "on", "or", "that", "the", "these", "this", "those", "to",
    "was", "were", "which", "with",
];

/// A stemmed query term is *specific* enough to define the all-words tier
/// when its document frequency in the archive's full-text index is at most
/// this fraction of the archive's document count. A term occurring in more
/// than 1% of the archive is background vocabulary - in a Wikipedia edition
/// "attractions"/"attraction"/"attractive" all stem to one term present in
/// 5.7% of all articles - so its incidental occurrences in near-random
/// documents must not define the all-words tier, and a genuinely relevant
/// article that happens to miss that one word must not sink under it.
const AND_TERM_MAX_DF_FRAC: f64 = 0.01;

/// One search result.
#[derive(Serialize, JsonSchema, Debug)]
pub struct SearchHit {
    /// ZIM file name, relative to the ZIM directory
    pub zim: String,
    /// Path of the article inside the ZIM file
    pub path: String,
    /// Page/article title
    pub title: String,
    /// Preview of the article: the first paragraph when the query matches
    /// the title or that paragraph, otherwise the sentence with the most
    /// query matches, followed by its paragraph's next sentences up to the
    /// length cap
    pub preview: String,
    /// Names of the regions holding query matches - the intro listed as
    /// `_intro` first when it matched, then the sections in document order;
    /// absent when the query matches the title or the first intro paragraph
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sections: Option<Vec<String>>,
}

/// The search result set (best matches first).
#[derive(Serialize, JsonSchema)]
pub struct SearchResults {
    /// The search results
    pub results: Vec<SearchHit>,
}

/// The query's non-stopword terms, stemmed the way the ZIM full-text
/// indexes were built (unprefixed Porter2 stems, lowercased): split on
/// whitespace, keep alphanumeric characters only per word, lowercase, drop
/// stopwords, stem, dedupe (preserving first-occurrence order).
fn query_terms(query: &str, stem: &mut Stemmer) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    for word in query.split_whitespace() {
        // Punctuation never appears in index terms either, so strip it.
        let word: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        let word = word.to_lowercase();
        if word.is_empty() || STOPWORDS.contains(&word.as_str()) {
            continue;
        }
        // The index has no spelling data, so an unknown word just stems to
        // something that matches nothing; it cannot break anything here.
        let stemmed = stem.stem(&word).to_string();
        if !terms.contains(&stemmed) {
            terms.push(stemmed);
        }
    }
    terms
}

/// Combine `terms` (non-empty) with `op`, left to right; Xapian flattens
/// the resulting tree itself.
fn combine_terms<T: AsRef<str>>(op: Operator, terms: &[T]) -> xapian2::Result<Query> {
    let mut query = Query::term(terms[0].as_ref())?;
    for term in &terms[1..] {
        query = Query::combine(op, &query, &Query::term(term.as_ref())?)?;
    }
    Ok(query)
}

/// The all-words branch of one archive's full-text query: an AND over the
/// query's *specific* terms (see [`AND_TERM_MAX_DF_FRAC`]) when at least
/// two of them qualify, else `None` - the parsed OR query runs alone. BM25
/// sums term weights across OR branches, so ORing this AND with the parsed
/// query ranks articles containing every specific word above partial
/// matches, while partial matches still return; the branch is built per
/// archive because document frequencies are per database. (OP_PHRASE and
/// OP_NEAR are no substitute: the indexes carry no positional data. A
/// misspelled word matches nothing here while the parsed OR side still
/// retrieves results for the good words.)
fn all_words_branch(
    fulltext: &Database,
    terms: &[String],
    max_df_frac: f64,
) -> xapian2::Result<Option<Query>> {
    let cap = max_df_frac * f64::from(fulltext.doc_count());
    let specific: Vec<&str> = terms
        .iter()
        .map(String::as_str)
        .filter(|t| f64::from(fulltext.termfreq(t)) <= cap)
        .collect();
    if specific.len() < 2 {
        return Ok(None);
    }
    Ok(Some(combine_terms(Operator::And, &specific)?))
}

/// A stemmer that remembers the stem of every word it has seen. Natural
/// text repeats its words heavily, and every uncached stem crosses the
/// Xapian FFI; one instance serves a whole search (the query terms and all
/// hits' paragraph matchers), so repeats dominate after the first hit.
struct Stemmer {
    stem: Stem,
    cache: std::collections::HashMap<String, String>,
}

impl Stemmer {
    fn new(language: &str) -> xapian2::Result<Self> {
        Ok(Self {
            stem: Stem::new(language)?,
            cache: std::collections::HashMap::new(),
        })
    }

    /// The word's stem, cased and stemmed the way `query_terms` and the ZIM
    /// full-text indexes were built (lowercased Porter2 stems).
    fn stem(&mut self, word: &str) -> &str {
        if !self.cache.contains_key(word) {
            let lowered = word.to_lowercase();
            let stemmed = self.stem.apply(&lowered).unwrap_or(lowered);
            self.cache.insert(word.to_string(), stemmed);
        }
        self.cache[word].as_str()
    }
}

/// Search all articles in all ZIM files of the library - the pipeline behind
/// the `zim_search` tool: ranked hits, best first.
pub fn search(library: &ZimLibrary, query: &str) -> Result<SearchResults, ToolError> {
    if query.trim().is_empty() {
        return Err(ToolError::InvalidArgument("query must not be empty".into()));
    }

    let mut qp = QueryParser::new()?;
    // openZIM's full-text indexes contain unprefixed Porter2/English stems
    // (libzim indexes with STEM_ALL), so queries must be stemmed the same
    // way - Xapian's default strategy would turn lowercase terms into
    // "Z"-prefixed stem terms that never match. Default combining op is OR.
    qp.set_stemmer("english")?;
    qp.set_stemming_strategy(StemStrategy::All)?;
    qp.set_default_op(Operator::Or)?;
    let xquery = qp
        .parse_query(query)
        .map_err(|e| ToolError::InvalidArgument(format!("failed to parse query: {e}")))?;

    // The query's terms drive three things: the all-words branch of the
    // full-text query (built per archive below, from that archive's term
    // statistics - see `all_words_branch`), the title-index band, and the
    // paragraph matching when the hits are built - one stemmer serves all
    // three. The parsed OR side of the full-text query is final here.
    let mut stemmer = Stemmer::new("english")?;
    let terms = query_terms(query, &mut stemmer);

    /// Which band produced a hit. Exact and title-index hits are title
    /// matches: their preview is the lead paragraph and `sections` is
    /// omitted.
    #[derive(Clone, Copy, PartialEq)]
    enum HitKind {
        /// Exact title/URL probe hit (the ZIM directory itself).
        ExactTitle,
        /// Hit from the archive's title index (`X/title/xapian`).
        TitleIndex,
        /// Hit from the archive's full-text index.
        Fulltext,
    }

    // Exact title/URL matches, found in the ZIM directory itself: redirects
    // are not in the search indexes, and a query that names an article
    // exactly must rank first no matter what BM25 produces. One probe per
    // archive; a failed probe simply contributes nothing. The title falls
    // back to the query (spaces restored) because modern openZIM archives
    // leave directory-entry titles empty.
    let mut merged: Vec<(&Arc<Archive>, String, String, HitKind)> = Vec::new();
    for arc in &library.archives {
        if let Some((path, title)) = arc.lookup_exact(query)? {
            let title = if title.is_empty() { query.replace('_', " ") } else { title };
            merged.push((arc, path, title, HitKind::ExactTitle));
        }
    }

    // Per-archive ranked hit lists. Each archive's Xapian handles are
    // checked out of that archive's pool for the duration of the search:
    // concurrent searches never share a handle (Xapian does not support
    // concurrent calls on one database object).
    //
    // The title-index band queries the archive's title index (`X/title/xapian`;
    // documents ARE titles: the same unprefixed stems as the full-text index,
    // so the same `terms` match directly) in two sub-bands: for multi-word
    // queries an AND over ALL the stemmed terms first (titles containing
    // every query word), then the OR of the terms (partial title matches).
    // Under the OR ranking alone, a title matching one ultra-common query
    // word can outrank titles containing every word - the AND sub-band
    // fixes the ordering inside the band. The band ranks its hits ahead of
    // every full-text match - a title that says the whole query is far
    // stronger evidence than body words - and is skipped for archives
    // without a title index and queries with no usable terms.
    let mut title_lists: Vec<(&Arc<Archive>, Vec<(String, String)>)> = Vec::new();
    // Full-text band: (weight, path, title from the index).
    let mut per_archive: Vec<(&Arc<Archive>, Vec<(f64, String, String)>)> = Vec::new();
    for arc in &library.archives {
        // Paths already reported as exact title/URL matches for THIS archive:
        // the title band must not report them again (the skip below).
        let exact_paths: std::collections::HashSet<&str> = merged
            .iter()
            .filter(|(a, ..)| Arc::ptr_eq(a, arc))
            .map(|(_, path, ..)| path.as_str())
            .collect();
        let Some((title_list, list)) = arc.with_xapian(|h| -> Result<_, ToolError> {
            let mut title_list = Vec::new();
            if !terms.is_empty() {
                if let Some(title_db) = &h.title {
                    // Sub-band queries, in band order: the AND over ALL the
                    // stemmed terms (multi-word queries only - with one term
                    // AND and OR are the same query), then the OR of the
                    // terms. Every AND hit reappears in the OR's results (a
                    // title with all the words also matches any subset of
                    // them), so the OR pass skips the docids the AND pass
                    // already reported.
                    let and_query = if terms.len() >= 2 {
                        Some(combine_terms(Operator::And, &terms)?)
                    } else {
                        None
                    };
                    let or_query = combine_terms(Operator::Or, &terms)?;
                    let mut taken: std::collections::HashSet<u32> = std::collections::HashSet::new();
                    let mut enquire = Enquire::new(title_db)?;
                    enquire.set_sort_by_relevance();
                    for tquery in [and_query, Some(or_query)].into_iter().flatten() {
                        enquire.set_query(&tquery)?;
                        let mset = enquire.get_mset(0, SEARCH_LIMIT, 0)?;
                        for j in 0..mset.size() {
                            // One band slot per title-index document.
                            if !taken.insert(mset.docid(j)) {
                                continue;
                            }
                            let mut doc = mset.document(j)?;
                            // The title-index document's data is the article
                            // path and its value slot 0 the title (same shape
                            // as the full-text index, one shared docid space).
                            // Should a producer leave either empty, the
                            // full-text document of the same docid fills it in.
                            let mut path = doc.data_str()?;
                            let mut title = String::from_utf8_lossy(&doc.value(0)?).into_owned();
                            if path.is_empty() || title.is_empty() {
                                if let Ok(mut ftdoc) = h.fulltext.get_document(mset.docid(j)) {
                                    if path.is_empty() {
                                        path = ftdoc.data_str()?;
                                    }
                                    if title.is_empty() {
                                        title =
                                            String::from_utf8_lossy(&ftdoc.value(0)?).into_owned();
                                    }
                                }
                            }
                            if path.is_empty() {
                                continue;
                            }
                            // An article already reported as an exact
                            // title/URL match keeps that first-band slot:
                            // its exact-match title (the query itself when
                            // the directory entry title is empty) need not
                            // equal the index title, which the final
                            // normalized-title dedupe cannot see through.
                            if exact_paths.contains(path.as_str()) {
                                continue;
                            }
                            title_list.push((path, title));
                        }
                    }
                }
            }

            // The full-text band: the parsed OR query, with this archive's
            // all-words branch OR-ed in when its term statistics produce
            // one (see `all_words_branch`).
            let mut enquire = Enquire::new(&h.fulltext)?;
            match all_words_branch(&h.fulltext, &terms, AND_TERM_MAX_DF_FRAC)? {
                Some(and_query) => {
                    enquire.set_query(&Query::combine(Operator::Or, &and_query, &xquery)?)?
                }
                None => enquire.set_query(&xquery)?,
            }
            enquire.set_sort_by_relevance();
            let mset = enquire.get_mset(0, SEARCH_LIMIT, 0)?;
            let mut list = Vec::with_capacity(mset.size() as usize);
            for (j, m) in mset.iter().enumerate() {
                let mut doc = mset.document(j as u32)?;
                let path = doc.data_str()?;
                if path.is_empty() {
                    continue;
                }
                let title = String::from_utf8_lossy(&doc.value(0)?).into_owned();
                list.push((m.weight, path, title));
            }
            Ok((title_list, list))
        })?
        else {
            continue;
        };
        title_lists.push((arc, title_list));
        per_archive.push((arc, list));
    }

    if per_archive.is_empty() {
        return Err(ToolError::Internal(format!(
            "no ZIM files with a Xapian full-text index were found in {}",
            library.root.display()
        )));
    }

    // Xapian weights are computed from per-database statistics and are not
    // comparable across archives, so each band merges its archives' ranked
    // lists by rotation instead of by weight: every archive contributes its
    // best match before any archive contributes its second best. The bands
    // keep their order: exact title/URL probe hits (already in `merged`),
    // then title-index matches, then full-text matches.
    let mut rank = 0usize;
    loop {
        let mut picked = false;
        for (arc, list) in &title_lists {
            if let Some((path, title)) = list.get(rank) {
                merged.push((arc, path.clone(), title.clone(), HitKind::TitleIndex));
                picked = true;
            }
        }
        if !picked {
            break;
        }
        rank += 1;
    }
    let mut rank = 0usize;
    loop {
        let mut picked = false;
        for (arc, list) in &per_archive {
            if let Some((_, path, title)) = list.get(rank) {
                merged.push((arc, path.clone(), title.clone(), HitKind::Fulltext));
                picked = true;
            }
        }
        if !picked {
            break;
        }
        rank += 1;
    }

    // The same article is often present in several archives (e.g. an HTML
    // and a Markdown edition of the same ZIM), and one article can surface
    // in several bands of the same archive (its title index document AND
    // its full-text document both match): dedupe by normalized title so
    // each article is reported once, from the band/archives ranked first.
    // Exact matches come first, so duplicates of them drop out here, and a
    // title-index hit shadows the same article's full-text hit.
    let mut seen = std::collections::HashSet::new();
    merged.retain(|(_, path, title, _)| {
        let key = if title.is_empty() { path.as_str() } else { title.as_str() };
        seen.insert(html::normalize(key))
    });
    merged.truncate(SEARCH_LIMIT as usize);

    let mut hits = Vec::with_capacity(merged.len());
    for (arc, path, idx_title, kind) in &merged {
        // Exact and title-index hits are title matches: the lead is the
        // right preview and there is nothing to point at section-wise.
        let title_match = *kind != HitKind::Fulltext;
        let (entry_title, mime, bytes) = match arc.article_preview(path, HIT_READ_BYTES) {
            Ok(Some((entry_title, mime, bytes))) => (entry_title, mime, bytes),
            _ => (String::new(), None, Vec::new()),
        };
        // An exact match's title is already final - the redirect's own
        // title, not the target's (which `entry_title` is). For the rest,
        // prefer the entry's own title; many openZIM archives leave the
        // directory-entry title empty and only carry the title in the
        // index (which we already read as `idx_title`).
        let title = if *kind == HitKind::ExactTitle {
            idx_title.clone()
        } else if !entry_title.is_empty() {
            entry_title
        } else if !idx_title.is_empty() {
            idx_title.clone()
        } else {
            path.clone()
        };
        let article = String::from_utf8_lossy(&bytes);
        // Markdown editions carry plain Markdown, not HTML: pick the matching
        // splitter so the paragraphs and section names are free of markup.
        let is_markdown = mime.as_deref().is_some_and(|m| m.contains("markdown"));
        let (preview, sections) = hit_preview(&article, &terms, title_match, &mut stemmer, is_markdown);
        hits.push(SearchHit {
            zim: arc.name.clone(),
            path: path.clone(),
            title,
            preview,
            sections,
        });
    }
    Ok(SearchResults { results: hits })
}

/// How many of `text`'s word occurrences are query terms - the paragraph's
/// match count, stemmed the same way the index and the query are.
fn para_matches(text: &str, terms: &[String], stem: &mut Stemmer) -> usize {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !STOPWORDS.iter().any(|s| s.eq_ignore_ascii_case(w)))
        .filter(|w| terms.iter().any(|t| t == stem.stem(w)))
        .count()
}

/// Whether every query term occurs (stemmed) somewhere in `text`'s words.
fn covers_all_terms(text: &str, terms: &[String], stem: &mut Stemmer) -> bool {
    let mut covered = vec![false; terms.len()];
    let mut left = terms.len();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() || STOPWORDS.iter().any(|s| s.eq_ignore_ascii_case(word)) {
            continue;
        }
        let stemmed = stem.stem(word);
        for (k, term) in terms.iter().enumerate() {
            if !covered[k] && term.as_str() == stemmed {
                covered[k] = true;
                left -= 1;
                if left == 0 {
                    return true;
                }
            }
        }
    }
    left == 0
}

/// Split a paragraph into sentences: a sentence ends after `.`, `!`, or `?`
/// followed by whitespace or the paragraph's end; a paragraph without any
/// terminator is a single sentence. (Abbreviations like "U.S." over-split,
/// which is acceptable for a preview.)
fn sentences(paragraph: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in paragraph.char_indices() {
        if matches!(c, '.' | '!' | '?') {
            let after = i + c.len_utf8();
            if after == paragraph.len() || paragraph[after..].starts_with(char::is_whitespace) {
                out.push(paragraph[start..after].trim());
                start = after;
            }
        }
    }
    if start < paragraph.len() {
        out.push(paragraph[start..].trim());
    }
    out.into_iter().filter(|s| !s.is_empty()).collect()
}

/// The `preview`/`sections` pair of one search hit, from the article's raw
/// text (`is_markdown` picks the Markdown or the HTML splitter):
///
/// - a title match (`title_match`: the query named the article exactly, or
///   the hit came from the title index) gets the first intro paragraph as
///   its preview and no sections - the title already said everything;
/// - otherwise, when every query term occurs in the first intro paragraph,
///   same: the lead already covers the query;
/// - otherwise every region is scanned - the intro first (under its
///   `_intro` name, `html::INTRO_SECTION`), then the body sections in
///   document order. The regions holding at least one query match are
///   reported as `sections`, and the preview is the best-matching sentence
///   (most matching word occurrences across all regions; ties keep the
///   earliest, so an intro sentence beats a body one), followed by its
///   paragraph's next sentences while the length stays under `INTRO_CHARS`
///   and truncated to `INTRO_CHARS` - the matched sentence sits at the
///   front, so the truncation cannot hide the words that matched;
/// - when no region matches either (only the title in the index matched
///   the query), the preview falls back to the first intro paragraph. Regions
///   without paragraphs degrade to an empty preview, never a panic.
///
/// The intro's paragraphs are extracted first and alone: the two title/lead
/// cases above - the common ones - never need the full article split.
fn hit_preview(
    article: &str,
    terms: &[String],
    title_match: bool,
    stem: &mut Stemmer,
    is_markdown: bool,
) -> (String, Option<Vec<String>>) {
    let intro = if is_markdown {
        markdown::intro_paragraphs(article, PARA_MATCH_CHARS)
    } else {
        html::intro_paragraphs(article, PARA_MATCH_CHARS)
    };
    let lead = || {
        intro.first()
            .map(|p| p.chars().take(INTRO_CHARS).collect())
            .unwrap_or_default()
    };
    if title_match || intro.first().is_some_and(|p| covers_all_terms(p, terms, stem)) {
        return (lead(), None);
    }
    let secs = if is_markdown {
        markdown::sections(article, PARA_MATCH_CHARS)
    } else {
        html::sections(article, PARA_MATCH_CHARS)
    };
    // Paragraphs are scored whole only for the `sections` reporting; the
    // preview picks the best-matching SENTENCE, so that a match in the
    // middle of a long paragraph is still visible in the preview.
    let mut best_count = 0usize;
    // The best-matching sentence, as (its paragraph, its index within it).
    let mut best_sent: Option<(&String, usize)> = None;
    let mut names: Vec<String> = Vec::new();
    for (name, paras) in &secs {
        let mut matched = false;
        for para in paras {
            if para_matches(para, terms, stem) > 0 {
                matched = true;
                for (i, s) in sentences(para).into_iter().enumerate() {
                    let n = para_matches(s, terms, stem);
                    if n > best_count {
                        // Ties keep the earlier sentence: only a strictly
                        // better count replaces the incumbent.
                        best_count = n;
                        best_sent = Some((para, i));
                    }
                }
            }
        }
        if matched {
            names.push(name.clone());
        }
    }
    match best_sent {
        Some((para, first)) => {
            let sent = sentences(para);
            let mut preview = String::new();
            for s in &sent[first..] {
                if preview.chars().count() >= INTRO_CHARS {
                    break;
                }
                if !preview.is_empty() {
                    preview.push(' ');
                }
                preview.push_str(s);
            }
            (preview.chars().take(INTRO_CHARS).collect(), Some(names))
        }
        None => (lead(), None),
    }
}

// ---------------------------------------------------------------------------
// Tests: end-to-end over a synthetic archive carrying a real Xapian index
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tools::{
        ZimGetParams, ZimGetSectionParams, ZimGetSectionTool, ZimGetTool, ZimMcpServer,
        ZimSearchParams, ZimSearchTool,
    };
    use crate::zim::testutil::{build_archive, build_archive_indexes, TestEntry, TestRedirect};
    use rmcp::handler::server::router::tool::AsyncTool;
    use std::future::Future;
    use xapian2::{Database, Document, Enquire, WritableDatabase};

    /// Run an async tool invocation to completion on this thread: the tools
    /// are async (each hops to a blocking thread), and these tests are sync
    /// `#[test]`s, so each invocation gets its own tiny current-thread
    /// runtime to await in.
    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    const APPLE_HTML: &str = "<html><head><title>Apple</title></head><body>\
        <h1>Apple</h1>\
        <p>An <b>apple</b> is the fruit of &lt;rosaceae&gt; trees.</p>\
        <h2 id=\"History\">History</h2>\
        <p>Apples have been cultivated for 10,000 years.</p>\
        <h3>Domestication</h3><p>Wild apples grew in Kazakhstan.</p>\
        <h2 id=\"Computers\">Computers</h2>\
        <p>Computing devices also go by that name.</p>\
        </body></html>";

    const BANANA_HTML: &str = "<html><body><h1>Banana</h1>\
        <h2 id=\"Growth\">Growth</h2>\
        <p>Banana trees are actually tall herbaceous plants.</p>\
        </body></html>";

    const CHERRY_HTML: &str = "<html><body><h1>Cherry</h1>\
        <p>A cherry is the fruit of trees of the genus <i>Prunus</i>.</p>\
        </body></html>";

    const NITROGEN_HTML: &str = "<html><body><h1>Nitrogen</h1>\
        <p>Nitrogen is a colorless, odorless gas.</p>\
        </body></html>";

    const ATMOSPHERE_HTML: &str = "<html><body><h1>Atmosphere</h1>\
        <p>The atmosphere is mostly nitrogen and oxygen.</p>\
        </body></html>";

    const AERONAUTICS_HTML: &str = "<html><body><h1>Aeronautics</h1>\
        <p>Aeronautics is the science of flight.</p>\
        </body></html>";

    /// For the title-index tests: an article whose title contains all the
    /// query's words ("Nitrogen Gas Effects" for "effects of nitrogen gas")
    /// with the matching words also in its body (so its full-text document
    /// matches too and the cross-band dedupe has something to do), and an
    /// article that only matches the query in its body.
    const NITROGEN_GAS_EFFECTS_HTML: &str = "<html><body><h1>Nitrogen Gas Effects</h1>\
        <p>Nitrogen gas surrounds us all.</p>\
        <h2>Everywhere</h2>\
        <p>The effects of nitrogen gas are unavoidable.</p>\
        </body></html>";

    const WEATHER_HTML: &str = "<html><body><h1>Weather</h1>\
        <p>Weather forecasts describe the effects of air pressure.</p>\
        <p>Gas laws explain the atmosphere.</p>\
        </body></html>";

    /// An article whose intro has two paragraphs: a query can match the
    /// first paragraph (the lead fast path), a later one (the intro
    /// reported as the region `_intro`), or the intro and a body section.
    const SALT_HTML: &str = "<html><body><h1>Salt</h1>\
        <p>Salt is a mineral composed primarily of sodium chloride.</p>\
        <p>The Himalaya range holds vast deposits of rock salt.</p>\
        <h2>Formation</h2>\
        <p>Salt beds form when seas evaporate.</p>\
        <h2>Uses</h2>\
        <p>People season their food with it.</p>\
        </body></html>";

    /// Articles whose best-matching paragraph holds several sentences, with
    /// the query matching a mid-paragraph sentence: the preview must START
    /// with that sentence, which the old paragraph preview did not (the
    /// matched words sat mid-paragraph). The Volcano lead does not cover
    /// its query, so the lead fallback does not fire.
    const VOLCANO_HTML: &str = "<html><body><h1>Volcano</h1>\
        <p>Volcanoes are openings in the crust.</p>\
        <p>Molten rock rises from chambers below. Eruptions reshape the \
        land. Ash clouds can ground aircraft. Farmers fear the fallout.</p>\
        </body></html>";

    const GLACIER_MD: &str = "\
# Glacier

A glacier is a body of dense ice.

## Movement

Glaciers move under their own weight. The flow is slower than a river. \
Meltwater streams out of the ice.
";

    /// Bodies for the all-words-tier pathology test (`e2e_search_common_word_does_not_block_all_words_tier`).
    const BEACONSFIELD_QUEBEC_HTML: &str = "<html><body><h1>Beaconsfield, Quebec</h1>\
        <p>Beaconsfield is a suburban borough of Montreal, Quebec, Canada.</p></body></html>";

    const CANADIAN_AMATEUR_HTML: &str = "<html><body><h1>Canadian Amateur Championship</h1>\
        <p>The Canadian Amateur Championship is a golf tournament.</p></body></html>";

    const FREDERICK_STANLEY_HTML: &str = "<html><body><h1>Frederick Stanley</h1>\
        <p>Frederick Stanley, 16th Earl of Derby, was Governor General of Canada.</p></body></html>";

    fn search(server: &ZimMcpServer, query: &str) -> Vec<SearchHit> {
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": query }),
        )
        .unwrap();
        block_on(ZimSearchTool::invoke(server, params)).unwrap().results
    }

    /// Build a single-file glass Xapian index, the way openZIM does: the
    /// document data is the article's full path inside the archive, the
    /// title sits in value slot 0, and the terms are unprefixed Porter2
    /// stems, exactly as libzim indexes with STEM_ALL ("appl" is the stem
    /// of "apple", "comput" of "computing").
    fn make_index(docs: &[(&str, &str, &str)]) -> Vec<u8> {
        let dir = tempfile::tempdir().unwrap();
        let db_dir = dir.path().join("db");
        {
            let mut wdb = WritableDatabase::create(&db_dir).unwrap();
            for (path, terms, title) in docs {
                let mut doc = Document::new().unwrap();
                doc.set_data(*path).unwrap();
                if !title.is_empty() {
                    doc.set_value(0, *title).unwrap();
                }
                for t in terms.split_whitespace() {
                    doc.add_term(t, 1).unwrap();
                }
                wdb.add_document(&doc).unwrap();
            }
            wdb.commit().unwrap();
        }
        let single = dir.path().join("single.xdb");
        let db = xapian2::Database::open(&db_dir).unwrap();
        db.compact_single_file(&single).unwrap();
        std::fs::read(&single).unwrap()
    }

    pub(crate) fn test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Apple", "appl histori 10 000 year domest wild kazakhstan comput devic nam", "Apple"),
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search() {
        let (server, _keep) = test_server();

        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "apple" }),
        )
        .unwrap();
        let hits = block_on(ZimSearchTool::invoke(&server, params)).unwrap().results;
        assert!(!hits.is_empty(), "search must return hits");
        let first = &hits[0];
        assert_eq!(first.zim, "test.zim");
        assert_eq!(first.path, "C/Apple");
        assert_eq!(first.title, "Apple");
        // An exact title match reports the lead paragraph as its preview.
        assert_eq!(first.preview, "An apple is the fruit of <rosaceae> trees.");
        assert_eq!(first.sections, None);

        // Stemmed query ("computing" -> "comput").
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "computing" }),
        )
        .unwrap();
        let hits = block_on(ZimSearchTool::invoke(&server, params)).unwrap().results;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "C/Apple");

        // Unrelated term: no hits.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "zzzzz" }),
        )
        .unwrap();
        let hits = block_on(ZimSearchTool::invoke(&server, params)).unwrap().results;
        assert!(hits.is_empty());

        // OR semantics: two terms from different articles.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "banana apple" }),
        )
        .unwrap();
        let hits = block_on(ZimSearchTool::invoke(&server, params)).unwrap().results;
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn e2e_search_empty_query_is_invalid() {
        let (server, _keep) = test_server();
        let params = ZimSearchParams { query: "   ".into() };
        assert!(matches!(
            block_on(ZimSearchTool::invoke(&server, params)),
            Err(ToolError::InvalidArgument(_))
        ));
    }

    #[test]
    fn e2e_search_concurrent_threads_on_one_library() {
        // Concurrent searches against ONE shared library: every thread runs
        // `search` at the same time (synchronized on the barrier) and must
        // get the correct results. Each search checks a Xapian handle out of
        // the per-archive pool and uses it alone; when the archives shared
        // one cached handle, concurrent searches corrupted the database
        // state and crashed the process - Xapian does not support concurrent
        // calls on one Database object (see xapian2/README.md).
        let (server, _keep) = test_server();
        let queries = ["apple", "apple", "computing", "banana apple", "apple", "computing"];
        let barrier = Arc::new(std::sync::Barrier::new(queries.len()));
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for query in queries {
                let server = &server;
                let barrier = barrier.clone();
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    let hits = search(server, query);
                    match query {
                        // Stemmed full-text hit ("computing" -> "comput").
                        "computing" => {
                            assert_eq!(hits.len(), 1, "{hits:?}");
                            assert_eq!(hits[0].path, "C/Apple");
                        }
                        // OR of two terms from different articles.
                        "banana apple" => {
                            assert_eq!(hits.len(), 2, "{hits:?}");
                        }
                        // Exact title match, identical on every thread.
                        _ => {
                            assert_eq!(hits[0].path, "C/Apple", "{hits:?}");
                            assert_eq!(hits[0].title, "Apple");
                            assert_eq!(hits[0].preview, "An apple is the fruit of <rosaceae> trees.");
                        }
                    }
                }));
            }
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn e2e_search_title_falls_back_to_index_title() {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Nitrogen", "nitrogen gas inert", "Nitrogen"),
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
        ]);
        let content = [
            // Empty directory-entry title, as in modern openZIM archives.
            TestEntry { namespace: b'C', url: "Nitrogen", title: "", mime: 0, body: NITROGEN_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        let server = ZimMcpServer::new(library);

        // Not an exact title/URL match ("nitrogen gas" is nobody's title),
        // so the hit comes from the full-text index: with the directory
        // title empty, the title falls back to the index title (value slot
        // 0). (The plain query "nitrogen" now resolves as an exact match.)
        let hits = search(&server, "nitrogen gas");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Nitrogen", "{hits:?}");
    }

    /// An archive with BOTH embedded indexes, the way libzim builds them:
    /// the full-text index over article bodies plus a title index
    /// (`X/title/xapian`) with one document per article whose terms are the
    /// title's words and whose value slot 0 is the title (the document data
    /// is the article path in both indexes).
    fn title_index_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Nitrogen_Gas_Effects", "nitrogen gas surround effect unavoid", "Nitrogen Gas Effects"),
            ("C/Weather", "weather forecast describ effect air pressur gas law explan atmospher", "Weather"),
        ]);
        let titles = make_index(&[
            ("C/Nitrogen_Gas_Effects", "nitrogen gas effect", "Nitrogen Gas Effects"),
            ("C/Weather", "weather", "Weather"),
        ]);
        let content = [
            // Empty directory-entry title, as in modern openZIM archives:
            // the title lives in the indexes (value slot 0) only.
            TestEntry { namespace: b'C', url: "Nitrogen_Gas_Effects", title: "", mime: 0, body: NITROGEN_GAS_EFFECTS_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Weather", title: "Weather", mime: 0, body: WEATHER_HTML.as_bytes() },
        ];
        let bytes = build_archive_indexes(&["text/html"], &content, &[], 0, Some(&index), Some(&titles));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_title_index_ranks_title_matches_above_fulltext() {
        let (server, _keep) = title_index_test_server();

        // "effects of nitrogen gas" is nobody's title or URL (the article is
        // "Nitrogen Gas Effects"), but the article's title contains every
        // query word: its title-index document outranks the full-text-only
        // matches (whose titles carry no query word), and the hit is styled
        // as a title match.
        let hits = search(&server, "effects of nitrogen gas");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].path, "C/Nitrogen_Gas_Effects", "{hits:?}");
        assert_eq!(hits[0].title, "Nitrogen Gas Effects");
        // Title-match semantics: the lead paragraph and no sections - NOT
        // the Everywhere section (the "effects ..." paragraph with more
        // query matches) the full-text band would have reported.
        assert_eq!(hits[0].preview, "Nitrogen gas surrounds us all.");
        assert_eq!(hits[0].sections, None);
        let json = serde_json::to_string(&hits[0]).unwrap();
        assert!(!json.contains("sections"), "{json}");
        // The article also matched in the full-text band (its body carries
        // the query words): reported exactly once, from the title band.
        assert_eq!(hits.iter().filter(|h| h.path == "C/Nitrogen_Gas_Effects").count(), 1);
        // The full-text-only match follows, with full-text hit semantics.
        assert_eq!(hits[1].path, "C/Weather", "{hits:?}");
        assert_eq!(hits[1].title, "Weather");
        assert_eq!(hits[1].sections, Some(vec!["_intro".to_string()]));

        // No usable terms: the title band is skipped by design and the
        // full-text band matches nothing (stopwords are stopped at index
        // time) - no hits, no error.
        let hits = search(&server, "the of");
        assert!(hits.is_empty(), "{hits:?}");
    }

    /// An archive whose title band's OR query alone ranks a PARTIAL title
    /// match above the all-words title match: the short "Quebec, Quebec"
    /// title repeats its query word, while the word it is missing - "city" -
    /// is ultra-common (seven of the eight titles contain it), so its BM25
    /// contribution collapses. OR-only weights measured on exactly this
    /// document set: "Quebec, Quebec" 1.298 vs "Quebec City" 1.081.
    fn title_and_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let titles = make_index(&[
            ("C/Quebec,_Quebec", "quebec quebec", "Quebec, Quebec"),
            ("C/Quebec_City", "quebec citi", "Quebec City"),
            ("C/New_York_City", "new york citi", "New York City"),
            ("C/Kansas_City", "kansas citi", "Kansas City"),
            ("C/Mexico_City", "mexico citi", "Mexico City"),
            ("C/Atlantic_City", "atlantic citi", "Atlantic City"),
            ("C/Jersey_City", "jersey citi", "Jersey City"),
            ("C/Salt_Lake_City", "salt lake citi", "Salt Lake City"),
        ]);
        let index = make_index(&[
            ("C/Quebec,_Quebec", "quebec quebec appear twice own titl", "Quebec, Quebec"),
            ("C/Quebec_City", "quebec citi capit provinc", "Quebec City"),
            ("C/New_York_City", "new york citi largest unit state", "New York City"),
            ("C/Kansas_City", "kansas citi", "Kansas City"),
            ("C/Mexico_City", "mexico citi", "Mexico City"),
            ("C/Atlantic_City", "atlantic citi", "Atlantic City"),
            ("C/Jersey_City", "jersey citi", "Jersey City"),
            ("C/Salt_Lake_City", "salt lake citi", "Salt Lake City"),
        ]);
        let content = [
            // Empty directory-entry titles, as in modern openZIM archives:
            // the titles live in the indexes (value slot 0) only.
            TestEntry { namespace: b'C', url: "Quebec,_Quebec", title: "", mime: 0, body: b"<html><body><h1>Quebec, Quebec</h1><p>Quebec, Quebec appears twice in its own title.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Quebec_City", title: "", mime: 0, body: b"<html><body><h1>Quebec City</h1><p>Quebec City is the capital of the province.</p></body></html>" },
            TestEntry { namespace: b'C', url: "New_York_City", title: "", mime: 0, body: b"<html><body><h1>New York City</h1><p>New York City is the largest in the United States.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Kansas_City", title: "", mime: 0, body: b"<html><body><h1>Kansas City</h1><p>Kansas City straddles two states.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Mexico_City", title: "", mime: 0, body: b"<html><body><h1>Mexico City</h1><p>Mexico City is the capital of Mexico.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Atlantic_City", title: "", mime: 0, body: b"<html><body><h1>Atlantic City</h1><p>Atlantic City is a resort town.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Jersey_City", title: "", mime: 0, body: b"<html><body><h1>Jersey City</h1><p>Jersey City sits opposite Manhattan.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Salt_Lake_City", title: "", mime: 0, body: b"<html><body><h1>Salt Lake City</h1><p>Salt Lake City hosts a famous temple.</p></body></html>" },
        ];
        let bytes = build_archive_indexes(&["text/html"], &content, &[], 0, Some(&index), Some(&titles));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_title_band_ranks_all_words_titles_first() {
        let (server, _keep) = title_and_test_server();

        // Under OR-only BM25 the partial title match outranks the all-words
        // title match (weights in the builder comment). Within the band, the
        // AND sub-band must put the title containing BOTH query words - the
        // AND query's only hit - ahead of it. ("quebec cities" is
        // deliberately nobody's URL, so the hits really come from the band.)
        let hits = search(&server, "quebec cities");
        assert_eq!(hits.len(), 8, "{hits:?}");
        assert_eq!(hits[0].path, "C/Quebec_City", "{hits:?}");
        assert_eq!(hits[0].title, "Quebec City");
        assert_eq!(hits[1].path, "C/Quebec,_Quebec", "{hits:?}");
        assert_eq!(hits[1].title, "Quebec, Quebec");
        // Title-band hits are title matches: the lead as the preview, no
        // sections - for the all-words hit and the partial one alike.
        assert_eq!(hits[0].preview, "Quebec City is the capital of the province.");
        assert_eq!(hits[0].sections, None);
        assert_eq!(hits[1].sections, None);

        // A single-term query runs the one OR query only - the unchanged
        // BM25 order (the short title repeating the term first).
        let hits = search(&server, "quebec");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].path, "C/Quebec,_Quebec", "{hits:?}");
        assert_eq!(hits[1].path, "C/Quebec_City", "{hits:?}");
    }

    /// An archive where BM25 alone ranks the wrong article first: the
    /// "Atmosphere" document repeats the term "nitrogen" seven times, so it
    /// out-scores the "Nitrogen" article for the query "nitrogen". "NACA"
    /// and "Usa" are redirects onto the Aeronautics article; redirect
    /// entries live only in the directory, not in the search index.
    fn exact_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        // Index terms are the stems the query parser produces ("atmosphere"
        // -> "atmospher"), unprefixed, as libzim indexes with STEM_ALL.
        let index = make_index(&[
            ("C/Nitrogen", "nitrogen colorless odorless gas", "Nitrogen"),
            ("C/Atmosphere", "nitrogen nitrogen nitrogen nitrogen nitrogen nitrogen nitrogen naca naca atmospher", "Atmosphere"),
            ("C/Aeronautics", "aeronautics naca aviation wind tunnel flight", "Aeronautics"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Nitrogen", title: "Nitrogen", mime: 0, body: NITROGEN_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Atmosphere", title: "Atmosphere", mime: 0, body: ATMOSPHERE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Aeronautics", title: "Aeronautics", mime: 0, body: AERONAUTICS_HTML.as_bytes() },
        ];
        let redirects = [
            TestRedirect { namespace: b'C', url: "NACA", title: "NACA", target_content: 2 },
            // Empty directory title, as in modern openZIM archives.
            TestRedirect { namespace: b'C', url: "Usa", title: "", target_content: 2 },
        ];
        let bytes = build_archive(&["text/html"], &content, &redirects, 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_exact_title_ranks_first() {
        let (server, _keep) = exact_test_server();

        // "nitrogen" is exactly the title/URL of C/Nitrogen, yet BM25 ranks
        // the Atmosphere document first (it repeats the term seven times):
        // the exact match must come out on top.
        let hits = search(&server, "nitrogen");
        assert_eq!(hits[0].path, "C/Nitrogen", "{hits:?}");
        assert_eq!(hits[0].title, "Nitrogen");
        assert_eq!(hits[0].zim, "test.zim");
        // An exact match is a title match: the lead paragraph, no sections.
        assert_eq!(hits[0].preview, "Nitrogen is a colorless, odorless gas.");
        assert_eq!(hits[0].sections, None);
        assert!(!serde_json::to_string(&hits[0]).unwrap().contains("sections"));
        // The BM25 runner-up is still reported, behind the exact match.
        assert_eq!(hits[1].path, "C/Atmosphere", "{hits:?}");
    }

    #[test]
    fn e2e_search_exact_redirect_title_ranks_first() {
        let (server, _keep) = exact_test_server();

        // "NACA" is a redirect (directory title "NACA") onto the Aeronautics
        // article. Redirects are not in the full-text index, so without the
        // directory lookup this query would report Atmosphere first (it
        // mentions "naca" twice).
        let hits = search(&server, "NACA");
        assert_eq!(hits[0].path, "C/NACA", "{hits:?}");
        assert_eq!(hits[0].title, "NACA");
        // The preview is built from the redirect target's content.
        assert_eq!(hits[0].preview, "Aeronautics is the science of flight.");
        assert_eq!(hits[0].sections, None);
        // Fulltext hits follow in BM25 order.
        assert_eq!(hits[1].path, "C/Atmosphere", "{hits:?}");
        assert_eq!(hits[2].path, "C/Aeronautics", "{hits:?}");

        // A redirect with an empty directory title: the query becomes the
        // title, and the all-lowercase query still finds the redirect via
        // the case variants of its URL.
        let hits = search(&server, "usa");
        assert_eq!(hits[0].path, "C/Usa", "{hits:?}");
        assert_eq!(hits[0].title, "usa");
    }

    #[test]
    fn e2e_search_query_without_exact_match_keeps_ranking() {
        let (server, _keep) = exact_test_server();

        // "nitrogen atmosphere" is nobody's title or URL, so the ranking is
        // the unchanged BM25 order: the document containing both terms
        // first, the one containing only "nitrogen" second.
        let hits = search(&server, "nitrogen atmosphere");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].path, "C/Atmosphere", "{hits:?}");
        assert_eq!(hits[1].path, "C/Nitrogen", "{hits:?}");
        // The Atmosphere lead covers the whole query ("The atmosphere is
        // mostly nitrogen and oxygen.") in its first paragraph: an intro
        // match, so no sections. The Nitrogen lead only covers "nitrogen"
        // - a partial intro match is reported like any other, as the
        // region _intro.
        assert_eq!(hits[0].sections, None, "{:?}", hits[0]);
        assert_eq!(hits[0].preview, "The atmosphere is mostly nitrogen and oxygen.");
        assert_eq!(hits[1].sections, Some(vec!["_intro".to_string()]), "{:?}", hits[1]);
        assert_eq!(hits[1].preview, "Nitrogen is a colorless, odorless gas.");
    }

    #[test]
    fn e2e_search_intro_match_reports_lead_without_sections() {
        // Both query terms occur in the lead paragraph ("An apple is the
        // fruit of <rosaceae> trees."): an intro match on a full-text hit
        // (nobody's title is "apple fruit"), so the preview is the lead and
        // the serialized JSON carries no "sections" field at all.
        let (server, _keep) = test_server();
        let hits = search(&server, "apple fruit");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Apple");
        assert_eq!(hits[0].preview, "An apple is the fruit of <rosaceae> trees.");
        assert_eq!(hits[0].sections, None);
        let json = serde_json::to_string(&hits[0]).unwrap();
        assert!(!json.contains("sections"), "{json}");
    }

    #[test]
    fn e2e_search_section_match_reports_sections_and_best_paragraph() {
        // The query term appears only in a later section of the article
        // ("Wild apples grew in Kazakhstan." under History): the intro
        // cannot cover it, so the hit reports the matched sections and the
        // best-matching paragraph, not the lead.
        let (server, _keep) = test_server();
        let hits = search(&server, "kazakhstan");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Apple");
        // The paragraph sits under History, whose range includes the nested
        // Domestication heading: both sections report the match.
        assert_eq!(
            hit.sections,
            Some(vec!["History".to_string(), "Domestication".to_string()]),
            "{hit:?}"
        );
        assert_eq!(hit.preview, "Wild apples grew in Kazakhstan.");
        // The serialized JSON carries the section names.
        let json = serde_json::to_string(hit).unwrap();
        assert!(json.contains(r#""sections":["History","Domestication"]"#), "{json}");
    }

    /// An archive whose only article (`SALT_HTML`) has a two-paragraph
    /// intro, for the intro-matching search semantics.
    fn intro_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        // Index terms are the stems the query parser produces ("beds" ->
        // "bed"), unprefixed, as libzim indexes with STEM_ALL.
        let index = make_index(&[(
            "C/Salt",
            "salt mineral chlorid sodium himalaya deposit rock bed form sea season food",
            "Salt",
        )]);
        let content = [TestEntry {
            namespace: b'C',
            url: "Salt",
            title: "Salt",
            mime: 0,
            body: SALT_HTML.as_bytes(),
        }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_intro_match_beyond_first_paragraph_reports_intro_section() {
        // "himalaya" matches only the intro's second paragraph: the preview is
        // that paragraph and the intro is reported as the matching region
        // _intro - not the lead, and not without sections.
        let (server, _keep) = intro_test_server();
        let hits = search(&server, "himalaya");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Salt");
        assert_eq!(hit.sections, Some(vec!["_intro".to_string()]), "{hit:?}");
        assert_eq!(hit.preview, "The Himalaya range holds vast deposits of rock salt.");
        let json = serde_json::to_string(hit).unwrap();
        assert!(json.contains(r#""sections":["_intro"]"#), "{json}");
    }

    #[test]
    fn e2e_search_intro_and_section_matches_report_both() {
        // "salt beds" matches the intro's second paragraph ("salt") and the
        // Formation section ("Salt beds ..."): both regions are reported,
        // _intro first, and the preview is the paragraph with the most
        // query-term occurrences across all regions (Formation's, two
        // against the intro paragraphs' one).
        let (server, _keep) = intro_test_server();
        let hits = search(&server, "salt beds");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Salt");
        assert_eq!(
            hit.sections,
            Some(vec!["_intro".to_string(), "Formation".to_string()]),
            "{hit:?}"
        );
        assert_eq!(hit.preview, "Salt beds form when seas evaporate.");
    }

    #[test]
    fn e2e_search_preview_starts_with_best_sentence() {
        // The best-matching paragraph holds several sentences and the query
        // matches a mid-paragraph one: the preview starts with that sentence
        // (a 300-character preview of the whole paragraph would cut the
        // matched words off), continued with the paragraph's remaining
        // sentences. Covered for an HTML article (match in the intro's
        // second paragraph) and a Markdown one (match in a body section).
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Volcano", "volcano crust molten rock erupt reshape land ash cloud aircraft farmer", "Volcano"),
            ("C/Glacier", "glacier ice movement weight flow river meltwater stream", "Glacier"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Volcano", title: "Volcano", mime: 0, body: VOLCANO_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Glacier", title: "Glacier", mime: 1, body: GLACIER_MD.as_bytes() },
        ];
        let bytes = build_archive(&["text/html", "text/markdown"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        let server = ZimMcpServer::new(library);

        let hits = search(&server, "aircraft");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Volcano");
        assert_eq!(hits[0].sections, Some(vec!["_intro".to_string()]));
        assert!(
            hits[0].preview.starts_with("Ash clouds can ground aircraft."),
            "{:?}",
            hits[0].preview
        );
        assert_eq!(
            hits[0].preview,
            "Ash clouds can ground aircraft. Farmers fear the fallout."
        );

        let hits = search(&server, "river");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Glacier");
        assert_eq!(hits[0].sections, Some(vec!["Movement".to_string()]));
        assert!(
            hits[0].preview.starts_with("The flow is slower than a river."),
            "{:?}",
            hits[0].preview
        );
        assert_eq!(
            hits[0].preview,
            "The flow is slower than a river. Meltwater streams out of the ice."
        );
    }

    /// A single-archive server whose full-text index holds `fillers` filler
    /// documents (the shared term "filler" 29 times plus one unique term:
    /// 30 terms each) ahead of `docs`' documents. This is big enough for
    /// real-corpus-style document frequencies: on the few-document archives
    /// of the other tests EVERY term exceeds the production all-words
    /// threshold (see `AND_TERM_MAX_DF_FRAC`), so the all-words tier never
    /// fires there and tests of the tier itself need an archive that keeps
    /// the query terms' df under 1%.
    ///
    /// `docs` entries are (path, index terms, title, body); the filler
    /// documents carry no content entries (they never match a query, so
    /// their articles are never read).
    fn big_archive_test_server(
        fillers: usize,
        docs: &[(&'static str, String, &'static str, &'static str)],
    ) -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let padding = vec!["filler"; 29].join(" ");
        let mut index_docs: Vec<(String, String, String)> = (0..fillers)
            .map(|i| {
                (
                    format!("C/Filler_{i}"),
                    format!("{padding} unicum{i}"),
                    format!("Filler {i}"),
                )
            })
            .collect();
        for (path, terms, title, _) in docs {
            index_docs.push(((*path).to_string(), terms.clone(), (*title).to_string()));
        }
        let refs: Vec<(&str, &str, &str)> = index_docs
            .iter()
            .map(|(p, t, ti)| (p.as_str(), t.as_str(), ti.as_str()))
            .collect();
        let index = make_index(&refs);
        let content: Vec<TestEntry> = docs
            .iter()
            .map(|&(path, _, title, body)| TestEntry {
                namespace: b'C',
                url: path.strip_prefix("C/").unwrap_or(path),
                title,
                mime: 0,
                body: body.as_bytes(),
            })
            .collect();
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn all_words_branch_gates_terms_by_document_frequency() {
        // 300 documents: "rarea"/"rareb" in two of them (df 0.7%), "common"
        // in six (df 2%). The tier decision is parameterized by the df
        // fraction, so both sides of it are exercised here: the production
        // threshold and the forced extremes.
        let dir = tempfile::tempdir().unwrap();
        let mut wdb = WritableDatabase::create(dir.path().join("db")).unwrap();
        for i in 0..300u32 {
            let mut doc = Document::new().unwrap();
            doc.set_data(format!("C/D{i}")).unwrap();
            if i < 2 {
                doc.add_term("rarea", 1).unwrap();
                doc.add_term("rareb", 1).unwrap();
            }
            if (10..16).contains(&i) {
                doc.add_term("common", 1).unwrap();
            }
            wdb.add_document(&doc).unwrap();
        }
        wdb.commit().unwrap();
        let db = Database::open(dir.path().join("db")).unwrap();
        assert_eq!(db.doc_count(), 300);

        let terms = vec!["common".into(), "rarea".into(), "rareb".into()];
        let mut enquire = Enquire::new(&db).unwrap();
        // At the production threshold the two rare terms are specific and
        // "common" is gated out: the branch is an AND over exactly those
        // two terms, matching only the documents carrying both. (An AND
        // that included "common" would match none of them.)
        let branch = all_words_branch(&db, &terms, AND_TERM_MAX_DF_FRAC)
            .unwrap()
            .unwrap();
        enquire.set_query(&branch).unwrap();
        let mset = enquire.get_mset(0, 10, 0).unwrap();
        assert_eq!(mset.size(), 2, "{:?}", (0..mset.size()).map(|i| mset.docid(i)).collect::<Vec<_>>());

        // At df fraction 0 nothing is specific: no branch. At 1 every term
        // is specific: the branch is the old all-terms AND, which here
        // matches nothing (no document carries all three terms).
        assert!(all_words_branch(&db, &terms, 0.0).unwrap().is_none());
        let all = all_words_branch(&db, &terms, 1.0).unwrap().unwrap();
        enquire.set_query(&all).unwrap();
        assert_eq!(enquire.get_mset(0, 10, 0).unwrap().size(), 0);

        // Fewer than two specific terms, no branch (an AND over one term is
        // plain OR with the term counted twice); a single-term query never
        // gets a branch.
        let two = vec!["common".into(), "rarea".into()];
        assert!(all_words_branch(&db, &two, AND_TERM_MAX_DF_FRAC).unwrap().is_none());
        let one = vec!["rarea".into()];
        assert!(all_words_branch(&db, &one, 1.0).unwrap().is_none());
    }

    #[test]
    fn e2e_search_all_words_tier_ranks_full_matches_first() {
        // BM25 alone ranks the "Cherry" document first: thirty repetitions
        // of one term in a short document outweigh a document mentioning
        // each term once. The all-words branch must lift "Dessert Recipes"
        // (the only document with BOTH terms) above it, while the partial
        // match still appears. The archive is big (see
        // `big_archive_test_server`) so that "cherri" (2 of 404 documents)
        // and "pie" (3 of 404) stay under the production df threshold and
        // the tier actually fires - on a tiny archive every term exceeds
        // 1% df and the branch would never be built.
        let cherri_terms = "cherri ".repeat(30);
        let dessert_terms = format!("cherri pie{}", " filler".repeat(28));
        let pie_terms = format!("pie pie{}", " filler".repeat(28));
        let (server, _keep) = big_archive_test_server(
            400,
            &[
                ("C/Cherry", cherri_terms, "Cherry", CHERRY_HTML),
                ("C/Dessert_Recipes", dessert_terms, "Dessert Recipes", CHERRY_HTML),
                ("C/Pie_1", pie_terms.clone(), "Pie 1", CHERRY_HTML),
                ("C/Pie_2", pie_terms, "Pie 2", CHERRY_HTML),
            ],
        );

        let hits = search(&server, "cherry pie");
        assert_eq!(hits[0].path, "C/Dessert_Recipes", "{hits:?}");
        assert_eq!(hits[0].title, "Dessert Recipes");
        // The partial match (only "cherry") is still reported, right behind.
        assert_eq!(hits[1].path, "C/Cherry", "{hits:?}");
        // Neither hit's lead covers both terms, and "cherry" does match the
        // intro: it is reported as the matching region _intro, and the preview
        // is the intro's matching paragraph.
        assert_eq!(hits[0].sections, Some(vec!["_intro".to_string()]));
        assert!(hits[0].preview.contains("cherry is the fruit"), "{:?}", hits[0].preview);
    }

    #[test]
    fn e2e_search_common_word_does_not_block_all_words_tier() {
        // The measured pathology this change fixes, synthetic: for
        // "beaconsfield quebec attractions" the term "attract" is common
        // (10 of 411 documents, 2.4% - gated), while "beaconsfield" and
        // "quebec" are specific (3 of 411 each). The target document is
        // missing the common word entirely; the incidental documents carry
        // it once. Under the old all-terms AND the target got nothing from
        // the AND branch while the incidentals were double-counted (they
        // match AND and OR), so they outranked it; with the common word
        // gated out, the target's strong beaconsfield/quebec matches put it
        // on top and the incidentals sink below it.
        let beaconsfield_terms = format!(
            "{}{}{}",
            " filler".repeat(12),
            " beaconsfield".repeat(10),
            " quebec".repeat(8)
        );
        let amateur_terms = format!("beaconsfield quebec attract{}", " filler".repeat(27));
        let stanley_terms = format!("beaconsfield quebec attract{}", " filler".repeat(27));
        let attract_filler = format!("attract{}", " filler".repeat(29));
        let (server, _keep) = big_archive_test_server(
            400,
            &[
                ("C/Beaconsfield,_Quebec", beaconsfield_terms, "Beaconsfield, Quebec", BEACONSFIELD_QUEBEC_HTML),
                ("C/Canadian_Amateur_Championship", amateur_terms, "Canadian Amateur Championship", CANADIAN_AMATEUR_HTML),
                ("C/Frederick_Stanley", stanley_terms, "Frederick Stanley", FREDERICK_STANLEY_HTML),
                ("C/Attract_Filler_1", attract_filler.clone(), "Attract Filler 1", CANADIAN_AMATEUR_HTML),
                ("C/Attract_Filler_2", attract_filler.clone(), "Attract Filler 2", CANADIAN_AMATEUR_HTML),
                ("C/Attract_Filler_3", attract_filler.clone(), "Attract Filler 3", CANADIAN_AMATEUR_HTML),
                ("C/Attract_Filler_4", attract_filler.clone(), "Attract Filler 4", CANADIAN_AMATEUR_HTML),
                ("C/Attract_Filler_5", attract_filler.clone(), "Attract Filler 5", CANADIAN_AMATEUR_HTML),
                ("C/Attract_Filler_6", attract_filler.clone(), "Attract Filler 6", CANADIAN_AMATEUR_HTML),
                ("C/Attract_Filler_7", attract_filler.clone(), "Attract Filler 7", CANADIAN_AMATEUR_HTML),
                ("C/Attract_Filler_8", attract_filler, "Attract Filler 8", CANADIAN_AMATEUR_HTML),
            ],
        );

        let hits = search(&server, "beaconsfield quebec attractions");
        assert_eq!(hits[0].path, "C/Beaconsfield,_Quebec", "{hits:?}");
        assert_eq!(hits[0].title, "Beaconsfield, Quebec");
        // The incidental documents (which carry the gated common word) rank
        // behind the target, and the attract-only filler documents behind
        // them.
        let pos = |p: &str| {
            hits.iter()
                .position(|h| h.path == p)
                .unwrap_or_else(|| panic!("{p} missing from {hits:?}"))
        };
        let amateur = pos("C/Canadian_Amateur_Championship");
        let stanley = pos("C/Frederick_Stanley");
        assert!(amateur > 0 && stanley > 0, "{hits:?}");
        let first_filler = hits
            .iter()
            .position(|h| h.title.starts_with("Attract Filler"))
            .unwrap_or(hits.len());
        assert!(first_filler > amateur && first_filler > stanley, "{hits:?}");
    }

    #[test]
    fn e2e_search_stopword_only_query_returns_nothing() {
        let (server, _keep) = test_server();

        // A query made only of stopwords matches nothing (they are stopped
        // at index time, so the terms are absent) and must not error.
        let hits = search(&server, "the in of");
        assert!(hits.is_empty(), "{hits:?}");
    }

    #[test]
    fn e2e_search_nonsense_word_still_returns_results() {
        let (server, _keep) = test_server();

        // The indexes have no spelling data, so a nonsense word matches
        // nothing: on this tiny archive no term is specific, so there is no
        // all-words branch at all, and the parsed OR query still retrieves
        // the results for the real word.
        let hits = search(&server, "apple zzzzqq");
        assert!(!hits.is_empty(), "{hits:?}");
        assert_eq!(hits[0].path, "C/Apple", "{hits:?}");
    }

    #[test]
    fn e2e_search_interleaves_archives_and_dedupes() {
        let dir = tempfile::tempdir().unwrap();
        // Two archives; both carry an "Apple" article (same article, as in an
        // HTML and a Markdown edition of the same ZIM), plus one exclusive
        // article each.
        let index_a = make_index(&[
            ("C/Apple", "appl histori 10 000 year domest wild kazakhstan comput devic nam", "Apple"),
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
        ]);
        let index_b = make_index(&[
            ("C/Apple", "appl comput devic nam", "Apple"),
            ("C/Cherry", "cherri pie fruit tree", "Cherry"),
        ]);
        let content_a = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
        ];
        let content_b = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Cherry", title: "Cherry", mime: 0, body: CHERRY_HTML.as_bytes() },
        ];
        std::fs::write(dir.path().join("a.zim"), build_archive(&["text/html"], &content_a, &[], 0, Some(&index_a))).unwrap();
        std::fs::write(dir.path().join("b.zim"), build_archive(&["text/html"], &content_b, &[], 0, Some(&index_b))).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert_eq!(library.archives.len(), 2);
        let server = ZimMcpServer::new(library);

        // The article present in both archives is reported exactly once.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "apple" }),
        )
        .unwrap();
        let hits = block_on(ZimSearchTool::invoke(&server, params)).unwrap().results;
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].zim, "a.zim");
        assert_eq!(hits[0].path, "C/Apple");

        // Distinct matches interleave: the best match of each archive first.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "banana cherry" }),
        )
        .unwrap();
        let hits = block_on(ZimSearchTool::invoke(&server, params)).unwrap().results;
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("a.zim", "C/Banana"));
        assert_eq!((hits[1].zim.as_str(), hits[1].path.as_str()), ("b.zim", "C/Cherry"));
    }

    #[test]
    fn e2e_search_and_get_single_file_library() {
        // A library opened from one ZIM file (no folder scan) behaves like a
        // scanned folder: the archive is addressed by its file name.
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[(
            "C/Salt",
            "salt miner primari sodium chlorid himalaya deposit rock",
            "Salt",
        )]);
        let content = [TestEntry {
            namespace: b'C',
            url: "Salt",
            title: "Salt",
            mime: 0,
            body: SALT_HTML.as_bytes(),
        }];
        let file = dir.path().join("one.zim");
        std::fs::write(&file, build_archive(&["text/html"], &content, &[], 0, Some(&index))).unwrap();
        let library = Arc::new(ZimLibrary::single(&file).unwrap());
        assert_eq!(library.archives.len(), 1);
        let server = ZimMcpServer::new(library);

        let hits = search(&server, "salt");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].zim, "one.zim");
        assert_eq!(hits[0].path, "C/Salt");

        let params = serde_json::from_value::<ZimGetParams>(
            serde_json::json!({ "zim": "one.zim", "path": "C/Salt" }),
        )
        .unwrap();
        let result = block_on(ZimGetTool::invoke(&server, params)).unwrap();
        assert_eq!(result.title, "Salt");
        assert!(result.content.contains("sodium chloride"));
    }

    /// An article in the shape wikizim_parser emits (`text/markdown`).
    const ZINC_MD: &str = "\
# Zinc

*This article is about the element. For other uses, see [[Zinc (disambiguation)]].*

**Zinc** is a [[Chemical element|chemical element]] with the symbol **Zn**.

## History

Zinc smelting is documented in ancient times.

### India

Ancient India smelted zinc early.
";

    #[test]
    fn e2e_search_and_section_markdown() {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[("C/Zinc", "zinc chemic element symbol smelt ancient india", "Zinc")]);
        let content = [TestEntry {
            namespace: b'C',
            url: "Zinc",
            title: "Zinc",
            mime: 0,
            body: ZINC_MD.as_bytes(),
        }];
        let bytes = build_archive(&["text/markdown"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("md.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        let server = ZimMcpServer::new(library);

        // Search: the preview is plain text derived from the Markdown, free of
        // markup, and is the lead paragraph - the leading `# Zinc` title
        // line (a separate field of every hit) and the hatnote are dropped.
        // An exact title match never carries sections.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "zinc" }),
        )
        .unwrap();
        let hits = block_on(ZimSearchTool::invoke(&server, params)).unwrap().results;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].zim, "md.zim");
        assert_eq!(hits[0].path, "C/Zinc");
        assert!(
            hits[0]
                .preview
                .starts_with("Zinc is a chemical element with the symbol Zn."),
            "{:?}",
            hits[0].preview
        );
        assert!(
            !hits[0].preview.contains("disambiguation")
                && !hits[0].preview.contains("**")
                && !hits[0].preview.contains("[[")
                && !hits[0].preview.contains('#'),
            "{:?}",
            hits[0].preview
        );
        assert_eq!(hits[0].sections, None);

        // A query matching only a body section reports the matched
        // sections, the nested one included, and the best-matching
        // paragraph; the intro holds no match, so _intro is absent.
        let params = serde_json::from_value::<ZimSearchParams>(
            serde_json::json!({ "query": "smelting" }),
        )
        .unwrap();
        let hits = block_on(ZimSearchTool::invoke(&server, params)).unwrap().results;
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].sections,
            Some(vec!["History".to_string(), "India".to_string()]),
            "{:?}",
            hits[0]
        );
        assert_eq!(hits[0].preview, "Zinc smelting is documented in ancient times.");
        // includes the subsection, reports the heading as written.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "history" }),
        )
        .unwrap();
        let result = block_on(ZimGetSectionTool::invoke(&server, params)).unwrap();
        assert_eq!(result.section, "History");
        assert!(result.content.contains("ancient times"), "{:?}", result.content);
        assert!(result.content.contains("Ancient India smelted zinc early"));

        // The reserved intro name: the Markdown between the leading title
        // line and the first heading (hatnote and lead, raw), echoed as its
        // reserved name.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "_intro" }),
        )
        .unwrap();
        let result = block_on(ZimGetSectionTool::invoke(&server, params)).unwrap();
        assert_eq!(result.section, "_intro");
        assert!(
            result.content.contains("For other uses, see [[Zinc (disambiguation)]]."),
            "{:?}",
            result.content
        );
        assert!(result.content.contains("with the symbol **Zn**"), "{:?}", result.content);
        assert!(!result.content.starts_with('#'), "{:?}", result.content);
        assert!(!result.content.contains("History"), "{:?}", result.content);

        // Missing section: same error shape as the HTML path.
        let params = serde_json::from_value::<ZimGetSectionParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "Nope" }),
        )
        .unwrap();
        assert!(matches!(
            block_on(ZimGetSectionTool::invoke(&server, params)),
            Err(ToolError::SectionNotFound(_))
        ));
    }
}
