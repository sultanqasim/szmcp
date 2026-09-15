//! The search pipeline behind the `zim_search` tool: three simple tiers
//! (exact title/URL probe, all-words title match, full-text OR over the
//! words that are not the archive's background vocabulary), ranked against
//! the ZIM embedded Xapian indexes, merged across archives, and reported
//! with per-hit previews.

use crate::html;
use crate::markdown;
use crate::tools::ToolError;
use crate::zim::{Archive, ZimLibrary};
use schemars::JsonSchema;
use serde::Serialize;
use std::sync::Arc;
use xapian2::{
    resolve_stem_language, Enquire, Operator, Query, QueryParser, Stem, StemStrategy,
};

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
/// A fulltext query word whose document frequency exceeds this fraction of
/// the archive's documents is dropped from the FULLTEXT query (and only
/// from it - the title tier keeps every word, and hit previews score every
/// query term).
///
/// A word appearing in more than half the archive's documents is background
/// vocabulary, not a query discriminator - English "the" (59% of the md1m
/// fulltext index), "of" (63%), "and" (56%); French "de" (79% of fr.zim),
/// "la" (74%), "un" (71%). It cannot separate the wanted articles from the
/// rest, and BM25 - whose IDF already down-weights it - still floods the
/// query with its huge postlist. This is the language-agnostic replacement
/// for a stopword list: no hand-picked list works across languages, while
/// the >50% document-frequency rule is computed from the archive itself and
/// drops exactly the true background words of whatever language(s) the
/// archive is written in.
const FT_WORD_MAX_DF_FRAC: f64 = 0.5;

/// One search result.
#[derive(Serialize, JsonSchema, Debug)]
pub struct SearchHit {
    /// ZIM file name, relative to the ZIM directory
    pub zim: String,
    /// Path of the article inside the ZIM file
    pub path: String,
    /// Page/article title
    pub title: String,
    /// Preview of the article: the first sentence of the introduction when
    /// the query matches the title, otherwise the sentence with the most
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

/// Fold accents away the way the ZIM indexes were built: libzim runs every
/// indexed text through the ICU transliterator `Lower; NFD; [:M:] remove;
/// NFC` (`removeAccents` in libzim's tools.cpp) - lowercase, canonically
/// decompose, drop the combining marks, recompose - so "élections" is
/// indexed as "elections" and neither embedded index carries accented terms
/// (measured on fr.zim's full-text index: "revolu" df 11042, every accented
/// variant absent). Query words must take the same route or every accented
/// word misses the index entirely. Marks outside U+0300..=U+036F (Hebrew
/// nikkud, Arabic harakat, Indic vowel signs) are not stripped - no measured
/// archive carries them; every mark a Latin, Greek, or Cyrillic letter
/// decomposes to sits in that range.
fn fold_accents(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.to_lowercase()
        .nfd()
        .filter(|c| !('\u{300}'..='\u{36f}').contains(c))
        .nfc()
        .collect()
}

/// The query's words for the title tier: split on runs of non-alphanumeric
/// characters (whitespace AND intra-word punctuation), lowercase, one entry
/// per distinct word in first-occurrence order. The title index stores
/// titles as written - lowercased, accent-folded surface forms (measured on
/// md1m: df("beatles")=186 against df("beatl")=0) - so the tier must match
/// the words exactly as they appear. The query text arrives accent-folded
/// already (see `search`).
///
/// The split must include punctuation, not just whitespace: index terms are
/// split the same way (libzim's indexer breaks text on non-alphanumeric
/// characters), so a title "Notre-Dame de Paris" is carried as the terms
/// `notre` (df 164) and `dame` (df 191) on fr.zim's title index, never as a
/// fused `notredame`. Stripping punctuation instead of splitting fused
/// "notre-dame" into "notredame" - a term with df 0 - and the title tier's
/// AND silently matched nothing for any query with a punctuation-delimited
/// word.
fn title_words(query: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    for word in query.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let word = word.to_lowercase();
        if !words.contains(&word) {
            words.push(word);
        }
    }
    words
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

/// A stemmer that remembers the stem of every word it has seen. Natural
/// text repeats its words heavily, and every uncached stem crosses the
/// Xapian FFI; one instance serves one archive's whole search (its query
/// terms and all its hits' paragraph matchers), so repeats dominate after
/// the first hit.
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

    /// The word's stem, built the way the ZIM full-text indexes were built
    /// (accents folded - see [`fold_accents`] - then lowercased Porter2
    /// stems). Hit-preview paragraph matching feeds raw article words
    /// through here, so the fold keeps them comparable with the folded
    /// query terms (the index never saw the accents either). The fold runs
    /// only on a cache miss - the memoization dedupes it to once per
    /// distinct word. Folding EVERY occurrence instead (cache hit or not)
    /// was measured at 4.7x the search time on fr.zim ("cathédrale
    /// notre-dame de paris" 9.7s against 2.0s): two Unicode normalization
    /// passes over every word of every scanned paragraph are not free.
    fn stem(&mut self, word: &str) -> &str {
        if !self.cache.contains_key(word) {
            let folded = fold_accents(word);
            let stemmed = self.stem.apply(&folded).unwrap_or(folded);
            self.cache.insert(word.to_string(), stemmed);
        }
        self.cache[word].as_str()
    }

    /// [`Stemmer::stem`] for the query path: those words arrive
    /// accent-folded once in `search` (see [`fold_accents`]), so the fold
    /// would be an identity pass and is skipped. Same cache, keyed on the
    /// word as received - a raw article word that folds to the same form
    /// stems to the same thing.
    fn stem_folded(&mut self, folded: &str) -> &str {
        if !self.cache.contains_key(folded) {
            let stemmed = self
                .stem
                .apply(folded)
                .unwrap_or_else(|_| folded.to_string());
            self.cache.insert(folded.to_string(), stemmed);
        }
        self.cache[folded].as_str()
    }
}

/// One archive's view of a query (see [`search`]): the archive's stemmer,
/// the query's words stemmed with it, and the resolved stemmer language.
/// All of it depends on the archive's language - libzim builds each
/// archive's embedded index with the stemmer chosen from the archive's
/// `Language` metadata, so querying a French archive with an English
/// stemmer finds nothing (and vice versa). The fulltext query is built per
/// archive too ([`fulltext_query`], inside the search's pool closure): the
/// words it keeps are decided by the archive's own document frequencies.
struct ArchiveQuery {
    stemmer: Stemmer,
    /// The archive's resolved stemmer language (see `build`): the stemmer
    /// was created with it, and so is the fulltext query's parser.
    language: String,
    /// Folded stems of the query's words (deduped, first-occurrence
    /// order): the hit previews' paragraph matching, scored against the
    /// archive's full-text vocabulary.
    terms: Vec<String>,
    /// The query's folded surface words (see [`title_words`]): the title
    /// tier's AND query, one term per word.
    title_words: Vec<String>,
    /// The folded query's whitespace tokens, as written (see `search`):
    /// the fulltext query is parsed from the KEPT ones (see
    /// [`FT_WORD_MAX_DF_FRAC`], [`fulltext_query`]). Tokens, not the title
    /// tier's alphanumeric-only words, because the QueryParser tokenizes
    /// intra-word punctuation itself - "notre-dame" must reach it whole,
    /// or the hyphen fusion ("notredame") builds a term the index never
    /// carries and the Cathédrale Notre-Dame article drops out of its own
    /// query (measured).
    ft_words: Vec<String>,
    /// The stems of `ft_words`' alphanumeric-only forms, same order: the
    /// df filter's lookup keys (the index's terms are folded Porter2
    /// stems of such forms).
    ft_stems: Vec<String>,
}

impl ArchiveQuery {
    fn build(arc: &Archive, query: &str) -> Result<Self, ToolError> {
        // The archive's own stemmer language: its Language metadata, with
        // the metadata code mapped to a language Xapian accepts (ISO-639-3
        // "fra" -> "fr"; unknown -> no stemming). Archives without the
        // metadata keep English stems - that is what their indexes use.
        let language = arc.language().unwrap_or_else(|| "eng".to_string());
        let language = resolve_stem_language(&language);
        let mut stemmer = Stemmer::new(&language).map_err(|e| {
            ToolError::Internal(format!("failed to create {language} stemmer: {}", e.msg()))
        })?;
        // `query` arrives accent-folded (see `search`), so the tier words
        // stem through the no-second-fold path (`stem_folded`).
        let title_words = title_words(query);
        let mut terms: Vec<String> = Vec::new();
        for word in &title_words {
            let stemmed = stemmer.stem_folded(word).to_string();
            if !terms.contains(&stemmed) {
                terms.push(stemmed);
            }
        }
        // The fulltext tier's filterable words: the folded query's tokens
        // as written, each paired with the stem of its alphanumeric-only
        // form - the df filter's lookup key (see `fulltext_query`). The
        // bare form is deliberately FUSED (non-alphanumeric characters
        // stripped, not split): it is not meant to be a term of the index
        // but a conservative key. A fused hyphenated compound like
        // "notre-dame" -> "notredame" is a word no document text contains
        // (the indexer splits on non-alphanumeric characters, exactly like
        // `title_words`), so its df measures 0 and the filter always keeps
        // the token - hyphenated compounds are discriminative, and the
        // QueryParser splits the token as written anyway. The pathological
        // shape is a fused form colliding with a common standalone word
        // ("u.s." -> "us", the df of the ordinary word "us"): such a token
        // can be dropped from the FULLTEXT query though its parts are rare
        // - accepted, as the loss is one OR term (the title tier still
        // carries every split word, and BM25 ranks whatever remains). A
        // punctuation-only token ("-", "!" ... after the fold) has no
        // alphanumeric form; its empty stem matches nothing, df 0, and the
        // token is kept - the parser sees it, as it did before the filter
        // existed.
        let mut ft_words: Vec<String> = Vec::new();
        let mut ft_stems: Vec<String> = Vec::new();
        for token in query.split_whitespace() {
            let bare: String = token
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            let stem = if bare.is_empty() {
                String::new()
            } else {
                stemmer.stem_folded(&bare).to_string()
            };
            ft_words.push(token.to_string());
            ft_stems.push(stem);
        }
        Ok(Self { stemmer, language, terms, title_words, ft_words, ft_stems })
    }
}

/// The fulltext query of one archive: `words` (the folded query's
/// whitespace tokens, as written) minus background vocabulary, joined by
/// single spaces and parsed by the QueryParser exactly as the whole query
/// was parsed before the filter existed (the archive-language stemmer,
/// `STEM_ALL`, default op OR).
///
/// A word whose document frequency in THIS archive's fulltext index is more
/// than [`FT_WORD_MAX_DF_FRAC`] of the archive's document count is dropped:
/// a word matching half the archive is background vocabulary ("the", French
/// "de"), not a query discriminator. The dfs come off the archive's own
/// fulltext handle (`Database::doc_count` + `Database::termfreq` on the
/// STEMS in `stems` - the index's terms are folded Porter2 stems), so the
/// kept set differs per archive: the query is per archive in a second way
/// (the stemmer language was the first). That is also why the parse happens
/// here, inside `search`'s pool closure, instead of in
/// `ArchiveQuery::build` - the handle exists only there.
///
/// Returns `None` when EVERY word is dropped (the query "the" against an
/// archive where "the" is 59% of the documents): the caller skips the
/// fulltext band, and the exact/title tiers serve the results.
///
/// What is lost: the raw query's boolean syntax. The kept words are
/// lowercased query words, so AND/OR/NOT/parentheses reach the parser as
/// ordinary words - and the ultra-common operator words are usually dropped
/// by the 50% rule anyway. (Uppercase operators had already lost their case
/// to the accent fold - Xapian's operator keywords are case-sensitive, so
/// "AND" has been an ordinary word on this tier since c8fb6d4; the filter
/// only adds the drop.) Quotes were already stripped (see `search`).
fn fulltext_query(
    words: &[String],
    stems: &[String],
    language: &str,
    fulltext: &xapian2::Database,
) -> Result<Option<Query>, ToolError> {
    let doc_count = fulltext.doc_count();
    let kept: Vec<&str> = words
        .iter()
        .zip(stems)
        // Not MORE than half the archive: kept. Exactly half can still
        // discriminate, barely.
        .filter(|(_, stem)| {
            fulltext.termfreq(stem) as f64 <= FT_WORD_MAX_DF_FRAC * doc_count as f64
        })
        .map(|(word, _)| word.as_str())
        .collect();
    if kept.is_empty() {
        return Ok(None);
    }
    // openZIM's full-text indexes contain unprefixed, accent-folded
    // Porter2 stems (libzim indexes STEM_ALL over removeAccents'd text),
    // so queries must be stemmed the same way - Xapian's default strategy
    // would turn lowercase terms into "Z"-prefixed stem terms that never
    // match. The words arrive accent-folded (see `search`), so the
    // parser's terms land in the index's folded vocabulary too. Default
    // combining op is OR.
    let mut qp = QueryParser::new()?;
    qp.set_stemmer(language)?;
    qp.set_stemming_strategy(StemStrategy::All)?;
    qp.set_default_op(Operator::Or)?;
    let xquery = qp
        .parse_query(&kept.join(" "))
        .map_err(|e| ToolError::InvalidArgument(format!("failed to parse query: {e}")))?;
    Ok(Some(xquery))
}

/// A display title derived from an article path: the namespace prefix
/// stripped, underscores as spaces ("C/Citric_acid_cycle" -> "Citric acid
/// cycle"). The last fallback for archives whose directory entries and
/// index documents carry no title.
fn path_title(path: &str) -> String {
    path.split_once('/')
        .map(|(_, url)| url)
        .unwrap_or(path)
        .replace('_', " ")
}

/// Search all articles in all ZIM files of the library - the pipeline behind
/// the `zim_search` tool: ranked hits, best first.
pub fn search(library: &ZimLibrary, query: &str) -> Result<SearchResults, ToolError> {
    if query.trim().is_empty() {
        return Err(ToolError::InvalidArgument("query must not be empty".into()));
    }

    // Quoted phrases build OP_PHRASE subqueries, but the indexes carry no
    // positional data (libzim indexes `index_text_without_positions`), so
    // a phrase matches nothing and silently loses the whole full-text
    // band: strip the quotes and parse the words (BM25 OR over them).
    // The accent fold (see `fold_accents`) is computed ONCE, over the
    // quote-stripped query, and the SAME folded string serves every tier:
    // `ArchiveQuery::build` tokenizes both tiers' words from it, so
    // accented words reach Xapian's stemmer in the index's folded form
    // ("révolution" -> "revolution"). The full-text tier parses the KEPT
    // words per archive (see `fulltext_query`) - background vocabulary is
    // an archive-local judgment, made on the archive's own document
    // frequencies inside the pool closure below. The exact title/URL probe
    // still gets the raw query - article paths and directory titles carry
    // their accents ("C/Université").
    let folded_query = fold_accents(&query.replace('"', " "));

    // Which tier produced a hit. Exact and title-tier hits are title
    // matches: their preview is the first intro sentence and `sections` is
    // omitted.
    #[derive(Clone, Copy, PartialEq)]
    enum HitKind {
        /// Exact title/URL probe hit (the ZIM directory itself).
        Exact,
        /// Hit from the archive's title index (`X/title/xapian`).
        Title,
        /// Hit from the archive's full-text index.
        Fulltext,
    }

    // One query view per archive, in `library.archives` order: libzim
    // stems each archive's embedded index with the stemmer chosen from
    // that archive's Language metadata, so the parsed query, the stemmed
    // terms, and the stemmer used for hit previews are per archive (one
    // English stemmer used to find nothing on a Language=fra archive).
    // The terms drive the previews' paragraph matching, the surface words
    // the title tier, and the parsed query the full-text tier.
    let mut queries: Vec<ArchiveQuery> = library
        .archives
        .iter()
        .map(|arc| ArchiveQuery::build(arc, &folded_query))
        .collect::<Result<_, ToolError>>()?;

    // Tier 1 - exact title/URL matches, found in the ZIM directory itself:
    // redirects are not in the search indexes, and a query that names an
    // article exactly must rank first no matter what BM25 produces. One
    // probe per archive; a failed probe simply contributes nothing. The
    // entry is resolved to its terminal article below (a redirect match
    // reports the article it names, not itself).
    let mut merged: Vec<(&Arc<Archive>, String, String, HitKind)> = Vec::new();
    for arc in &library.archives {
        if let Some((path, _)) = arc.lookup_exact(query)? {
            merged.push((arc, path, String::new(), HitKind::Exact));
        }
    }

    // Tiers 2 and 3, per archive. Each archive's Xapian handles are
    // checked out of that archive's pool for the duration of the search:
    // concurrent searches never share a handle (Xapian does not support
    // concurrent calls on one database object).
    //
    // Tier 2 - the title tier queries the archive's title index
    // (`X/title/xapian`; documents ARE titles) with an AND over ALL the
    // query words' surface forms (see `title_words`): a title that says
    // the whole query is far stronger evidence than body words, and an
    // article matching only part of the query in its title is not
    // promoted at all - that judgment is left to the full-text tier's
    // BM25, whose IDF already down-weights common words. Single-word
    // queries are the one-word AND. The band is skipped for archives
    // without a title index and queries with no usable words.
    //
    // Tier 3 - the full-text tier is the parsed query over the KEPT words:
    // plain BM25 OR, after words matching more than half the archive's
    // documents were dropped as background vocabulary (`FT_WORD_MAX_DF_FRAC`,
    // `fulltext_query` - the language-agnostic replacement for a stopword
    // list). BM25's IDF still down-weights the merely common words, and no
    // all-words AND branch is layered on top (both deleted in e98568c - the
    // hand-rolled AND double-counted every document that matched it).
    let mut title_lists: Vec<(&Arc<Archive>, Vec<(String, String)>)> = Vec::new();
    // Full-text tier: (weight, path, title from the index).
    let mut per_archive: Vec<(&Arc<Archive>, Vec<(f64, String, String)>)> = Vec::new();
    for (arc, query_state) in library.archives.iter().zip(queries.iter_mut()) {
        let Some((title_list, list)) = arc.with_xapian(|h| -> Result<_, ToolError> {
            let mut title_list = Vec::new();
            if !query_state.title_words.is_empty() {
                if let Some(title_db) = &h.title {
                    let and_query = combine_terms(Operator::And, &query_state.title_words)?;
                    let mut enquire = Enquire::new(title_db)?;
                    enquire.set_sort_by_relevance();
                    enquire.set_query(&and_query)?;
                    let mset = enquire.get_mset(0, SEARCH_LIMIT, 0)?;
                    for j in 0..mset.size() {
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
                        title_list.push((path, title));
                    }
                }
            }

            // The full-text tier: plain BM25 over the parsed query, built
            // HERE inside the pool closure because the background-vocabulary
            // filter reads the archive's own fulltext index (document
            // frequencies - see `fulltext_query`). `None` - every query word
            // is background vocabulary on THIS archive ("the" at 59% of
            // md1m) - skips the band: the exact/title tiers still serve
            // results.
            let Some(xquery) =
                fulltext_query(&query_state.ft_words, &query_state.ft_stems, &query_state.language, &h.fulltext)?
            else {
                return Ok((title_list, Vec::new()));
            };
            let mut enquire = Enquire::new(&h.fulltext)?;
            enquire.set_query(&xquery)?;
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
    // comparable across archives, so each tier merges its archives' ranked
    // lists by rotation instead of by weight: every archive contributes
    // its best match before any archive contributes its second best. The
    // tiers keep their order: exact title/URL probe hits (already in
    // `merged`), then title-tier matches, then full-text matches.
    let mut rank = 0usize;
    loop {
        let mut picked = false;
        for (arc, list) in &title_lists {
            if let Some((path, title)) = list.get(rank) {
                merged.push((arc, path.clone(), title.clone(), HitKind::Title));
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

    // Resolve every candidate to the terminal article of its entry before
    // dedupe, then dedupe on that identity: the same article must never
    // appear twice, no matter how many tiers and redirect spellings reach
    // it. Title-index redirect documents (old-namespace archives) carry
    // their OWN title in value slot 0 while pointing at the redirect
    // entry - keying on the index title reported the same article under
    // several titles (the measured French "Tour Eiffel" duplicates), so
    // the identity here is the terminal entry's path (within the archive)
    // plus its resolved title for the cross-archive dedupe (the same
    // article published twice - an HTML and a Markdown edition - carries
    // the same title but different paths). The reported title is the
    // terminal entry's directory title when one exists, else the index
    // title, else derived from the terminal path ("C/Citric_acid_cycle" ->
    // "Citric acid cycle"). Resolution failures (a degenerate index doc)
    // keep the raw path, like the preview below.
    let mut seen_titles = std::collections::HashSet::new();
    let mut seen_paths = std::collections::HashSet::new();
    merged = merged
        .into_iter()
        .filter_map(|(arc, path, idx_title, kind)| {
            let (path, entry_title) =
                arc.resolve_terminal(&path).ok().flatten().unwrap_or((path, String::new()));
            let title = if !entry_title.is_empty() {
                entry_title
            } else if !idx_title.is_empty() {
                idx_title
            } else {
                path_title(&path)
            };
            let title_key = if title.is_empty() { path.clone() } else { title.clone() };
            if !seen_titles.insert(html::normalize(&title_key)) {
                return None;
            }
            // Within one archive the terminal path is the article's
            // identity; two index documents can still name it under
            // different titles.
            if !seen_paths.insert((Arc::as_ptr(arc) as usize, path.clone())) {
                return None;
            }
            Some((arc, path, title, kind))
        })
        .collect();
    merged.truncate(SEARCH_LIMIT as usize);

    let mut hits = Vec::with_capacity(merged.len());
    for (arc, path, title, kind) in &merged {
        // Exact and title-tier hits are title matches: the first intro
        // sentence is the right preview and there is nothing to point at
        // section-wise.
        let title_match = *kind != HitKind::Fulltext;
        let (mime, bytes) = match arc.article_preview(path, HIT_READ_BYTES) {
            Ok(Some((_, mime, bytes))) => (mime, bytes),
            _ => (None, Vec::new()),
        };
        let article = String::from_utf8_lossy(&bytes);
        // Markdown editions carry plain Markdown, not HTML: pick the matching
        // splitter so the paragraphs and section names are free of markup.
        let is_markdown = mime.as_deref().is_some_and(|m| m.contains("markdown"));
        // Paragraph matching uses the hit's archive stemmer and terms (the
        // archive is always from the library, so the lookup cannot fail).
        let qi = library
            .archives
            .iter()
            .position(|a| Arc::ptr_eq(a, arc))
            .expect("hit archive is from the library");
        let query_state = &mut queries[qi];
        let (preview, sections) = hit_preview(
            &article,
            &query_state.terms,
            title_match,
            &mut query_state.stemmer,
            is_markdown,
        );
        hits.push(SearchHit {
            zim: arc.name.clone(),
            path: path.clone(),
            title: title.clone(),
            preview,
            sections,
        });
    }
    Ok(SearchResults { results: hits })
}

/// How many of `text`'s word occurrences are query terms - the paragraph's
/// match count, stemmed the same way the index and the query are. Every
/// query term counts: BM25 scored them all too.
fn para_matches(text: &str, terms: &[String], stem: &mut Stemmer) -> usize {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .filter(|w| terms.iter().any(|t| t == stem.stem(w)))
        .count()
}

/// Whether every query term occurs (stemmed) somewhere in `text`'s words.
fn covers_all_terms(text: &str, terms: &[String], stem: &mut Stemmer) -> bool {
    let mut covered = vec![false; terms.len()];
    let mut left = terms.len();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
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
///   the hit came from the title tier) gets the lead paragraph's FIRST
///   sentence as its preview and no sections - the title already said
///   everything;
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
    if title_match {
        // The first sentence of the lead: the title said what the article
        // is; one sentence of it is enough of a preview (and stays far
        // from the body sections the full-text hits point at).
        let first = intro
            .first()
            .and_then(|p| {
                sentences(p)
                    .first()
                    .map(|s| s.chars().take(INTRO_CHARS).collect::<String>())
            })
            .unwrap_or_default();
        return (first, None);
    }
    if intro.first().is_some_and(|p| covers_all_terms(p, terms, stem)) {
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
    use crate::zim::testutil::{
        build_archive, build_archive_indexes, language_metadata_entry, TestEntry, TestRedirect,
    };
    use rmcp::handler::server::router::tool::AsyncTool;
    use std::future::Future;
    use xapian2::{Document, WritableDatabase};

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

    /// For the title-tier tests: an article whose title contains all the
    /// query's words ("Nitrogen Gas Effects" for "effects of nitrogen gas")
    /// with the matching words also in its body (so its full-text document
    /// matches too and the cross-tier dedupe has something to do), and an
    /// article that only matches the query in its body.
    const NITROGEN_GAS_EFFECTS_HTML: &str = "<html><body><h1>Effects of Nitrogen Gas</h1>\
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
    /// stems, exactly as libzim indexes the FULL-TEXT index with STEM_ALL
    /// ("appl" is the stem of "apple", "comput" of "computing"). The terms
    /// are passed in already stemmed (the English stem shapes the existing
    /// tests use). Real TITLE indexes store surface word forms instead -
    /// every title fixture below passes surfaces there, as the band
    /// matches surface forms.
    fn make_index(docs: &[(&str, &str, &str)]) -> Vec<u8> {
        make_index_stemmed(None, docs)
    }

    /// [`make_index`], parameterized by the index language: `Some(code)`
    /// folds and stems the given SURFACE words with that language's stemmer
    /// (resolved exactly like the query side resolves an archive's Language
    /// metadata, accents folded away first), the way libzim indexes an
    /// archive's content with its Language metadata (removeAccents, then
    /// the stemmer). `None` adds the terms verbatim.
    fn make_index_stemmed(language: Option<&str>, docs: &[(&str, &str, &str)]) -> Vec<u8> {
        let mut stem = language.map(|code| {
            Stem::new(&resolve_stem_language(code)).expect("resolved language must be stemmable")
        });
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
                    let term = match &mut stem {
                        // libzim folds before stemming (removeAccents).
                        Some(stem) => stem.apply(&fold_accents(t)).unwrap(),
                        None => t.to_string(),
                    };
                    doc.add_term(&term, 1).unwrap();
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
    fn stem_language_resolution() {
        // Xapian rejects ISO-639-3 codes (Stem::new("fra") throws
        // "Language code fra unknown") and only knows English names plus
        // two-letter ISO-639-1 codes: the fallback maps a 639-3 metadata
        // code to its two-letter prefix ("fra" -> French stemming).
        assert_eq!(resolve_stem_language("fra"), "fr");
        assert_eq!(resolve_stem_language("eng"), "en");
        // Full names and 639-1 codes pass through unchanged.
        assert_eq!(resolve_stem_language("french"), "french");
        assert_eq!(resolve_stem_language("fr"), "fr");
        assert_eq!(resolve_stem_language("en"), "en");
        // Languages Xapian cannot stem (Chinese) and garbage fall back to
        // no stemming - never an error, never a made-up stemmer.
        assert_eq!(resolve_stem_language("zho"), "none");
        assert_eq!(resolve_stem_language("zh"), "none");
        assert_eq!(resolve_stem_language(""), "none");
    }

    #[test]
    fn fold_accents_mirrors_the_index_folding() {
        // libzim's "Lower; NFD; [:M:] remove; NFC" transliterator.
        assert_eq!(fold_accents("Révolution française"), "revolution francaise");
        assert_eq!(fold_accents("ÉLECTIONS"), "elections");
        // No canonical decomposition: ß, ø, and the Œ ligature keep their
        // shape (NFD leaves them whole, so the index does too).
        assert_eq!(fold_accents("Straße"), "straße");
        assert_eq!(fold_accents("Ørestad Øl"), "ørestad øl");
        // ASCII passes through untouched (the EN pipeline is a no-op).
        assert_eq!(fold_accents("Black Holes!"), "black holes!");
    }

    #[test]
    fn resolved_stemmer_stems_like_the_index_language() {
        // The "fra" fallback really stems FRENCH: inflected forms share a
        // stem - the FOLDED stem the index carries ("elections" -> "elect";
        // the fold in `Stemmer::stem` happens before stemming, like libzim's
        // removeAccents before its TermGenerator).
        let mut fr = Stemmer::new(&resolve_stem_language("fra")).unwrap();
        let plural = fr.stem("élections").to_string();
        assert_eq!(plural, fr.stem("élection"));
        assert_eq!(plural, "elect");
        // The already-folded query path (see `search`): no fold pass, the
        // same stems as the raw-word path (whose fold runs once per
        // distinct word, on a cache miss only).
        assert_eq!(fr.stem_folded("elections"), "elect");
        assert_eq!(
            fr.stem("Élections").to_string(),
            fr.stem_folded("elections").to_string()
        );
        // It still stems deeper than the English stemmer where the
        // languages genuinely diverge ("chevaux": French "cheval",
        // English leaves the word whole).
        assert_ne!(
            fr.stem("chevaux"),
            Stemmer::new(&resolve_stem_language("eng")).unwrap().stem("chevaux")
        );
        // "zho" resolves to no stemming: words come back accent-folded and
        // lowercased but unstemed (a Chinese index and Chinese queries then
        // agree word for word).
        let mut zh = Stemmer::new(&resolve_stem_language("zho")).unwrap();
        assert_eq!(zh.stem("的"), "的");
        assert_eq!(zh.stem("Élections"), "elections");
    }

    /// A Language=fra archive whose index was built with French stems
    /// (make_index_stemmed mirrors libzim's indexer): the inflected query
    /// form "élections" matches only after FRENCH stemming ("élect") - the
    /// English stemmer leaves "élection" whole, which is why French
    /// inflected forms used to miss on this archive.
    #[test]
    fn e2e_search_stems_queries_with_the_archive_language() {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index_stemmed(
            Some("fra"),
            &[
                (
                    "C/Élection",
                    "une élection est un scrutin les élections présidentielles ont lieu \
                     tous les cinq ans",
                    "Élection",
                ),
                (
                    "C/Géographie",
                    "la géographie étudie les paysages et les reliefs de la terre",
                    "Géographie",
                ),
                (
                    "C/École",
                    "une école primaire accueille les enfants du village",
                    "École",
                ),
            ],
        );
        let election_html: &'static [u8] = "<html><body><h1>Élection</h1>\
            <p>Une élection est un scrutin. Les élections présidentielles \
            ont lieu tous les cinq ans.</p></body></html>".as_bytes();
        let geo_html: &'static [u8] = "<html><body><h1>Géographie</h1>\
            <p>La géographie étudie les paysages.</p></body></html>".as_bytes();
        let school_html: &'static [u8] = "<html><body><h1>École</h1>\
            <p>Une école primaire accueille les enfants du village.</p></body></html>".as_bytes();
        let content = [
            TestEntry { namespace: b'C', url: "Élection", title: "Élection", mime: 0, body: election_html },
            TestEntry { namespace: b'C', url: "Géographie", title: "Géographie", mime: 0, body: geo_html },
            TestEntry { namespace: b'C', url: "École", title: "École", mime: 0, body: school_html },
            language_metadata_entry("fra"),
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("fr.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert_eq!(library.archives[0].language().as_deref(), Some("fra"));
        let server = ZimMcpServer::new(library);

        // The inflected plural form hits - only French stemming gets there.
        let hits = search(&server, "élections");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "C/Élection");
        assert!(!hits[0].preview.is_empty());
        // The singular form shares the French stem ("élect").
        let hits = search(&server, "élection");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "C/Élection");
        // The other article matches its own (accented) words...
        let hits = search(&server, "paysages");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "C/Géographie");
        // ...and not the election words (no cross-language stem collisions).
        let hits = search(&server, "scrutin");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "C/Élection");
        // An accented inflected form finds the school article through the
        // FOLD: "écoles" folds to "ecoles" and stems to "ecol", the index
        // term of "école" (the fold runs before the stemmer, like libzim's
        // removeAccents). Unfolded, the query stems to "écol" and the
        // folded index term matches nothing.
        let hits = search(&server, "écoles");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/École");
        assert_eq!(hits[0].title, "École");
        // Nothing matches a term absent from the index.
        let hits = search(&server, "zzzzz");
        assert!(hits.is_empty());
    }

    /// An e2e proof that the FULLTEXT tier's parsed query is built from the
    /// accent-FOLDED words (see `search`): the synthetic index is built the
    /// way libzim builds a full-text index (`make_index_stemmed` folds
    /// before stemming), so it carries ONLY folded French stems - the shape
    /// measured on the extracted fr.zim full-text index ("revolu" df 11042,
    /// "francais" 26817; the accented variants "révolution"/"français" df 0).
    /// An unfolded parse would stem "révolution" to a term the index does
    /// not carry, and this query would match nothing.
    #[test]
    fn e2e_search_fulltext_query_folds_accents() {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index_stemmed(
            Some("fra"),
            &[
                (
                    "C/Révolution",
                    "la révolution française éclate en 1789 la monarchie est \
                     renversée la république proclamée",
                    "Révolution",
                ),
                // Fillers: a one-document index makes every term 100%-df,
                // which the fulltext query's background-vocabulary filter
                // (FT_WORD_MAX_DF_FRAC) drops - this query would match
                // nothing. Three documents keep the query words at 33% and
                // the folded-parse proof below unchanged.
                (
                    "C/Géographie",
                    "la géographie étudie les paysages et les reliefs de la terre",
                    "Géographie",
                ),
                (
                    "C/École",
                    "une école primaire accueille les enfants du village",
                    "École",
                ),
            ],
        );
        let html: &'static [u8] = "<html><body><h1>Révolution</h1>\
            <p>La Révolution française éclate en 1789. La monarchie est \
            renversée et la république proclamée.</p></body></html>".as_bytes();
        let content = [
            TestEntry { namespace: b'C', url: "Révolution", title: "Révolution", mime: 0, body: html },
            TestEntry { namespace: b'C', url: "Géographie", title: "Géographie", mime: 0, body: "<html><body><h1>Géographie</h1><p>La géographie étudie les paysages.</p></body></html>".as_bytes() },
            TestEntry { namespace: b'C', url: "École", title: "École", mime: 0, body: "<html><body><h1>École</h1><p>Une école primaire accueille les enfants.</p></body></html>".as_bytes() },
            language_metadata_entry("fra"),
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("fr.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        let server = ZimMcpServer::new(library);

        // Not an exact title/URL hit ("révolution française" is not the
        // article's title/URL) and the archive has no title index, so the
        // ONLY route to the article is the full-text tier - which matches
        // only because the parsed query's words were folded first
        // ("révolution" -> "revolution" -> the stem "revolu"; the index
        // carries no "révolution"/"révolut" term, as on the real fr.zim).
        let hits = search(&server, "révolution française");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Révolution");
        // A full-text hit whose lead paragraph covers every query term:
        // the whole two-sentence lead is the preview (a title-match
        // preview would be the first sentence only).
        assert_eq!(
            hits[0].preview,
            "La Révolution française éclate en 1789. La monarchie est renversée et la république proclamée."
        );
        assert_eq!(hits[0].sections, None);
    }

    /// The fulltext tier's background-vocabulary filter ([`fulltext_query`],
    /// [`FT_WORD_MAX_DF_FRAC`]), white-box: a word in MORE than half the
    /// archive's documents is dropped from the parsed fulltext query, a word
    /// in at most half is kept, and a query of ONLY background words yields
    /// `None` (the caller skips the band; exact/title still serve results).
    /// Fixture via the raw `WritableDatabase` (the same shape `make_index`
    /// builds): 10 documents, "commonword" in 6 (60%), "halfword" in 5
    /// (exactly 50%), "rarea"/"rareb" in one each, "middling" in 2. The
    /// terms are the STEMS - what the filter looks up (the index's terms are
    /// folded Porter2 stems) and what the STEM_ALL parser produces.
    #[test]
    fn fulltext_query_drops_words_over_half_the_archive() {
        let mut stemmer = Stemmer::new(&resolve_stem_language("en")).unwrap();
        let s_common = stemmer.stem_folded("commonword").to_string();
        let s_half = stemmer.stem_folded("halfword").to_string();
        let s_rarea = stemmer.stem_folded("rarea").to_string();
        let s_rareb = stemmer.stem_folded("rareb").to_string();
        let s_middling = stemmer.stem_folded("middling").to_string();
        let dir = tempfile::tempdir().unwrap();
        let db_dir = dir.path().join("db");
        {
            let mut wdb = WritableDatabase::create(&db_dir).unwrap();
            for i in 0..10 {
                let mut doc = Document::new().unwrap();
                doc.set_data(format!("C/D{i}")).unwrap();
                if i < 6 {
                    doc.add_term(&s_common, 1).unwrap();
                }
                if i < 5 {
                    doc.add_term(&s_half, 1).unwrap();
                }
                if i == 6 {
                    doc.add_term(&s_rarea, 1).unwrap();
                }
                if i == 7 {
                    doc.add_term(&s_rareb, 1).unwrap();
                }
                if i >= 8 {
                    doc.add_term(&s_middling, 1).unwrap();
                }
                wdb.add_document(&doc).unwrap();
            }
            wdb.commit().unwrap();
        }
        let db = xapian2::Database::open(&db_dir).unwrap();
        assert_eq!(db.doc_count(), 10);

        // Run a built query and return the matched documents' paths, sorted.
        let matches = |q: &Query| -> Vec<String> {
            let mut enquire = Enquire::new(&db).unwrap();
            enquire.set_query(q).unwrap();
            enquire.set_sort_by_relevance();
            let mset = enquire.get_mset(0, 10, 0).unwrap();
            let mut out = Vec::new();
            for j in 0..mset.size() {
                out.push(mset.document(j).unwrap().data_str().unwrap());
            }
            out.sort();
            out
        };
        let words = |ws: &[&str]| ws.iter().map(|w| w.to_string()).collect::<Vec<String>>();
        let query = |ws: &[&str], stems: &[String]| fulltext_query(&words(ws), stems, "en", &db);

        // 6 of 10 documents (60% > 50%): "commonword" is dropped, "rarea"
        // (10%) kept - the parsed query matches ONLY the rarea document
        // (the unfiltered OR would match 7).
        let q = query(&["commonword", "rarea"], &[s_common.clone(), s_rarea.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(matches(&q), vec!["C/D6"]);

        // 5 of 10 is exactly half, not MORE than half: kept - the parsed
        // query matches its 5 documents.
        let q = query(&["halfword"], &[s_half.clone()]).unwrap().unwrap();
        assert_eq!(matches(&q).len(), 5);

        // 2 of 10 (20%): kept, next to the dropped common word.
        let q = query(&["commonword", "middling"], &[s_common.clone(), s_middling.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(matches(&q), vec!["C/D8", "C/D9"]);

        // Both rare words kept: the OR matches both.
        let q = query(&["rarea", "rareb"], &[s_rarea.clone(), s_rareb.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(matches(&q), vec!["C/D6", "C/D7"]);

        // EVERY word is background vocabulary: no fulltext query at all.
        let q = query(&["commonword"], &[s_common.clone()]).unwrap();
        assert!(q.is_none());
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
        // An exact title match reports the lead's first sentence as its
        // preview (the lead is one sentence here).
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
    /// title's SURFACE words (lowercased, as written) and whose value slot
    /// 0 is the title (the document data is the article path in both
    /// indexes).
    fn title_index_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Nitrogen_Gas_Effects", "nitrogen gas surround effect unavoid", "Nitrogen Gas Effects"),
            ("C/Weather", "weather forecast describ effect air pressur gas law explan atmospher", "Weather"),
            // Fillers: on the two-document index above, "effect" and "gas"
            // (shared by both documents) are 100%-df background words, which
            // the fulltext query's filter (FT_WORD_MAX_DF_FRAC) drops - and
            // the "Weather" fulltext hit below would vanish. Four documents
            // hold the shared words at exactly 50%: kept (the rule drops
            // only MORE than half the archive).
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
            ("C/Glacier", "glacier ice dens movem weight flow", "Glacier"),
        ]);
        let titles = make_index(&[
            ("C/Nitrogen_Gas_Effects", "effects of nitrogen gas", "Effects of Nitrogen Gas"),
            ("C/Weather", "weather", "Weather"),
            // Real title indexes carry function words unstopped (measured on
            // md1m: df("of")=204037, df("on")=7658) - one such doc keeps the
            // AND below realistic against a query containing "of".
            ("C/History_Of_Salt", "history of salt", "History of Salt"),
        ]);
        let content = [
            // Empty directory-entry title, as in modern openZIM archives:
            // the title lives in the indexes (value slot 0) only.
            TestEntry { namespace: b'C', url: "Nitrogen_Gas_Effects", title: "", mime: 0, body: NITROGEN_GAS_EFFECTS_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Weather", title: "Weather", mime: 0, body: WEATHER_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "History_Of_Salt", title: "", mime: 0, body: b"<html><body><h1>History of Salt</h1><p>Salt has been traded for centuries.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Glacier", title: "Glacier", mime: 0, body: b"<html><body><h1>Glacier</h1><p>A glacier is a body of dense ice.</p></body></html>" },
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

        // "effects of nitrogen gas" is nobody's title or URL (the article
        // sits on C/Nitrogen_Gas_Effects), but the article's title contains
        // every query word - function word included, like real unstopped
        // title indexes ("Effects of climate change on agriculture" for the
        // measured md1m query): its title-index document IS the title tier
        // and ranks ahead of the full-text-only matches (whose titles carry
        // no query word); the hit is styled as a title match.
        let hits = search(&server, "effects of nitrogen gas");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].path, "C/Nitrogen_Gas_Effects", "{hits:?}");
        assert_eq!(hits[0].title, "Effects of Nitrogen Gas");
        // Title-match semantics: the lead's FIRST sentence and no sections -
        // NOT the Everywhere section (the "effects ..." paragraph with more
        // query matches) the full-text tier would have reported.
        assert_eq!(hits[0].preview, "Nitrogen gas surrounds us all.");
        assert_eq!(hits[0].sections, None);
        let json = serde_json::to_string(&hits[0]).unwrap();
        assert!(!json.contains("sections"), "{json}");
        // The article also matched in the full-text tier (its body carries
        // the query words): reported exactly once, from the title tier.
        assert_eq!(hits.iter().filter(|h| h.path == "C/Nitrogen_Gas_Effects").count(), 1);
        // The full-text-only matches follow, with full-text hit semantics.
        assert_eq!(hits[1].path, "C/Weather", "{hits:?}");
        assert_eq!(hits[1].title, "Weather");
        assert_eq!(hits[1].sections, Some(vec!["_intro".to_string()]));

        // No usable terms: the title tier's AND over words absent from the
        // title index matches nothing and the full-text tier matches
        // nothing either (no such terms in this index) - no hits, no error.
        let hits = search(&server, "the of");
        assert!(hits.is_empty(), "{hits:?}");
    }

    /// An archive whose titles do NOT contain every query word: under the
    /// title tier's AND-only query a partial title match must not be
    /// promoted at all - that judgment belongs to the full-text tier's
    /// BM25. This is the measured junk source of the old title band's OR
    /// sub-band (any title sharing ONE query word flooded ahead of the
    /// full-text results).
    fn title_and_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let titles = make_index(&[
            ("C/New_York_City", "new york city", "New York City"),
            ("C/Quebec_City", "quebec city", "Quebec City"),
            ("C/Kansas_City", "kansas city", "Kansas City"),
            ("C/Mexico_City", "mexico city", "Mexico City"),
            ("C/Weather", "weather", "Weather"),
        ]);
        let index = make_index(&[
            ("C/New_York_City", "new york citi largest unit state", "New York City"),
            ("C/Quebec_City", "quebec citi capit provinc", "Quebec City"),
            // "kansa", not "kansas": the index terms are the STEMS the
            // query parser produces (Porter2 strips the s), as libzim
            // indexes them.
            ("C/Kansas_City", "kansa citi straddl state", "Kansas City"),
            ("C/Mexico_City", "mexico citi capit", "Mexico City"),
            ("C/Weather", "weather forecast effect atmospher", "Weather"),
        ]);
        let content = [
            // Empty directory-entry titles, as in modern openZIM archives:
            // the titles live in the indexes (value slot 0) only.
            TestEntry { namespace: b'C', url: "New_York_City", title: "", mime: 0, body: b"<html><body><h1>New York City</h1><p>New York City is the largest city in the United States. It sits at the mouth of the Hudson.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Quebec_City", title: "", mime: 0, body: b"<html><body><h1>Quebec City</h1><p>Quebec City is the capital of the province.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Kansas_City", title: "", mime: 0, body: b"<html><body><h1>Kansas City</h1><p>Kansas City straddles two states.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Mexico_City", title: "", mime: 0, body: b"<html><body><h1>Mexico City</h1><p>Mexico City is the capital of Mexico.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Weather", title: "", mime: 0, body: b"<html><body><h1>Weather</h1><p>Weather forecasts describe the atmosphere.</p></body></html>" },
        ];
        let bytes = build_archive_indexes(&["text/html"], &content, &[], 0, Some(&index), Some(&titles));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_title_and_tier_matches_all_words_titles() {
        let (server, _keep) = title_and_test_server();

        // "new york" is nobody's URL (the article is C/New_York_City), but
        // the article's title contains BOTH query words: the title tier's
        // AND surface match ranks it first, styled as a title match. Its
        // lead holds two sentences; the preview is the FIRST one only
        // (a full-text preview would have continued into the next one).
        let hits = search(&server, "new york");
        assert!(!hits.is_empty(), "{hits:?}");
        assert_eq!(hits[0].path, "C/New_York_City", "{hits:?}");
        assert_eq!(hits[0].title, "New York City");
        assert_eq!(hits[0].sections, None);
        assert_eq!(hits[0].preview, "New York City is the largest city in the United States.");

        // A partial title match ("kansas city" shares two words with the
        // "New York City" title, one with "Mexico City" and "Quebec City")
        // is NOT promoted to the title tier: no title contains all three
        // query words, so every hit comes from the full-text tier with
        // full-text semantics (sections reported).
        let hits = search(&server, "kansas city new");
        assert!(hits.len() >= 2, "{hits:?}");
        for hit in &hits {
            assert!(hit.sections.is_some(), "{hit:?}");
        }
        // The city articles are still reachable - through full text.
        let pos = |t: &str| hits.iter().position(|h| h.title == t);
        assert!(pos("Kansas City").is_some() && pos("New York City").is_some(), "{hits:?}");

        // A single-word query is the one-word AND: title-tier hits first.
        let hits = search(&server, "city");
        assert!(!hits.is_empty(), "{hits:?}");
        assert_eq!(hits[0].sections, None, "{hits:?}");
        assert!(hits[0].title.contains("City"), "{hits:?}");
    }

    /// The title index stores titles as WRITTEN - lowercased surface word
    /// forms (measured on md1m: df("beatles")=186 against df("beatl")=0;
    /// on fr.zim "revolution" 328) - so the tier matches words by their
    /// surface form only; no stem variants. Titles deliberately sit on URLs
    /// the queries cannot hit exactly, so every reported hit really comes
    /// from the title tier.
    fn title_inflection_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        // Title index: surface word forms, as libzim's indexer stores them.
        let titles = make_index(&[
            ("C/Physics1", "black hole", "Black hole"),
            ("C/Physics2", "black holes", "Black holes"),
            ("C/Movie1", "holes", "Holes"),
            ("C/Movie2", "movi", "The Movie"),
        ]);
        // Full-text index: unprefixed stems, as libzim's STEM_ALL builds it.
        let index = make_index(&[
            ("C/Physics1", "black hole graviti spacetime", "Black hole"),
            ("C/Physics2", "black hole graviti spacetime", "Black holes"),
            ("C/Movie1", "hole plot movi", "Holes"),
            ("C/Movie2", "movi film", "The Movie"),
            // Fillers: on the four-document index above "hole" (the stem of
            // both "hole" and "holes") covers 3 of 4 documents (>50%) - a
            // background word the fulltext query's filter
            // (FT_WORD_MAX_DF_FRAC) drops, and the body matches below would
            // lose Movie1. Six documents hold it at exactly 50%: kept.
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
            ("C/Glacier", "glacier ice dens movem weight flow", "Glacier"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Physics1", title: "", mime: 0, body: b"<html><body><h1>Black hole</h1><p>A black hole bends spacetime.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Physics2", title: "", mime: 0, body: b"<html><body><h1>Black holes</h1><p>Black holes bend spacetime.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Movie1", title: "", mime: 0, body: b"<html><body><h1>Holes</h1><p>The plot of Holes moves to a camp.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Movie2", title: "", mime: 0, body: b"<html><body><h1>The Movie</h1><p>The movie film runs two hours.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Banana", title: "", mime: 0, body: BANANA_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Glacier", title: "", mime: 0, body: b"<html><body><h1>Glacier</h1><p>A glacier is a body of dense ice.</p></body></html>" },
        ];
        let bytes = build_archive_indexes(&["text/html"], &content, &[], 0, Some(&index), Some(&titles));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_title_and_tier_matches_surface_forms_only() {
        let (server, _keep) = title_inflection_test_server();

        // The title tier is an AND over the SURFACE forms: "black holes"
        // matches only the "Black holes" title (the "Black hole" title has
        // no surface "holes", and there are no stem variants anymore).
        // The body-matching articles follow through the full-text tier:
        // Physics1 and Movie1 match the parsed OR (black/hole), Physics2 is
        // already reported (deduped by terminal path), and "The Movie"
        // matches nothing. Under the old per-word OR(surface, stem) band
        // the "Black hole" title doc ALSO matched (via the stem variant).
        let hits = search(&server, "black holes");
        assert_eq!(hits.len(), 3, "{hits:?}");
        assert_eq!(hits[0].path, "C/Physics2", "{hits:?}");
        assert_eq!(hits[0].title, "Black holes");
        assert_eq!(hits[0].sections, None, "{hits:?}");
        assert_eq!(hits[1].path, "C/Physics1", "{hits:?}");
        assert_eq!(hits[2].path, "C/Movie1", "{hits:?}");

        // One word: the one-word AND matches every title containing the
        // surface word - "Holes" and "Black holes", ranked by BM25 within
        // the tier (the one-word title first), then the body-only matches.
        let hits = search(&server, "holes");
        assert_eq!(hits.len(), 3, "{hits:?}");
        assert_eq!(hits[0].path, "C/Movie1", "{hits:?}");
        assert_eq!(hits[1].path, "C/Physics2", "{hits:?}");
        assert_eq!(hits[2].path, "C/Physics1", "{hits:?}");
    }

    /// An archive whose title index carries a punctuation-delimited title
    /// exactly the way libzim's indexer stores titles: "Cathédrale
    /// Notre-Dame de Paris" is indexed as the folded surface words
    /// cathedrale, notre, dame, de, paris - the indexer splits on
    /// non-alphanumeric characters, so no fused "notredame" term exists
    /// (measured on fr.zim's title index: notre df=164, dame df=191,
    /// notredame absent). The article's full-text document matches none of
    /// the query's words, so the title band is the ONLY tier that can
    /// retrieve it.
    fn hyphen_title_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let titles = make_index(&[
            (
                "C/Notre_Dame_De_Paris",
                "cathedrale notre dame de paris",
                "Cathédrale Notre-Dame de Paris",
            ),
        ]);
        let index = make_index(&[
            ("C/Notre_Dame_De_Paris", "church gothic island french landmark", "Notre-Dame"),
        ]);
        let notre_dame_html: &'static [u8] = "<html><body><h1>Cathédrale Notre-Dame de Paris</h1>\
            <p>The cathedral stands on the Île de la Cité.</p></body></html>".as_bytes();
        let content = [
            TestEntry {
                namespace: b'C',
                url: "Notre_Dame_De_Paris",
                title: "",
                mime: 0,
                body: notre_dame_html,
            },
        ];
        let bytes = build_archive_indexes(&["text/html"], &content, &[], 0, Some(&index), Some(&titles));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_title_band_splits_punctuation_delimited_words() {
        let (server, _keep) = hyphen_title_test_server();

        // The title tier's AND is built from the query's punctuation-SPLIT
        // words (cathedrale, notre, dame, de, paris): every one is a term
        // of the title-index document, so the article is found, styled as a
        // title match (first intro sentence, no sections). While
        // `title_words` still FUSED punctuation-delimited words
        // ("notre-dame" -> "notredame"), the AND queried a term no title
        // document carries and the band came back silently empty - this
        // search returned nothing at all.
        let hits = search(&server, "cathédrale notre-dame de paris");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Notre_Dame_De_Paris", "{hits:?}");
        assert_eq!(hits[0].title, "Cathédrale Notre-Dame de Paris");
        assert_eq!(hits[0].preview, "The cathedral stands on the Île de la Cité.");
        assert_eq!(hits[0].sections, None);

        // The fused form is nobody's term: not a URL/title, not in the
        // title index (the indexer splits it), not in the full-text index.
        let hits = search(&server, "notredame");
        assert!(hits.is_empty(), "{hits:?}");
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
            // Filler: on the three-document index above, "nitrogen" and
            // "naca" each cover 2 of 3 documents (>50%) - background words
            // the fulltext query's filter (FT_WORD_MAX_DF_FRAC) drops, and
            // the BM25 runner-up assertions below need both words matched.
            // Four documents hold both at exactly 50%: kept.
            ("C/Weather", "weather forecast rain snow climat", "Weather"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Nitrogen", title: "Nitrogen", mime: 0, body: NITROGEN_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Atmosphere", title: "Atmosphere", mime: 0, body: ATMOSPHERE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Aeronautics", title: "Aeronautics", mime: 0, body: AERONAUTICS_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Weather", title: "Weather", mime: 0, body: b"<html><body><h1>Weather</h1><p>Weather forecasts describe rain and snow.</p></body></html>" },
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
        // An exact match is a title match: the lead's first sentence, no
        // sections.
        assert_eq!(hits[0].preview, "Nitrogen is a colorless, odorless gas.");
        assert_eq!(hits[0].sections, None);
        assert!(!serde_json::to_string(&hits[0]).unwrap().contains("sections"));
        // The BM25 runner-up is still reported, behind the exact match.
        assert_eq!(hits[1].path, "C/Atmosphere", "{hits:?}");
    }

    #[test]
    fn e2e_search_exact_redirect_reports_terminal_article() {
        let (server, _keep) = exact_test_server();

        // "NACA" is a redirect (directory title "NACA") onto the Aeronautics
        // article. Redirects are not in the full-text index, so without the
        // directory lookup this query would report Atmosphere first (it
        // mentions "naca" twice). The exact tier follows the redirect
        // chain: the RESULT reports the TERMINAL article's title and path,
        // not the redirect's, and the same article's full-text hit dedupes
        // into it.
        let hits = search(&server, "NACA");
        assert_eq!(hits[0].path, "C/Aeronautics", "{hits:?}");
        assert_eq!(hits[0].title, "Aeronautics");
        // The preview is the terminal article's lead, first sentence.
        assert_eq!(hits[0].preview, "Aeronautics is the science of flight.");
        assert_eq!(hits[0].sections, None);
        // Fulltext hits follow in BM25 order - Aeronautics itself is
        // already reported once.
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[1].path, "C/Atmosphere", "{hits:?}");

        // A redirect with an empty directory title: the query still finds
        // it via the case variants of its URL, and the terminal article is
        // reported again (its directory title names it).
        let hits = search(&server, "usa");
        assert_eq!(hits[0].path, "C/Aeronautics", "{hits:?}");
        assert_eq!(hits[0].title, "Aeronautics");
    }

    /// An archive whose TITLE INDEX contains a redirect document, the way
    /// old-namespace openZIM archives do: the document's data path points
    /// at the redirect entry and value slot 0 carries the redirect's own
    /// title - while the directory also holds the redirect. One query can
    /// then reach the same article three ways: exact probe (redirect),
    /// title tier (redirect document), and full text (the target article).
    fn redirect_dedupe_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/Aeronautics", "aeronautics naca aviation flight", "Aeronautics"),
        ]);
        let titles = make_index(&[
            ("C/Aeronautics", "aeronautics", "Aeronautics"),
            // The redirect's own title-index document.
            ("C/NACA", "naca", "NACA"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Aeronautics", title: "Aeronautics", mime: 0, body: AERONAUTICS_HTML.as_bytes() },
        ];
        let redirects = [
            TestRedirect { namespace: b'C', url: "NACA", title: "NACA", target_content: 0 },
        ];
        let bytes = build_archive_indexes(&["text/html"], &content, &redirects, 0, Some(&index), Some(&titles));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn e2e_search_redirect_dedupes_across_tiers_by_terminal_path() {
        let (server, _keep) = redirect_dedupe_test_server();

        // All three tiers reach the Aeronautics article - the exact probe
        // through the directory redirect, the title tier through the
        // redirect's own title-index document, and the full-text tier
        // through the article body - and it must be reported exactly once,
        // at its highest rank (the exact tier's), under the terminal
        // article's identity.
        let hits = search(&server, "naca");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Aeronautics", "{hits:?}");
        assert_eq!(hits[0].title, "Aeronautics");
        assert_eq!(hits[0].preview, "Aeronautics is the science of flight.");
        assert_eq!(hits[0].sections, None);

        // The article's own title still routes through the tiers to one
        // hit (exact + title index + full text again).
        let hits = search(&server, "aeronautics");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Aeronautics", "{hits:?}");
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
        let index = make_index(&[
            (
                "C/Salt",
                "salt mineral chlorid sodium himalaya deposit rock bed form sea season food",
                "Salt",
            ),
            // Fillers: a one-document index makes "salt" 100%-df - a
            // background word the fulltext query's filter
            // (FT_WORD_MAX_DF_FRAC) drops, and both searches below would
            // return nothing. Three documents keep the query words at 33%.
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
            ("C/Glacier", "glacier ice dens movem weight flow", "Glacier"),
        ]);
        let content = [
            TestEntry {
                namespace: b'C',
                url: "Salt",
                title: "Salt",
                mime: 0,
                body: SALT_HTML.as_bytes(),
            },
            TestEntry {
                namespace: b'C',
                url: "Banana",
                title: "Banana",
                mime: 0,
                body: BANANA_HTML.as_bytes(),
            },
            TestEntry {
                namespace: b'C',
                url: "Glacier",
                title: "Glacier",
                mime: 0,
                body: b"<html><body><h1>Glacier</h1><p>A glacier is a body of dense ice.</p></body></html>",
            },
        ];
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

    #[test]
    fn e2e_search_all_words_doc_wins_under_plain_or() {
        // Plain BM25 (OR over the query terms) must rank the document that
        // mentions EACH query term above the ones that repeat a single
        // term: BM25 saturates term frequency (30 repetitions of "cherri"
        // are worth only ~1.2x one occurrence) and normalizes by document
        // length, while IDF rewards the second, rare term. The old code
        // needed a hand-rolled all-words AND branch (double-counting every
        // document that matched it) for this; BM25 does it alone.
        let dir = tempfile::tempdir().unwrap();
        let cherry_terms = format!("{}{}", "cherri ".repeat(30), "filler ".repeat(400));
        let pie_terms = format!("{}{}", "pie ".repeat(30), "filler ".repeat(400));
        let index = make_index(&[
            ("C/Cherry", cherry_terms.as_str(), "Cherry"),
            ("C/Dessert_Recipes", "cherri cherri pie pie", "Dessert Recipes"),
            ("C/Pie_1", pie_terms.as_str(), "Pie 1"),
            ("C/Pie_2", pie_terms.as_str(), "Pie 2"),
            // Fillers: on the four-document index above "pie" covers 3 of 4
            // documents (>50%) - a background word the fulltext query's
            // filter (FT_WORD_MAX_DF_FRAC) drops, collapsing the query to
            // "cherry", which the tf-30 "Cherry" document would outrank
            // "Dessert_Recipes" on. Six documents hold "pie" at exactly
            // 50%: kept (only MORE than half is dropped), and the OR
            // ranking below is unchanged.
            ("C/Mango", "mango tropic tree sweet", "Mango"),
            ("C/Peach", "peach orchard stone fruit", "Peach"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "Cherry", title: "Cherry", mime: 0, body: CHERRY_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Dessert_Recipes", title: "Dessert Recipes", mime: 0, body: CHERRY_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Pie_1", title: "Pie 1", mime: 0, body: CHERRY_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Pie_2", title: "Pie 2", mime: 0, body: CHERRY_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Mango", title: "Mango", mime: 0, body: b"<html><body><h1>Mango</h1><p>A mango is a tropical stone fruit.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Peach", title: "Peach", mime: 0, body: b"<html><body><h1>Peach</h1><p>A peach grows in orchards.</p></body></html>" },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        let server = ZimMcpServer::new(library);

        let hits = search(&server, "cherry pie");
        assert_eq!(hits[0].path, "C/Dessert_Recipes", "{hits:?}");
        assert_eq!(hits[0].title, "Dessert Recipes");
        // The single-term matches still appear, behind the all-words doc.
        assert_eq!(hits[1].path, "C/Cherry", "{hits:?}");
        assert_eq!(hits[2].path, "C/Pie_1", "{hits:?}");
        assert_eq!(hits[3].path, "C/Pie_2", "{hits:?}");
        // Neither hit's lead covers both terms, and "cherry" does match the
        // intro: it is reported as the matching region _intro, and the
        // preview is the intro's matching paragraph.
        assert_eq!(hits[0].sections, Some(vec!["_intro".to_string()]));
        assert!(hits[0].preview.contains("cherry is the fruit"), "{:?}", hits[0].preview);
    }

    #[test]
    fn e2e_search_stopword_only_query_returns_nothing() {
        let (server, _keep) = test_server();

        // A query made only of function words matches nothing on this
        // archive (none of the words is in the index) and must not error.
        // Real libzim indexes do carry such words (nothing stops them), so
        // there this stays graceful junk - like every OR query.
        let hits = search(&server, "the in of");
        assert!(hits.is_empty(), "{hits:?}");
    }

    #[test]
    fn e2e_search_nonsense_word_still_returns_results() {
        let (server, _keep) = test_server();

        // The indexes have no spelling data, so a nonsense word matches
        // nothing and the OR over the real word still retrieves its
        // results.
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
        let index = make_index(&[
            (
                "C/Salt",
                "salt miner primari sodium chlorid himalaya deposit rock",
                "Salt",
            ),
            // Fillers: a one-document index makes "salt" 100%-df - a
            // background word the fulltext query's filter
            // (FT_WORD_MAX_DF_FRAC) drops, and the search below would
            // return nothing. Three documents keep it at 33%.
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
            ("C/Cherry", "cherri pie fruit tree", "Cherry"),
        ]);
        let content = [
            TestEntry {
                namespace: b'C',
                url: "Salt",
                title: "Salt",
                mime: 0,
                body: SALT_HTML.as_bytes(),
            },
            TestEntry {
                namespace: b'C',
                url: "Banana",
                title: "Banana",
                mime: 0,
                body: BANANA_HTML.as_bytes(),
            },
            TestEntry {
                namespace: b'C',
                url: "Cherry",
                title: "Cherry",
                mime: 0,
                body: CHERRY_HTML.as_bytes(),
            },
        ];
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

    /// A filler markdown article for fixtures that would otherwise carry a
    /// one-document index (every word 100%-df, dropped by the fulltext
    /// query's background-vocabulary filter).
    const BANANA_MD: &str = "\
# Banana

A banana is a tall herbaceous plant.
";

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
        let index = make_index(&[
            ("C/Zinc", "zinc chemic element symbol smelt ancient india", "Zinc"),
            // Fillers (same reason as e2e_search_and_get_single_file_library):
            // a one-document index makes every word 100%-df, which the
            // fulltext query's filter (FT_WORD_MAX_DF_FRAC) drops.
            ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
            ("C/Glacier", "glacier ice dens movem weight flow", "Glacier"),
        ]);
        let content = [
            TestEntry {
                namespace: b'C',
                url: "Zinc",
                title: "Zinc",
                mime: 0,
                body: ZINC_MD.as_bytes(),
            },
            TestEntry {
                namespace: b'C',
                url: "Banana",
                title: "Banana",
                mime: 0,
                body: BANANA_MD.as_bytes(),
            },
            TestEntry {
                namespace: b'C',
                url: "Glacier",
                title: "Glacier",
                mime: 0,
                body: GLACIER_MD.as_bytes(),
            },
        ];
        let bytes = build_archive(&["text/markdown"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("md.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        let server = ZimMcpServer::new(library);

        // Search: the preview is plain text derived from the Markdown, free of
        // markup, and is the lead's first sentence - the leading `# Zinc`
        // title line (a separate field of every hit) and the hatnote are
        // dropped. An exact title match never carries sections.
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
