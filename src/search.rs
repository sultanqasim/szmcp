//! The search pipeline behind the `zim_search` tool: three simple tiers
//! (exact title/URL probe, all-words title match, full-text OR over all
//! the query words), ranked against the ZIM embedded Xapian indexes,
//! merged across archives (optionally restricted to the single archive a
//! `zim` name resolves to) - the full-text tier by re-scoring every
//! archive's candidates with one BM25 over pooled cross-archive
//! statistics - and reported with per-hit previews.

use crate::html;
use crate::markdown;
use crate::tools::ToolError;
use crate::zim::{Archive, ZimLibrary};
use schemars::JsonSchema;
use serde::Serialize;
use std::sync::Arc;
use xapian2::{
    bm25_weight, resolve_stem_language, Document, Enquire, Operator, Query, QueryParser, Stem,
    StemStrategy, WritableDatabase,
};

/// Number of results `zim_search` returns in total (across all archives).
const SEARCH_LIMIT: u32 = 20;
/// How many full-text candidates each archive contributes to the pooled
/// re-ranking (its own BM25 order, best first): [`SEARCH_LIMIT`] results
/// must survive the merge, and pooling rewards an archive with more than
/// one strong match, so the per-archive fetch is well past the limit.
const FT_FETCH_PER_ARCHIVE: u32 = 50;
/// Maximum characters of the `preview` reported per search hit. A longer
/// preview is cut at the last word boundary at or before the cap (a single
/// word longer than the cap takes the hard cut) - see [`preview_truncate`].
const INTRO_CHARS: usize = 360;
/// Raw bytes read of an article to locate its matches (regions and
/// paragraphs). For compressed clusters the whole cluster decompresses anyway.
const HIT_READ_BYTES: u64 = 1024 * 1024;
/// Relative BM25 threshold for the `sections` of a fulltext hit: a region
/// is reported when its BM25 score for the hit's full-text query is at
/// least this fraction of the article's best-scoring region.
const SECTION_MIN_SCORE_FRAC: f64 = 0.5;

/// Most of the article's regions qualifying for the `sections` list means the
/// whole article is relevant: report no list rather than a near-complete one.
const SECTION_MAX_COVERAGE: f64 = 0.4;
/// One search result.
#[derive(Serialize, JsonSchema, Debug)]
pub struct SearchHit {
    /// ZIM file name, relative to the ZIM directory
    pub zim: String,
    /// Path of the article inside the ZIM file
    pub path: String,
    /// Page/article title
    pub title: String,
    /// Preview of the article: the intro's first sentence for title matches,
    /// otherwise the lead (the first intro paragraph), capped at the length
    /// above
    pub preview: String,
    /// Regions holding query matches (`_intro` first when it matched, then
    /// sections in document order), BM25-ranked against the query with the
    /// best-scoring regions kept; absent when the query matches the title
    /// or no region matches any query word
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sections: Option<Vec<String>>,
}

/// The search result set (best matches first).
#[derive(Serialize, JsonSchema, Debug)]
pub struct SearchResults {
    /// The search results
    pub results: Vec<SearchHit>,
}

/// Fold accents the way libzim indexes text (`removeAccents` in tools.cpp:
/// `Lower; NFD; [:M:] remove; NFC`), so "élections" is indexed as
/// "elections" and neither embedded index carries accented terms (measured
/// on fr.zim's full-text index: "revolu" df 11042, every accented variant
/// absent). Query words must take the same route. Only marks in
/// U+0300..=U+036F are stripped - every mark a Latin, Greek, or Cyrillic
/// letter decomposes to sits in that range, no measured archive carries
/// more (Hebrew nikkud, Arabic harakat, Indic signs).
fn fold_accents(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.to_lowercase()
        .nfd()
        .filter(|c| !('\u{300}'..='\u{36f}').contains(c))
        .nfc()
        .collect()
}

/// The title tier's query: `text` (the folded query with punctuation mapped
/// to spaces, see `title_text`) parsed with the QueryParser, which
/// tokenizes like the indexer's TermGenerator ("notre-dame" splits into
/// `notre` + `dame`, the surface forms a title index carries) and ANDs all
/// words. `None` when `text` has no words (an all-punctuation query): the
/// caller skips the band, as [`fulltext_query`]'s `None` skips the fulltext
/// band.
fn title_query(text: &str) -> Result<Option<Query>, ToolError> {
    if text.split_whitespace().next().is_none() {
        return Ok(None);
    }
    let mut qp = QueryParser::new()?;
    qp.set_stemmer("none")?;
    qp.set_stemming_strategy(StemStrategy::None)?;
    qp.set_default_op(Operator::And)?;
    let query = qp
        .parse_query(text)
        .map_err(|e| ToolError::InvalidArgument(format!("failed to parse title query: {e}")))?;
    Ok(Some(query))
}

/// A stemmer that memoizes every stem: one instance serves one archive's
/// whole search, and every uncached stem crosses the Xapian FFI, so text's
/// heavy word repetition pays for the cache.
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
    /// stems), so raw article words stay comparable with the folded query
    /// terms. The fold runs on a cache miss only; folding every occurrence
    /// instead was measured at 4.7x the search time on fr.zim (9.7s against
    /// 2.0s) - two Unicode normalization passes per word are not free.
    fn stem(&mut self, word: &str) -> &str {
        if !self.cache.contains_key(word) {
            let folded = fold_accents(word);
            let stemmed = self.stem.apply(&folded).unwrap_or(folded);
            self.cache.insert(word.to_string(), stemmed);
        }
        self.cache[word].as_str()
    }

    /// [`Stemmer::stem`] for the query path: those words arrive
    /// accent-folded once in `search`, so the fold would be an identity
    /// pass. Same cache, so words folding to the same form share the stem.
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

/// One archive's view of a query (see [`search`]). Everything here depends
/// on the archive's language: libzim stems each archive's embedded index
/// with the stemmer chosen from its `Language` metadata, so querying a
/// French archive with an English stemmer finds nothing (and vice versa).
struct ArchiveQuery {
    stemmer: Stemmer,
    /// Folded stems of the query's words (deduped, first-occurrence order):
    /// the terms the per-hit section index receives the within-region
    /// frequencies of.
    terms: Vec<String>,
    /// The title tier's query text: the folded query with every
    /// non-alphanumeric character mapped to a space, so the QueryParser
    /// (see [`title_query`]) only ever sees clean words - no `foo:bar`
    /// prefixes, operators, or phrases - while it owns the tokenization.
    /// Already lowercase (the fold lowercases).
    title_text: String,
    /// The fulltext tier's parsed query ([`fulltext_query`]): every
    /// whitespace token of the folded query, as written - tokens, not
    /// `title_text`'s per-word split, because the fulltext index is stemmed
    /// (`STEM_ALL`) and "notre-dame" must reach the parser whole. Also the
    /// query scored against each fulltext hit's section index. `None` for
    /// an all-punctuation query: the caller skips the band.
    ft_query: Option<Query>,
    /// The fulltext query's (term, wqf) pairs: the exact stems the parsed
    /// query addresses and how many query tokens yield each one - the term
    /// multiplicity Xapian calls wqf. The pooled re-ranking re-scores every
    /// candidate outside Xapian ([`FtPool`]) and needs both, since Xapian
    /// reports only per-database weights, which are not comparable across
    /// archives.
    ft_terms: Vec<(String, u32)>,
}

impl ArchiveQuery {
    fn build(arc: &Archive, query: &str) -> Result<Self, ToolError> {
        // The archive's stemmer language: its Language metadata mapped to a
        // code Xapian accepts (ISO-639-3 "fra" -> "fr"; unknown -> none).
        // Archives without the metadata keep English stems - that is what
        // their indexes use.
        let language = arc.language().unwrap_or_else(|| "eng".to_string());
        let language = resolve_stem_language(&language);
        let mut stemmer = Stemmer::new(&language).map_err(|e| {
            ToolError::Internal(format!("failed to create {language} stemmer: {}", e.msg()))
        })?;
        // `query` arrives accent-folded (see `search`), so the tier words
        // stem through the no-second-fold path (`stem_folded`).
        let title_text: String = query
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();
        let mut terms: Vec<String> = Vec::new();
        for word in title_text.split_whitespace() {
            let stemmed = stemmer.stem_folded(word).to_string();
            if !terms.contains(&stemmed) {
                terms.push(stemmed);
            }
        }
        // The fulltext tier parses every token of the folded query as
        // written (see `fulltext_query`), punctuation-only ones included.
        let ft_words: Vec<String> = query.split_whitespace().map(str::to_string).collect();
        let ft_query = fulltext_query(&ft_words, &language)?;
        // The (term, wqf) pairs, derived per token: Xapian scores the
        // parser's flat OR as one wqf-1 leaf per occurrence, so a term's
        // multiplicity is the number of tokens whose parse yields it.
        // Re-parsing each token with the same settings `fulltext_query`
        // uses (the whole query is semantically the flat OR of those
        // parses) enumerates the stems through `Query::terms`; a parse
        // failing on a single token while the whole query parsed cannot
        // happen for the flags in use, and the whole-query parse is the
        // authority for matching, so a token that somehow fails is skipped.
        let mut ft_terms: Vec<(String, u32)> = Vec::new();
        for word in &ft_words {
            let Ok(Some(token_query)) = fulltext_query(std::slice::from_ref(word), &language)
            else {
                continue;
            };
            for term in token_query.terms().map_err(|e| {
                ToolError::Internal(format!("failed to enumerate query terms: {}", e.msg()))
            })? {
                match ft_terms.iter_mut().find(|(t, _)| *t == term) {
                    Some((_, wqf)) => *wqf += 1,
                    None => ft_terms.push((term, 1)),
                }
            }
        }
        Ok(Self { stemmer, terms, title_text, ft_query, ft_terms })
    }
}

/// The fulltext query of one archive: `words` (the folded query's
/// whitespace tokens, as written) joined by single spaces and parsed by the
/// archive-language QueryParser (`STEM_ALL`, default op OR) - a plain BM25
/// OR over all the query words, whose IDF down-weights the merely common
/// ones. Boolean syntax is lost: the words are already lowercased, so
/// AND/OR/NOT reach the parser as ordinary words. Returns `None` for an
/// all-punctuation query: the caller skips the fulltext band.
fn fulltext_query(words: &[String], language: &str) -> Result<Option<Query>, ToolError> {
    if words.is_empty() {
        return Ok(None);
    }
    // openZIM full-text indexes carry unprefixed, accent-folded Porter2
    // stems (libzim indexes STEM_ALL over removeAccents'd text); Xapian's
    // default strategy would turn lowercase terms into "Z"-prefixed stems
    // that never match. The words are pre-folded (see `search`).
    let mut qp = QueryParser::new()?;
    qp.set_stemmer(language)?;
    qp.set_stemming_strategy(StemStrategy::All)?;
    qp.set_default_op(Operator::Or)?;
    let xquery = qp
        .parse_query(&words.join(" "))
        .map_err(|e| ToolError::InvalidArgument(format!("failed to parse query: {e}")))?;
    Ok(Some(xquery))
}

/// One full-text candidate of one archive, with the raw per-term data the
/// pooled re-score ([`FtPool`]) needs: Xapian's own MSet weights come from
/// per-database statistics and are not comparable across archives, so the
/// archive's top candidates travel out of the handle-pool closure as data
/// instead of as weights.
struct FtCandidate {
    path: String,
    /// The index document's title (value slot 0).
    title: String,
    /// The document's length (the wdf sum of its termlist).
    doclen: f64,
    /// Within-document frequencies, aligned with [`FtArchive::terms`].
    wdfs: Vec<u32>,
}

/// One archive's full-text tier payload ([`fetch_fulltext`]): the
/// collection statistics the pooled re-score consumes plus the archive's
/// top candidates in its own BM25 order.
struct FtArchive {
    /// Documents in the archive's index (`doc_count`).
    docs: u32,
    /// Average document length in the archive's index (`average_length`).
    avlen: f64,
    /// The archive's (term, wqf) pairs ([`ArchiveQuery::ft_terms`]).
    terms: Vec<(String, u32)>,
    /// Each term's document frequency in the archive's index
    /// (`termfreq`), aligned with `terms`.
    dfs: Vec<u32>,
    /// The archive's own top candidates, its local BM25 order.
    candidates: Vec<FtCandidate>,
}

/// Fetch one archive's full-text tier payload: the archive's collection
/// statistics plus its top [`FT_FETCH_PER_ARCHIVE`] candidates (the
/// archive's own BM25 order) with the per-term raw data the pooled
/// re-score needs. Everything Xapian happens here - the search handles are
/// pooled and never shared between searches - so the raw numbers travel
/// out instead of the weights.
fn fetch_fulltext(
    h: &crate::zim::XapianHandles,
    query_state: &ArchiveQuery,
) -> Result<FtArchive, ToolError> {
    let docs = h.fulltext.doc_count();
    let avlen = h.fulltext.average_length();
    let terms = query_state.ft_terms.clone();
    let dfs: Vec<u32> = terms.iter().map(|(t, _)| h.fulltext.termfreq(t)).collect();
    let mut candidates = Vec::new();
    // `None` (an all-punctuation query) skips the band.
    if let Some(xquery) = &query_state.ft_query {
        let mut enquire = Enquire::new(&h.fulltext)?;
        enquire.set_query(xquery)?;
        enquire.set_sort_by_relevance();
        let mset = enquire.get_mset(0, FT_FETCH_PER_ARCHIVE, 0)?;
        for (j, m) in mset.iter().enumerate() {
            let mut doc = mset.document(j as u32)?;
            let path = doc.data_str()?;
            if path.is_empty() {
                continue;
            }
            let title = String::from_utf8_lossy(&doc.value(0)?).into_owned();
            candidates.push(FtCandidate {
                path,
                title,
                doclen: h.fulltext.doc_length(m.docid)?,
                wdfs: terms.iter().map(|(t, _)| h.fulltext.wdf(m.docid, t)).collect(),
            });
        }
    }
    Ok(FtArchive { docs, avlen, terms, dfs, candidates })
}

/// The pooled full-text statistics of one search: Xapian's per-database
/// statistics summed over the searched archives, exactly how Xapian itself
/// pools when one search spans several databases. Scoring every archive's
/// candidates against THESE makes the union one ranking, where each
/// archive's own weights are on incomparable per-database scales.
struct FtPool {
    /// Documents over the searched archives.
    docs: u32,
    /// Average document length over the searched archives (the total
    /// length over the total docs).
    avlen: f64,
    /// Pooled document frequency per term - the union of the archives'
    /// `ft_terms` with their `termfreq`s summed. A stem another archive's
    /// stemmer never produced is simply absent there and adds 0.
    df: std::collections::HashMap<String, u32>,
}

impl FtPool {
    fn new(per_archive: &[(&Arc<Archive>, FtArchive)]) -> Self {
        let mut docs = 0u32;
        let mut total_len = 0.0;
        let mut df = std::collections::HashMap::new();
        for (_, ft) in per_archive {
            docs += ft.docs;
            total_len += ft.docs as f64 * ft.avlen;
            for ((term, _), freq) in ft.terms.iter().zip(&ft.dfs) {
                *df.entry(term.clone()).or_insert(0) += freq;
            }
        }
        let avlen = if docs > 0 { total_len / docs as f64 } else { 0.0 };
        Self { docs, avlen, df }
    }

    /// One candidate's pooled BM25 score, computed with the same formula
    /// Xapian uses, so on the pooled statistics it equals what Xapian would
    /// score the candidate with ([`bm25_weight`]).
    fn score(&self, ft: &FtArchive, cand: &FtCandidate) -> f64 {
        let dfs: Vec<u32> =
            ft.terms.iter().map(|(t, _)| self.df.get(t).copied().unwrap_or(0)).collect();
        let wqfs: Vec<u32> = ft.terms.iter().map(|(_, wqf)| *wqf).collect();
        bm25_weight(self.docs, self.avlen, &dfs, &wqfs, cand.doclen, &cand.wdfs)
    }
}

/// Merge the archives' full-text candidates best-first: re-score every
/// candidate against the pooled statistics ([`FtPool`]) and sort the union.
/// Ties (identical term profiles and lengths score identically) keep a
/// deterministic order - archive name, then path.
fn pooled_fulltext(
    per_archive: Vec<(&Arc<Archive>, FtArchive)>,
) -> Vec<(&Arc<Archive>, String, String)> {
    let pool = &FtPool::new(&per_archive);
    let mut scored: Vec<(&Arc<Archive>, &FtCandidate, f64)> = per_archive
        .iter()
        .flat_map(|(arc, ft)| ft.candidates.iter().map(move |c| (*arc, c, pool.score(ft, c))))
        .collect();
    scored.sort_by(|(arc_a, cand_a, score_a), (arc_b, cand_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| arc_a.name.cmp(&arc_b.name))
            .then_with(|| cand_a.path.cmp(&cand_b.path))
    });
    scored
        .into_iter()
        .map(|(arc, cand, _)| (arc, cand.path.clone(), cand.title.clone()))
        .collect()
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

/// Search the ZIM files of the library - the pipeline behind the
/// `zim_search` tool: ranked hits, best first. `only` restricts the whole
/// pipeline to the one named archive (`None` searches every archive); the
/// name is the relative form results report - see [`ZimLibrary::archive`].
pub fn search(
    library: &ZimLibrary,
    only: Option<&str>,
    query: &str,
) -> Result<SearchResults, ToolError> {
    if query.trim().is_empty() {
        return Err(ToolError::InvalidArgument("query must not be empty".into()));
    }

    // The archives the search runs over: the one named archive when a filter
    // is given (every per-archive step below covers only it), otherwise all
    // of them in scan order. An unresolvable filter name (unknown, or one
    // that leaves the ZIM directory) is a NotFound - the same message shape
    // the article lookups report.
    let archives: Vec<&Arc<Archive>> = match only {
        None => library.archives.iter().collect(),
        Some(name) => match library.archive(name) {
            Some(arc) => vec![arc],
            None => {
                return Err(ToolError::NotFound(format!(
                    "ZIM file not found: {name} (loaded: {})",
                    library
                        .archives
                        .iter()
                        .map(|a| a.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
        },
    };

    // Quoted phrases build OP_PHRASE subqueries, but the indexes carry no
    // positional data (libzim indexes `index_text_without_positions`), so a
    // phrase matches nothing and silently loses the full-text band: strip
    // the quotes. The fold is computed once over the quote-stripped query
    // and the SAME folded string serves every tier (see `ArchiveQuery`);
    // the exact title/URL probe still gets the raw query, since article
    // paths and directory titles carry their accents ("C/Université").
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

    // One query view per archive under the search (see [`ArchiveQuery`]:
    // the embedded index's stemmer is per archive).
    let mut queries: Vec<ArchiveQuery> = archives
        .iter()
        .map(|arc| ArchiveQuery::build(arc, &folded_query))
        .collect::<Result<_, ToolError>>()?;

    // Tier 1 - exact title/URL matches from the ZIM directory itself:
    // redirects are not in the search indexes, and a query that names an
    // article exactly must rank first no matter what BM25 produces. The
    // entry is resolved to its terminal article below.
    let mut merged: Vec<(&Arc<Archive>, String, String, HitKind)> = Vec::new();
    for arc in &archives {
        if let Some((path, _)) = arc.lookup_exact(query)? {
            merged.push((*arc, path, String::new(), HitKind::Exact));
        }
    }

    // Tiers 2 and 3, per archive. Each archive's Xapian handles are checked
    // out of that archive's pool for the duration of the search:
    // concurrent searches never share a handle (Xapian does not support
    // concurrent calls on one database object).
    //
    // Tier 2 - the title tier queries the archive's title index with an AND
    // over ALL the query words' surface forms (`title_query`): a title
    // saying the whole query is far stronger evidence than body words, and
    // a partial title match is not promoted at all - that judgment is left
    // to the full-text BM25.
    //
    // Tier 3 - the full-text tier: each archive's candidates are fetched
    // with its own BM25 ranking ([`FT_FETCH_PER_ARCHIVE`] of them) and
    // re-scored across archives with one BM25 over pooled statistics below.
    let mut title_lists: Vec<(&Arc<Archive>, Vec<(String, String)>)> = Vec::new();
    // Full-text tier: the archive's collection statistics plus its top
    // candidates with the per-term raw data pooled re-scoring needs
    // ([`FtArchive`]).
    let mut ft_archives: Vec<(&Arc<Archive>, FtArchive)> = Vec::new();
    for (&arc, query_state) in archives.iter().zip(queries.iter_mut()) {
        let Some((title_list, ft)) = arc.with_xapian(|h| -> Result<_, ToolError> {
            let mut title_list = Vec::new();
            if let Some(and_query) = title_query(&query_state.title_text)? {
                if let Some(title_db) = &h.title {
                    let mut enquire = Enquire::new(title_db)?;
                    enquire.set_sort_by_relevance();
                    enquire.set_query(&and_query)?;
                    let mset = enquire.get_mset(0, SEARCH_LIMIT, 0)?;
                    for j in 0..mset.size() {
                        let mut doc = mset.document(j)?;
                        // The title-index document's data is the article
                        // path, its value slot 0 the title (same shape as
                        // the full-text index, one shared docid space); a
                        // producer that left either empty is filled in from
                        // the full-text document of the same docid.
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

            // The full-text tier: fetch the archive's payload for the
            // pooled re-ranking (`None`, an all-punctuation query, skips
            // the band inside `fetch_fulltext`).
            let ft = fetch_fulltext(h, query_state)?;
            Ok((title_list, ft))
        })?
        else {
            continue;
        };
        title_lists.push((arc, title_list));
        ft_archives.push((arc, ft));
    }

    if ft_archives.is_empty() {
        return Err(ToolError::Internal(format!(
            "no ZIM files with a Xapian full-text index were found in {}",
            library.root.display()
        )));
    }

    // The tiers keep their order (exact, then title, then full-text). The
    // title tier merges its archives' ranked lists by rotation: Xapian
    // weights are computed from per-database statistics and are not
    // comparable across archives, so the tier interleaves them - every
    // archive contributes its best match before any archive contributes its
    // second best.
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
    // The full-text tier merges by re-ranking instead of rotating: the
    // per-database BM25 weights are not comparable across archives (an
    // archive with 60 documents and one with 600k weight their own matches
    // on different scales), so every candidate is re-scored with ONE BM25
    // formula over statistics pooled across the searched archives
    // ([`FtPool`]) and the union is sorted - one ranking, where a strong
    // match in a small archive beats a weak match in a big one instead of
    // the archives merely taking turns.
    for (arc, path, title) in pooled_fulltext(ft_archives) {
        merged.push((arc, path, title, HitKind::Fulltext));
    }

    // Resolve every candidate to its terminal entry before dedupe, then
    // dedupe on that identity: the same article must never appear twice no
    // matter how many tiers and redirect spellings reach it. The identity
    // is the archive together with the terminal path: within one archive
    // the terminal path is the article (title-index redirect documents
    // carry their OWN title in value slot 0, and several redirect
    // spellings collapse onto one terminal path). Titles never dedupe
    // across archives: different archives may hold different articles
    // under the same title, so each archive's copy is reported on its
    // own. The reported title is the terminal entry's directory title,
    // else the index title, else derived from the terminal path.
    // Resolution failures keep the raw path.
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
            if !seen_paths.insert((Arc::as_ptr(arc) as usize, path.clone())) {
                return None;
            }
            Some((arc, path, title, kind))
        })
        .collect();
    merged.truncate(SEARCH_LIMIT as usize);

    let mut hits = Vec::with_capacity(merged.len());
    for (arc, path, title, kind) in &merged {
        // Exact and title-tier hits are title matches: first intro sentence
        // as preview, no sections.
        let title_match = *kind != HitKind::Fulltext;
        let (mime, bytes) = match arc.article_preview(path, HIT_READ_BYTES) {
            Ok(Some((_, mime, bytes))) => (mime, bytes),
            _ => (None, Vec::new()),
        };
        let article = String::from_utf8_lossy(&bytes);
        // Markdown editions carry plain Markdown, not HTML: pick the matching
        // splitter so the paragraphs and section names are free of markup.
        let is_markdown = mime.as_deref().is_some_and(|m| m.contains("markdown"));
        // Section scoring runs the hit archive's own full-text query and its
        // stemmer (the archive is always from the library, so the lookup
        // cannot fail).
        let qi = archives
            .iter()
            .position(|a| Arc::ptr_eq(a, arc))
            .expect("hit archive is from the library");
        let query_state = &mut queries[qi];
        let (preview, sections) = hit_preview(
            &article,
            &query_state.terms,
            &mut query_state.stemmer,
            query_state.ft_query.as_ref(),
            title_match,
            is_markdown,
        )?;
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

/// The `sections` selection from the regions' BM25 scores (`names` and
/// `scores` are parallel vectors in document order, `_intro` first): every
/// region scoring at least `min_frac` of the article's best region score is
/// reported, in document order. Regions matching nothing score 0 and never
/// qualify. A best score of 0 means no region matched any query word:
/// nothing is reported. (For any `min_frac <= 1` the best region itself
/// always clears the bar, so something is reported whenever any region
/// matched.) When the kept set covers most of the article's regions
/// (`max_coverage` of them, counting `_intro`), the whole article is
/// likely relevant and no list is worth reporting.
fn scored_sections(
    names: &[String],
    scores: &[f64],
    min_frac: f64,
    max_coverage: f64,
) -> Option<Vec<String>> {
    let best = scores.iter().cloned().fold(0.0, f64::max);
    if best <= 0.0 {
        return None;
    }
    let threshold = best * min_frac;
    let kept: Vec<String> = names
        .iter()
        .zip(scores)
        .filter(|(_, score)| **score >= threshold)
        .map(|(name, _)| name.clone())
        .collect();
    if kept.len() as f64 > max_coverage * names.len() as f64 {
        return None;
    }
    Some(kept)
}

/// A Xapian failure while scoring one hit's sections, with the step named.
fn section_error(e: xapian2::Error, what: &str) -> ToolError {
    ToolError::Internal(format!("failed to score sections ({what}): {}", e.msg()))
}

/// The padding term of a section-index document: it carries the wdf of the
/// region's words that are not query terms, so the document's length (the
/// wdf sum of its termlist) matches a full index of the region text. The
/// uppercase `Q` keeps it disjoint from every query stem, which is
/// lowercase (accents folded, see [`fold_accents`]).
const SECTION_PAD_TERM: &str = "Qpadding";

/// BM25-score every region of one article against `query` (the hit's own
/// full-text query): the regions go into a temporary **in-memory** Xapian
/// index, one document per region in document order (`_intro` first), the
/// query runs against that index through an `Enquire` once it is committed,
/// and the best score of the article decides the [`SECTION_MIN_SCORE_FRAC`]
/// cutoff. Returns one score per region in document order, 0.0 where the
/// region matched nothing.
///
/// Each region document receives exactly what the BM25 weighting consumes -
/// the query terms' within-region frequencies and the region's document
/// length (the termlist's wdf sum, padded through [`SECTION_PAD_TERM`]) -
/// because the words are already at hand in the scan below, while feeding
/// the whole region text through a TermGenerator was measured at 165 ms per
/// 100 KB here, an order of magnitude past the scan it replaces. The scan
/// folds and stems with the archive's own stemmer, so the terms are the
/// unprefixed stems the ZIM full-text index carries and the parsed query's
/// stems address; the scores are identical to a full index of the text.
fn region_scores(
    secs: &[(String, Vec<String>)],
    terms: &[String],
    stem: &mut Stemmer,
    query: &Query,
) -> Result<Vec<f64>, ToolError> {
    let mut wdb = WritableDatabase::in_memory()
        .map_err(|e| section_error(e, "create the in-memory section index"))?;
    for (name, paras) in secs {
        let mut doc = Document::new().map_err(|e| section_error(e, "create a document"))?;
        doc.set_data(name.as_str())
            .map_err(|e| section_error(e, "set a document's data"))?;
        // One scan over the region's words: every occurrence counts toward
        // the document length, and an occurrence of a query term adds to
        // that term's within-document frequency.
        let mut total = 0u32;
        let mut wdfs = vec![0u32; terms.len()];
        for para in paras {
            for word in para.split(|c: char| !c.is_alphanumeric()) {
                if word.is_empty() {
                    continue;
                }
                total += 1;
                let stemmed = stem.stem(word);
                if let Some(i) = terms.iter().position(|t| t == stemmed) {
                    wdfs[i] += 1;
                }
            }
        }
        let mut matched = 0u32;
        for (term, wdf) in terms.iter().zip(&wdfs) {
            if *wdf > 0 {
                doc.add_term(term, *wdf)
                    .map_err(|e| section_error(e, "add a term to a section document"))?;
                matched += wdf;
            }
        }
        if total > matched {
            doc.add_term(SECTION_PAD_TERM, total - matched)
                .map_err(|e| section_error(e, "pad a section document's length"))?;
        }
        wdb.add_document(&doc)
            .map_err(|e| section_error(e, "add a region to the section index"))?;
    }
    // commit() publishes the documents to search: without it the Enquire
    // below would see an empty index.
    wdb.commit().map_err(|e| section_error(e, "commit the section index"))?;
    let mut enquire =
        Enquire::new_writable(&wdb).map_err(|e| section_error(e, "create the enquire"))?;
    enquire
        .set_query(query)
        .map_err(|e| section_error(e, "set the section query"))?;
    // One score per region: `add_document` assigned docids 1.. sequentially
    // in document order, so a match's docid maps straight back to its
    // region; regions absent from the MSet matched nothing and stay at 0.
    let mset = enquire
        .get_mset(0, secs.len() as u32, 0)
        .map_err(|e| section_error(e, "run the section query"))?;
    let mut scores = vec![0.0; secs.len()];
    for i in 0..mset.size() {
        let docid = mset.docid(i) as usize;
        if (1..=secs.len()).contains(&docid) {
            scores[docid - 1] = mset.weight(i);
        }
    }
    Ok(scores)
}

/// Split a paragraph into sentences at `.`, `!`, `?` followed by whitespace
/// or end of paragraph ("U.S." over-splits, acceptable for a preview).
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

/// Truncate a preview to at most [`INTRO_CHARS`] characters, ending on a
/// complete word: the text up to the cap, backed up to the last whitespace
/// boundary so the trailing partial word (and any trailing whitespace) is
/// dropped - a preview cut mid-word reads as broken. Text within the cap
/// passes through unchanged; a single word longer than the cap (no
/// whitespace inside it) falls back to the hard cut at the cap.
fn preview_truncate(s: &str) -> String {
    if s.chars().count() <= INTRO_CHARS {
        return s.to_string();
    }
    let head: String = s.chars().take(INTRO_CHARS).collect();
    match head.rfind(char::is_whitespace) {
        // Byte index of the last whitespace in `head`: a char boundary, so
        // the slice is safe, and `trim_end` drops a run of whitespace.
        Some(bound) => head[..bound].trim_end().to_string(),
        None => head,
    }
}

/// The `preview`/`sections` pair of one search hit, from the article's raw
/// text (`is_markdown` picks the Markdown or the HTML splitter). The
/// preview is the article's lead - the first intro paragraph, truncated at
/// the last word boundary at or before `INTRO_CHARS` ([`preview_truncate`])
/// - no matter where in the article the query matched; a title match keeps
/// the intro's first sentence, truncated the same way. The `sections` of
/// a full-text hit are BM25-scored: the article's regions (`_intro` first,
/// then one region per heading in document order) go into a temporary
/// in-memory Xapian index ([`region_scores`]) that the hit's own full-text
/// query is run against, and the regions scoring at least
/// [`SECTION_MIN_SCORE_FRAC`] of the article's best region are reported in
/// document order - the intro and body sections compete on equal BM25
/// terms, so a region that merely grazes a common query word drops out
/// while the region carrying the query's substance wins. When more than
/// [`SECTION_MAX_COVERAGE`] of the regions qualify, the whole article is
/// likely relevant and no list is reported. No region matching any query
/// word reports no sections; empty regions give an empty preview, never a
/// panic.
fn hit_preview(
    article: &str,
    terms: &[String],
    stem: &mut Stemmer,
    ft_query: Option<&Query>,
    title_match: bool,
    is_markdown: bool,
) -> Result<(String, Option<Vec<String>>), ToolError> {
    let intro = if is_markdown {
        markdown::intro_paragraphs(article)
    } else {
        html::intro_paragraphs(article)
    };
    let lead = || {
        intro.first()
            .map(|p| preview_truncate(p))
            .unwrap_or_default()
    };
    if title_match {
        let first = intro
            .first()
            .and_then(|p| {
                sentences(p)
                    .first()
                    .map(|s| preview_truncate(s))
            })
            .unwrap_or_default();
        return Ok((first, None));
    }
    // The preview is the lead regardless of where the query matched: the
    // old best-matching-sentence search (re-scoring every sentence of
    // every matched paragraph) is gone. The region index below feeds only
    // the `sections` reporting.
    let Some(query) = ft_query else {
        return Ok((lead(), None));
    };
    let secs = if is_markdown {
        markdown::sections(article)
    } else {
        html::sections(article)
    };
    if secs.is_empty() {
        return Ok((lead(), None));
    }
    let scores = region_scores(&secs, terms, stem, query)?;
    let names: Vec<String> = secs.iter().map(|(name, _)| name.clone()).collect();
    Ok((lead(), scored_sections(&names, &scores, SECTION_MIN_SCORE_FRAC, SECTION_MAX_COVERAGE)))
}

// ---------------------------------------------------------------------------
// Tests: end-to-end over a synthetic archive carrying a real Xapian index
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tools::{
        ZimGetSectionDirParams, ZimGetSectionDirTool, ZimMcpServer, ZimSearchParams, ZimSearchTool,
    };
    use crate::zim::testutil::{
        build_archive, build_archive_indexes, language_metadata_entry, TestEntry, TestRedirect,
    };
    use rmcp::handler::server::router::tool::AsyncTool;
    use std::future::Future;
    use xapian2::{Document, Stem, WritableDatabase};

    /// Await an async tool invocation (each hops to a blocking thread) from
    /// a sync `#[test]` on a tiny current-thread runtime.
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
        <h2>Cultivation</h2>\
        <p>Orchards grow the fruit in temperate climates.</p>\
        <h2>See also</h2>\
        <p>Other rosaceae genera are described elsewhere.</p>\
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
    /// query's words and whose body matches too (so the cross-tier dedupe
    /// has something to do), plus one body-only article.
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

    /// Articles whose query matches a mid-paragraph sentence of a body
    /// region: the preview is still the intro start, while the matched
    /// region is reported as `sections`. The Volcano lead does not cover
    /// its query, so the lead fast path does not fire. Filler sections in
    /// both fixtures keep the matching region a minority under the
    /// coverage cap.
    const VOLCANO_HTML: &str = "<html><body><h1>Volcano</h1>\
        <p>Volcanoes are openings in the crust.</p>\
        <p>Molten rock rises from chambers below. Eruptions reshape the \
        land. Ash clouds can ground aircraft. Farmers fear the fallout.</p>\
        <h2>Formation</h2>\
        <p>Magma accumulates in underground chambers.</p>\
        <h2>Hazards</h2>\
        <p>Eruptions endanger nearby settlements.</p>\
        </body></html>";

    const GLACIER_MD: &str = "\
# Glacier

A glacier is a body of dense ice.

## Movement

Glaciers move under their own weight. The flow is slower than a river. \
Meltwater streams out of the ice.

## Mass balance

Snowfall accumulates faster than ablation in the upper reaches.

## Study

Scientists measure the flow from observatories.
";

    /// Run a search through the tool's server: the whole pipeline behind
    /// `zim_search`, hits in rank order. The tool layer itself
    /// (`ZimSearchTool`) is exercised by the empty-query validation test
    /// below.
    fn search(server: &ZimMcpServer, query: &str) -> Vec<SearchHit> {
        super::search(&server.library, None, query).unwrap().results
    }

    /// [`search`] with the optional single-archive filter filled in.
    fn search_one(server: &ZimMcpServer, only: &str, query: &str) -> Vec<SearchHit> {
        super::search(&server.library, Some(only), query).unwrap().results
    }

    /// Build a single-file glass Xapian index the way openZIM does: document
    /// data = the article's full path, title in value slot 0, unprefixed
    /// Porter2 stem terms as libzim indexes a full-text index (STEM_ALL).
    /// The terms are passed in already stemmed. Real title indexes store
    /// surface word forms instead - title fixtures pass surfaces.
    fn make_index(docs: &[(&str, &str, &str)]) -> Vec<u8> {
        make_index_stemmed(None, docs)
    }

    /// [`make_index`], parameterized by the index language: `Some(code)`
    /// folds and stems the given surface words with that language's stemmer,
    /// the way libzim indexes an archive's content (removeAccents, then the
    /// stemmer). `None` adds the terms verbatim.
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

    /// The bytes of the shared test archive: Apple and Banana with a real
    /// full-text index, one ZIM file named `test.zim`.
    fn test_zim_bytes() -> Vec<u8> {
        let index = make_index(
            &[
                ("C/Apple", "appl histori 10 000 year domest wild kazakhstan comput devic nam", "Apple"),
                ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
            ],
        );
        let content = [
            TestEntry { namespace: b'C', url: "Apple", title: "Apple", mime: 0, body: APPLE_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
        ];
        build_archive(&["text/html"], &content, &[], 0, Some(&index))
    }

    pub(crate) fn test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.zim"), test_zim_bytes()).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    /// The same fixture opened as a single-file library: the server boots
    /// in single mode, whose tools take no `zim` argument.
    pub(crate) fn test_single_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.zim");
        std::fs::write(&file, test_zim_bytes()).unwrap();
        let library = Arc::new(ZimLibrary::single(&file).unwrap());
        assert_eq!(library.mode, crate::zim::Mode::Single);
        assert!(library.archives[0].searchable());
        (ZimMcpServer::new(library), dir)
    }

    #[test]
    fn stem_language_resolution() {
        // Xapian rejects ISO-639-3 codes ("Language code fra unknown") and
        // only knows English names plus two-letter ISO-639-1 codes: the
        // fallback maps a 639-3 code to its two-letter prefix.
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
    /// A Language=fra archive whose index was built with French stems: the
    /// inflected query form "élections" matches only after FRENCH stemming.
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
        // An accented inflected form matches only through the FOLD: "écoles"
        // folds to "ecoles" and stems to "ecol", the index term of "école".
        let hits = search(&server, "écoles");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/École");
        assert_eq!(hits[0].title, "École");
        // Nothing matches a term absent from the index.
        let hits = search(&server, "zzzzz");
        assert!(hits.is_empty());
    }

    /// An e2e proof that the fulltext tier's parsed query is built from the
    /// accent-FOLDED words: the synthetic index carries only folded French
    /// stems (the shape measured on fr.zim's full-text index), so an
    /// unfolded parse of "révolution" would match nothing.
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

        // Not an exact title/URL hit and the archive has no title index, so
        // the ONLY route to the article is the full-text tier - which
        // matches only because the parsed query's words were folded first.
        let hits = search(&server, "révolution française");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Révolution");
        // A full-text hit whose lead covers every query term: the whole
        // two-sentence lead is the preview (a title match would be the
        // first sentence only).
        assert_eq!(
            hits[0].preview,
            "La Révolution française éclate en 1789. La monarchie est renversée et la république proclamée."
        );
        // BM25 over the section index scores the intro region against the
        // query (the lead's terms live there too), but the intro is the
        // article's only region: the kept set covers the whole article, so
        // the coverage cap reports no list.
        assert_eq!(hits[0].sections, None);
    }

    #[test]
    fn e2e_search() {
        let (server, _keep) = test_server();

        let hits = search(&server, "apple");
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
        let hits = search(&server, "computing");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "C/Apple");

        // Unrelated term: no hits.
        let hits = search(&server, "zzzzz");
        assert!(hits.is_empty());

        // OR semantics: two terms from different articles.
        let hits = search(&server, "banana apple");
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
        let index = make_index(
            &[
                ("C/Nitrogen", "nitrogen gas inert", "Nitrogen"),
                ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
            ],
        );
        let content = [
            // Empty directory-entry title, as in modern openZIM archives.
            TestEntry { namespace: b'C', url: "Nitrogen", title: "", mime: 0, body: NITROGEN_HTML.as_bytes() },
            TestEntry { namespace: b'C', url: "Banana", title: "Banana", mime: 0, body: BANANA_HTML.as_bytes() },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        let server = ZimMcpServer::new(library);

        // Not an exact title/URL match, so the hit comes from the full-text
        // index: with the directory title empty, the title falls back to
        // the index title (value slot 0).
        let hits = search(&server, "nitrogen gas");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Nitrogen", "{hits:?}");
    }

    /// An archive with BOTH embedded indexes: a full-text index over article
    /// bodies plus a title index whose documents carry the title's surface
    /// words as terms and the title in value slot 0.
    fn title_index_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(
            &[
                ("C/Nitrogen_Gas_Effects", "nitrogen gas surround effect unavoid", "Nitrogen Gas Effects"),
                ("C/Weather", "weather forecast describ effect air pressur gas law explan atmospher", "Weather"),
                ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
                ("C/Glacier", "glacier ice dens movem weight flow", "Glacier"),
            ],
        );
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

        // The article's title contains every query word - function word
        // included, like real unstopped title indexes - so its title-index
        // document IS the title tier and ranks ahead of the full-text-only
        // matches; the hit is styled as a title match.
        let hits = search(&server, "effects of nitrogen gas");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].path, "C/Nitrogen_Gas_Effects", "{hits:?}");
        assert_eq!(hits[0].title, "Effects of Nitrogen Gas");
        // Title-match semantics: the lead's FIRST sentence and no sections -
        // NOT the Everywhere section the full-text tier would have reported.
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
        // Weather's only region is its intro and it matches the query, so
        // the kept set covers the whole article and the coverage cap
        // suppresses the list.
        assert_eq!(hits[1].sections, None);

        // No usable terms: the title tier's AND over words absent from the
        // title index matches nothing and the full-text tier matches
        // nothing either (no such terms in this index) - no hits, no error.
        let hits = search(&server, "the of");
        assert!(hits.is_empty(), "{hits:?}");
    }

    /// An archive whose titles do NOT contain every query word: a partial
    /// title match must not be promoted at all - that judgment belongs to
    /// the full-text tier's BM25.
    fn title_and_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let titles = make_index(&[
            ("C/New_York_City", "new york city", "New York City"),
            ("C/Quebec_City", "quebec city", "Quebec City"),
            ("C/Kansas_City", "kansas city", "Kansas City"),
            ("C/Mexico_City", "mexico city", "Mexico City"),
            ("C/Weather", "weather", "Weather"),
        ]);
        let index = make_index(
            &[
                ("C/New_York_City", "new york citi largest unit state", "New York City"),
                ("C/Quebec_City", "quebec citi capit provinc", "Quebec City"),
                // "kansa", not "kansas": the index terms are the STEMS the
                // query parser produces (Porter2 strips the s), as libzim
                // indexes them.
                ("C/Kansas_City", "kansa citi straddl state", "Kansas City"),
                ("C/Mexico_City", "mexico citi capit", "Mexico City"),
                ("C/Weather", "weather forecast effect atmospher", "Weather"),
            ],
        );
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

        // The article's title contains both query words: the title tier's
        // AND surface match ranks it first, styled as a title match (lead's
        // first sentence only, not a full-text preview).
        let hits = search(&server, "new york");
        assert!(!hits.is_empty(), "{hits:?}");
        assert_eq!(hits[0].path, "C/New_York_City", "{hits:?}");
        assert_eq!(hits[0].title, "New York City");
        assert_eq!(hits[0].sections, None);
        assert_eq!(hits[0].preview, "New York City is the largest city in the United States.");

        // A partial title match is NOT promoted to the title tier: no title
        // contains all three query words, so every hit comes from the
        // full-text tier with full-text semantics (the full lead as the
        // preview). Each article's only region is its intro and it matches,
        // so the coverage cap suppresses the section list.
        let hits = search(&server, "kansas city new");
        assert!(hits.len() >= 2, "{hits:?}");
        for hit in &hits {
            assert!(hit.sections.is_none(), "{hit:?}");
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

    /// The title index stores titles as written - lowercased surface word
    /// forms (measured on md1m: df("beatles")=186 against df("beatl")=0) -
    /// so the tier matches surface forms only, no stem variants. Titles sit
    /// on URLs the queries cannot hit exactly, so every reported hit really
    /// comes from the title tier.
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
        let index = make_index(
            &[
                ("C/Physics1", "black hole graviti spacetime", "Black hole"),
                ("C/Physics2", "black hole graviti spacetime", "Black holes"),
                ("C/Movie1", "hole plot movi", "Holes"),
                ("C/Movie2", "movi film", "The Movie"),
                ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
                ("C/Glacier", "glacier ice dens movem weight flow", "Glacier"),
            ],
        );
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
        // matches only the "Black holes" title (no surface "holes" in the
        // "Black hole" title, no stem variants). The body-matching articles
        // follow through the full-text tier.
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

    /// A title index carrying a punctuation-delimited title exactly the way
    /// libzim's indexer stores titles: "Cathédrale Notre-Dame de Paris" is
    /// indexed as the folded surface words cathedrale, notre, dame, de,
    /// paris - the indexer splits on non-alphanumeric characters, so no
    /// fused "notredame" term exists (measured on fr.zim: notre df=164,
    /// dame df=191, notredame absent). The full-text document matches none
    /// of the query's words, so the title band is the ONLY tier that can
    /// retrieve the article.
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

        // The title tier parses the punctuation-to-spaces normalized query,
        // so the words are punctuation-SPLIT and the parser's AND over them
        // finds the article (it pins the regression where a hand-built AND
        // fused "notre-dame" into the df-0 term "notredame" and the band
        // came back silently empty).
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

    /// An archive where BM25 alone ranks the wrong article first (the
    /// "Atmosphere" document repeats "nitrogen" seven times); "NACA" and
    /// "Usa" are redirects onto the Aeronautics article, and redirect
    /// entries live only in the directory, never in the search index.
    fn exact_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        // Index terms are the stems the query parser produces ("atmosphere"
        // -> "atmospher"), unprefixed, as libzim indexes with STEM_ALL.
        let index = make_index(
            &[
                ("C/Nitrogen", "nitrogen colorless odorless gas", "Nitrogen"),
                ("C/Atmosphere", "nitrogen nitrogen nitrogen nitrogen nitrogen nitrogen nitrogen naca naca atmospher", "Atmosphere"),
                ("C/Aeronautics", "aeronautics naca aviation wind tunnel flight", "Aeronautics"),
                ("C/Weather", "weather forecast rain snow climat", "Weather"),
            ],
        );
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
        // the Atmosphere document first: the exact match must come out on
        // top.
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

        // "NACA" is a redirect onto the Aeronautics article; redirects are
        // not in the full-text index, so without the directory lookup the
        // query would report Atmosphere first. The exact tier follows the
        // redirect chain: the RESULT reports the TERMINAL article, and the
        // same article's full-text hit dedupes into it.
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

    /// A title index containing a redirect document, as old-namespace
    /// openZIM archives do: one query can reach the same article three
    /// ways - exact probe (directory redirect), title tier (redirect
    /// document), and full text (the target article).
    fn redirect_dedupe_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(
            &[
                ("C/Aeronautics", "aeronautics naca aviation flight", "Aeronautics"),
            ],
        );
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

        // All three tiers reach the Aeronautics article; it must be reported
        // exactly once, at its highest rank (the exact tier's), under the
        // terminal article's identity.
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

        // Nobody's title or URL, so the ranking is the unchanged BM25
        // order: the document with both terms first.
        let hits = search(&server, "nitrogen atmosphere");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].path, "C/Atmosphere", "{hits:?}");
        assert_eq!(hits[1].path, "C/Nitrogen", "{hits:?}");
        // The Atmosphere lead covers the whole query ("The atmosphere is
        // mostly nitrogen and oxygen.") in its first paragraph, and the
        // Nitrogen lead covers "nitrogen": BM25 over the section index
        // scores the intro region of each - but each article's ONLY region
        // is its intro, so the kept set covers the whole article and the
        // coverage cap suppresses the list for both hits.
        assert_eq!(hits[0].sections, None, "{:?}", hits[0]);
        assert_eq!(hits[0].preview, "The atmosphere is mostly nitrogen and oxygen.");
        assert_eq!(hits[1].sections, None, "{:?}", hits[1]);
        assert_eq!(hits[1].preview, "Nitrogen is a colorless, odorless gas.");
    }
    #[test]
    fn e2e_search_section_match_reports_sections() {
        // The query term appears only in a later section of the article
        // ("Wild apples grew in Kazakhstan." under History): the intro
        // cannot cover it, so the hit reports the matched sections while
        // the preview stays the article's lead.
        let (server, _keep) = test_server();
        let hits = search(&server, "kazakhstan");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Apple");
        // The paragraph sits under History, whose range includes the nested
        // Domestication heading: both sections report the match. (The filler
        // sections in APPLE_HTML keep the article at six regions, so the two
        // matching ones stay a minority under the coverage cap.)
        assert_eq!(
            hit.sections,
            Some(vec!["History".to_string(), "Domestication".to_string()]),
            "{hit:?}"
        );
        assert_eq!(hit.preview, "An apple is the fruit of <rosaceae> trees.");
        // The serialized JSON carries the section names.
        let json = serde_json::to_string(hit).unwrap();
        assert!(json.contains(r#""sections":["History","Domestication"]"#), "{json}");
    }

    /// An archive whose only article (`SALT_HTML`) has a two-paragraph
    /// intro, for the intro-matching search semantics.
    fn intro_test_server() -> (ZimMcpServer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(
            &[
                (
                    "C/Salt",
                    "salt mineral chlorid sodium himalaya deposit rock bed form sea season food",
                    "Salt",
                ),
                ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
                ("C/Glacier", "glacier ice dens movem weight flow", "Glacier"),
            ],
        );
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
        // "himalaya" matches only the intro's second paragraph: the intro
        // is reported as the matching region _intro - the preview is
        // unchanged (the lead).
        let (server, _keep) = intro_test_server();
        let hits = search(&server, "himalaya");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Salt");
        assert_eq!(hit.sections, Some(vec!["_intro".to_string()]), "{hit:?}");
        assert_eq!(
            hit.preview,
            "Salt is a mineral composed primarily of sodium chloride."
        );
        let json = serde_json::to_string(hit).unwrap();
        assert!(json.contains(r#""sections":["_intro"]"#), "{json}");
    }

    #[test]
    fn e2e_search_on_target_section_outranks_the_intro_mention() {
        // "salt beds" matches the intro ("salt" twice, no "beds") and the
        // Formation section ("Salt beds ...", both words): BM25 over the
        // section index scores Formation - the region carrying both query
        // words - far above the intro's grazing "salt" mentions, and only
        // Formation clears the relative bar. The preview stays the lead.
        let (server, _keep) = intro_test_server();
        let hits = search(&server, "salt beds");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Salt");
        assert_eq!(hit.sections, Some(vec!["Formation".to_string()]), "{hit:?}");
        assert_eq!(
            hit.preview,
            "Salt is a mineral composed primarily of sodium chloride."
        );
    }

    /// A query whose common word ("salt") appears in every region but whose
    /// substance ("salt mining") lives in one section: the old stem-count
    /// scan reported every region holding any query word - the grazing
    /// "See also" section included - while BM25 over the section index
    /// scores the on-target region far above a lone common-word occurrence
    /// and the grazing region drops out.
    #[test]
    fn e2e_search_bm25_drops_grazing_junk_sections() {
        let dir = tempfile::tempdir().unwrap();
        // The article's URL must not be reachable by the exact title/URL
        // probe ("salt mining" -> "Salt_Mining" would be an exact match,
        // styled as a title match with no sections), so the article carries
        // a longer name.
        let index = make_index(&[(
            "C/Salt_Mining_Industry",
            "salt mine anci industri extract rock deposit oper worldwid shape trade rout \
             centuri shaker kitchen tool",
            "Salt mining industry",
        )]);
        let html: &'static [u8] = "<html><body><h1>Salt mining industry</h1>\
            <p>Salt mining is an ancient industry.</p>\
            <h2>Overview</h2>\
            <p>Salt mining extracts rock salt from salt deposits. Salt mines \
            operate worldwide, and mining salt shaped trade routes for \
            centuries.</p>\
            <h2>See also</h2>\
            <p>Salt shakers are kitchen tools.</p>\
            <h2>Geography</h2>\
            <p>The province lies between two rivers, and its harbors trade \
            in fish.</p>\
            <h2>Demographics</h2>\
            <p>Most of the population lives in coastal towns.</p>\
            <h2>References</h2>\
            <p>Printed surveys of the region appear every decade.</p>\
            </body></html>".as_bytes();
        let content = [TestEntry {
            namespace: b'C',
            url: "Salt_Mining_Industry",
            title: "Salt mining industry",
            mime: 0,
            body: html,
        }];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        let server = ZimMcpServer::new(library);

        let hits = search(&server, "salt mining");
        assert_eq!(hits.len(), 1, "{hits:?}");
        let hit = &hits[0];
        assert_eq!(hit.path, "C/Salt_Mining_Industry");
        // The intro (both words) and the on-target Overview region are
        // reported, in document order; "See also" - one lone "salt" - does
        // not clear the bar. The three filler sections carry no query word
        // and keep the article at six regions, so the two on-target ones
        // stay a minority under the coverage cap.
        assert_eq!(
            hit.sections,
            Some(vec!["_intro".to_string(), "Overview".to_string()]),
            "{hit:?}"
        );
        let json = serde_json::to_string(hit).unwrap();
        assert!(json.contains(r#""sections":["_intro","Overview"]"#), "{json}");
        assert!(!json.contains("See also"), "{json}");
    }

    #[test]
    fn scored_sections_threshold() {
        let names = [
            "_intro".to_string(),
            "History".to_string(),
            "Trivia".to_string(),
            "Geography".to_string(),
            "See also".to_string(),
        ];
        // The on-topic regions lead; the grazing region sits below the 0.4
        // bar and is dropped, while the reported order stays document order
        // (not score order). Two of the five regions clear the bar - exactly
        // the 0.4 coverage cap, so the list is still reported.
        let scores = [0.674, 0.837, 0.153, 0.0, 0.1];
        assert_eq!(
            scored_sections(&names, &scores, SECTION_MIN_SCORE_FRAC, SECTION_MAX_COVERAGE),
            Some(vec!["_intro".to_string(), "History".to_string()])
        );
        // The same shape over three regions: two of three clear the bar -
        // 0.667 coverage, past the cap - so the article reads as wholly
        // relevant and no list is reported.
        let names3 = ["_intro".to_string(), "History".to_string(), "Trivia".to_string()];
        assert_eq!(
            scored_sections(
                &names3,
                &[0.674, 0.837, 0.153],
                SECTION_MIN_SCORE_FRAC,
                SECTION_MAX_COVERAGE
            ),
            None
        );
        // No region matched any query term: no sections.
        assert_eq!(
            scored_sections(
                &names,
                &[0.0, 0.0, 0.0, 0.0, 0.0],
                SECTION_MIN_SCORE_FRAC,
                SECTION_MAX_COVERAGE
            ),
            None
        );
        assert_eq!(
            scored_sections(&names, &[], SECTION_MIN_SCORE_FRAC, SECTION_MAX_COVERAGE),
            None
        );
        // min_frac > 1 could empty the set, but the constant is 0.4; the
        // best region always clears any bar at or below 1.0.
        assert_eq!(
            scored_sections(&names, &[0.1, 0.9, 0.2, 0.3, 0.1], 1.0, SECTION_MAX_COVERAGE),
            Some(vec!["History".to_string()])
        );
        // Too many regions qualify: the whole article is relevant, so no
        // list is worth reporting (here 3 of 5 clear the bar - 0.6 coverage,
        // past the cap).
        assert_eq!(
            scored_sections(
                &names,
                &[0.9, 0.8, 0.7, 0.1, 0.0],
                SECTION_MIN_SCORE_FRAC,
                SECTION_MAX_COVERAGE
            ),
            None
        );
    }

    #[test]
    fn preview_truncation_ends_on_a_complete_word() {
        // Within the cap the text passes through unchanged.
        let short = "A cherry is the fruit of trees of the genus Prunus.";
        assert_eq!(preview_truncate(short), short);

        // A lead whose cap falls INSIDE a word: the preview ends at the
        // last complete word before INTRO_CHARS, not mid-word. 29 repeats
        // of a 12-character unit put the cap 12 characters into the 15-char
        // word that follows, so the fixed cut this replaces chopped it.
        let paragraph = format!("{}{}", "lorem ipsum ".repeat(29), "wordthatislong! now more text");
        assert!(paragraph.chars().count() > INTRO_CHARS);
        let hard_cut: String = paragraph.chars().take(INTRO_CHARS).collect();
        // The word straddles the cap: the old fixed cut ended mid-word.
        assert!(!hard_cut.ends_with(char::is_whitespace));
        let preview = preview_truncate(&paragraph);
        assert!(preview.chars().count() <= INTRO_CHARS);
        assert!(!preview.ends_with(char::is_whitespace), "{preview:?}");
        assert!(paragraph.starts_with(&preview), "{preview:?}");
        assert!(preview.ends_with("ipsum"), "{preview:?}");
        // The cut landed on the whitespace before the straddling word.
        assert!(
            paragraph[preview.len()..].starts_with(char::is_whitespace),
            "{preview:?}"
        );

        // One giant word with no whitespace within the cap: the hard cut.
        let giant = "x".repeat(INTRO_CHARS + 40);
        assert_eq!(preview_truncate(&giant), "x".repeat(INTRO_CHARS));

        // The same paragraph as a fulltext hit's lead: the preview is the
        // lead truncated at the word boundary, and only the lead (no
        // sections) is reported without a full-text query.
        let mut stem = Stemmer::new("en").unwrap();
        let article = format!("<html><body><h1>T</h1><p>{paragraph}</p></body></html>");
        let (lead_preview, sections) =
            hit_preview(&article, &[], &mut stem, None, false, false).unwrap();
        assert_eq!(sections, None);
        assert_eq!(lead_preview, preview_truncate(&paragraph));
    }

    #[test]
    fn e2e_search_section_match_still_reports_sections() {
        // The query matches a mid-paragraph sentence of a body region: the
        // preview is the article's lead (the intro start), NOT the matched
        // sentence - but the matched region is still reported. Covered for
        // an HTML article (intro match) and a Markdown one (body-section
        // match).
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(
            &[
                ("C/Volcano", "volcano crust molten rock erupt reshape land ash cloud aircraft farmer", "Volcano"),
                ("C/Glacier", "glacier ice movement weight flow river meltwater stream", "Glacier"),
            ],
        );
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
        assert_eq!(hits[0].preview, "Volcanoes are openings in the crust.");

        let hits = search(&server, "river");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].path, "C/Glacier");
        assert_eq!(hits[0].sections, Some(vec!["Movement".to_string()]));
        assert_eq!(hits[0].preview, "A glacier is a body of dense ice.");
    }

    #[test]
    fn e2e_search_all_words_doc_wins_under_plain_or() {
        // Plain BM25 (OR over the query terms) must rank the document that
        // mentions EACH query term above the ones repeating a single term:
        // BM25 saturates term frequency and rewards the second, rare term.
        let dir = tempfile::tempdir().unwrap();
        // The repeated "filler" term pads document length (BM25 length
        // normalization); it is no query word.
        let cherry_terms = format!("{}{}", "cherri ".repeat(30), "filler ".repeat(400));
        let pie_terms = format!("{}{}", "pie ".repeat(30), "filler ".repeat(400));
        let index = make_index(
            &[
                ("C/Cherry", cherry_terms.as_str(), "Cherry"),
                ("C/Dessert_Recipes", "cherri cherri pie pie", "Dessert Recipes"),
                ("C/Pie_1", pie_terms.as_str(), "Pie 1"),
                ("C/Pie_2", pie_terms.as_str(), "Pie 2"),
                ("C/Mango", "mango tropic tree sweet", "Mango"),
                ("C/Peach", "peach orchard stone fruit", "Peach"),
            ],
        );
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
        // Neither hit's lead covers both terms; the intro matches "cherry"
        // but is the article's only region, so the kept set covers the
        // whole article and the coverage cap suppresses the list.
        assert_eq!(hits[0].sections, None);
        assert!(hits[0].preview.contains("cherry is the fruit"), "{:?}", hits[0].preview);
    }    /// Two archives (a.zim, b.zim) that both carry an "Apple" article (the
    /// same article, as in an HTML and a Markdown edition of the same ZIM),
    /// plus one exclusive article each: a.zim holds C/Banana, b.zim holds
    /// C/Cherry. Returns the tempdir (it must outlive the searches) and the
    /// scanned library.
    pub(crate) fn two_archive_library() -> (tempfile::TempDir, Arc<ZimLibrary>) {
        let dir = tempfile::tempdir().unwrap();
        let index_a = make_index(
            &[
                ("C/Apple", "appl histori 10 000 year domest wild kazakhstan comput devic nam", "Apple"),
                ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
            ],
        );
        let index_b = make_index(
            &[
                ("C/Apple", "appl comput devic nam", "Apple"),
                ("C/Cherry", "cherri pie fruit tree", "Cherry"),
            ],
        );
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
        (dir, library)
    }

    #[test]
    fn e2e_search_pools_fulltext_across_archives_and_dedupes_within_archives() {
        let (_dir, library) = two_archive_library();
        let server = ZimMcpServer::new(library);

        // The same title in both archives is NOT deduped across archives:
        // different archives may hold different articles under one title,
        // so each archive's copy is reported (the exact tier's library
        // order first).
        let hits = search(&server, "apple");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("a.zim", "C/Apple"));
        assert_eq!((hits[1].zim.as_str(), hits[1].path.as_str()), ("b.zim", "C/Apple"));

        // Distinct matches rank by the pooled re-score, not by archive
        // rank rotation: re-scored with one BM25 over the pooled
        // statistics, b.zim's Cherry (its only term sits in a shorter
        // document) outscores a.zim's Banana - the rotation this replaces
        // reported the archives' best matches in library order.
        let hits = search(&server, "banana cherry");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("b.zim", "C/Cherry"));
        assert_eq!((hits[1].zim.as_str(), hits[1].path.as_str()), ("a.zim", "C/Banana"));
    }

    /// Two archives whose "cherry" hits differ in strength: a.zim (first in
    /// library order) holds a weak match - the term once in a long document
    /// - and b.zim a strong one - four occurrences in a short document.
    /// Returns the tempdir (it must outlive the searches) and the scanned
    /// library.
    fn pooled_rerank_library() -> (tempfile::TempDir, Arc<ZimLibrary>) {
        let dir = tempfile::tempdir().unwrap();
        let weak_terms = format!("cherri {}", "filler ".repeat(60));
        let strong_terms = format!("{}pad pad", "cherri ".repeat(4));
        let unrelated = "unrelated words entirely";
        let index_a = make_index(&[
            ("C/Long_Doc", weak_terms.as_str(), "Long doc"),
            ("C/A_Other", unrelated, "Other"),
        ]);
        let index_b = make_index(&[
            ("C/Short_Doc", strong_terms.as_str(), "Short doc"),
            ("C/B_Other", unrelated, "More"),
        ]);
        let entry = |url: &'static str, title: &'static str| TestEntry {
            namespace: b'C',
            url,
            title,
            mime: 0,
            body: b"<html><body><h1>Filler.</h1><p>Filler text.</p></body></html>",
        };
        let content_a = [entry("Long_Doc", "Long doc"), entry("A_Other", "Other")];
        let content_b = [entry("Short_Doc", "Short doc"), entry("B_Other", "More")];
        std::fs::write(
            dir.path().join("a.zim"),
            build_archive(&["text/html"], &content_a, &[], 0, Some(&index_a)),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("b.zim"),
            build_archive(&["text/html"], &content_b, &[], 0, Some(&index_b)),
        )
        .unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert_eq!(library.archives.len(), 2);
        (dir, library)
    }

    #[test]
    fn e2e_search_pooled_rerank_ranks_the_strong_match_first() {
        let (_dir, library) = pooled_rerank_library();
        let server = ZimMcpServer::new(library);

        // Rank rotation interleaved by archive rank and would have reported
        // the weak a.zim match first (a.zim is scanned first and each
        // archive holds exactly one hit). The pooled re-score ranks both
        // candidates with one BM25 over the pooled statistics: b.zim's
        // short document carries the term four times, a.zim's long one
        // once, and the strong match must come first.
        let hits = search(&server, "cherry");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("b.zim", "C/Short_Doc"));
        assert_eq!((hits[1].zim.as_str(), hits[1].path.as_str()), ("a.zim", "C/Long_Doc"));
    }

    #[test]
    fn e2e_search_fulltext_rescore_matches_xapian_weights() {
        // The full-text tier's pooled re-ranking re-scores every candidate
        // outside Xapian. On a single archive the pooled statistics ARE the
        // archive's statistics, so each recomputed score must equal the
        // MSet weight Xapian itself reported for the same query - the FFI
        // accessors, the formula, and the wqf derivation (a repeated query
        // word, a term indexing several documents through the tw<2
        // softening branch, and a rare term through the unsoftened one)
        // validated end to end over a real parsed query. The fixture's
        // index terms are the stems its query words stem to.
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(&[
            ("C/One", "appl appl banana kazakhstan", "One"),
            ("C/Two", "appl banana cherri", "Two"),
            ("C/Three", "appl pie", "Three"),
            ("C/Four", "banana glacier", "Four"),
        ]);
        let content = [
            TestEntry { namespace: b'C', url: "One", title: "One", mime: 0, body: b"<html><body><h1>One</h1><p>One text.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Two", title: "Two", mime: 0, body: b"<html><body><h1>Two</h1><p>Two text.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Three", title: "Three", mime: 0, body: b"<html><body><h1>Three</h1><p>Three text.</p></body></html>" },
            TestEntry { namespace: b'C', url: "Four", title: "Four", mime: 0, body: b"<html><body><h1>Four</h1><p>Four text.</p></body></html>" },
        ];
        let bytes = build_archive(&["text/html"], &content, &[], 0, Some(&index));
        std::fs::write(dir.path().join("test.zim"), &bytes).unwrap();
        let library = Arc::new(ZimLibrary::scan(dir.path()).unwrap());
        assert!(library.archives[0].searchable());
        let server = ZimMcpServer::new(library);

        let arc = &server.library.archives[0];
        let query_state =
            ArchiveQuery::build(arc, &fold_accents("apple apple cherry kazakhstan")).unwrap();
        arc.with_xapian(|h| -> Result<(), ToolError> {
            let ft = fetch_fulltext(h, &query_state)?;
            // The pooled == local statistics: 4 documents of lengths
            // 4/3/2/2, and the terms' document frequencies.
            assert_eq!(ft.docs, 4);
            assert!((ft.avlen - 2.75).abs() < 1e-9);
            assert_eq!(
                ft.terms,
                [("appl".to_string(), 2), ("cherri".to_string(), 1), ("kazakhstan".to_string(), 1)]
            );
            assert_eq!(ft.dfs, [3, 1, 1]);
            let paths: Vec<&str> = ft.candidates.iter().map(|c| c.path.as_str()).collect();
            assert_eq!(paths, ["C/One", "C/Two", "C/Three"]);

            // Xapian's own weights for the same query and cap, keyed by the
            // candidates' paths (unique per archive).
            let mut enquire = Enquire::new(&h.fulltext)?;
            enquire.set_query(query_state.ft_query.as_ref().unwrap())?;
            enquire.set_sort_by_relevance();
            let mset = enquire.get_mset(0, FT_FETCH_PER_ARCHIVE, 0)?;
            let mut weight_of = std::collections::HashMap::new();
            for i in 0..mset.size() {
                weight_of.insert(mset.document(i)?.data_str()?, mset.weight(i));
            }
            assert_eq!(weight_of.len(), 3);

            // The re-score, the merge's own scoring path (pooled == local).
            let payload = (arc, ft);
            let pool = FtPool::new(std::slice::from_ref(&payload));
            for cand in &payload.1.candidates {
                let score = pool.score(&payload.1, cand);
                let expected = weight_of[&cand.path];
                assert!(
                    (score - expected).abs() <= 1e-9 * expected.abs().max(1e-12),
                    "{}: recomputed {score} against Xapian {expected}",
                    cand.path
                );
            }
            Ok(())
        })
        .unwrap()
        .unwrap();
    }

    /// Real-index verification, run explicitly: the synthetic fixtures are
    /// built through xapian2's own `WritableDatabase`, but real ZIM indexes
    /// are glass databases written by libzim with `DB_NO_TERMLIST` (termlist
    /// access throws on every document there), so the pooled re-score's
    /// accessors and scores are verified against one. Run with
    /// `SZMCP_REAL_ZIM=<zim file> cargo test --bin szmcp real_zim -- --ignored`
    /// (a symlinked file is fine); skipped without the variable.
    #[test]
    #[ignore]
    fn real_zim_fulltext_rescore_matches_xapian_weights() {
        let Ok(path) = std::env::var("SZMCP_REAL_ZIM") else {
            eprintln!("SZMCP_REAL_ZIM not set; skipping the real-ZIM verification");
            return;
        };
        let library = Arc::new(ZimLibrary::single(std::path::Path::new(&path)).unwrap());
        assert!(library.archives[0].searchable(), "{path}");
        let arc = &library.archives[0];
        // A repeated query word ("acid" twice) over common chemistry words:
        // both index many documents, so the pooled re-score sees df > 1.
        let query_state = ArchiveQuery::build(arc, &fold_accents("acid acid base")).unwrap();
        let started = std::time::Instant::now();
        arc.with_xapian(|h| -> Result<(), ToolError> {
            let ft = fetch_fulltext(h, &query_state)?;
            // The accessors on a real libzim glass database: sane lengths
            // and document frequencies (the DB_NO_TERMLIST shape must not
            // break doc lengths or wdfs).
            assert!(ft.docs > 0);
            assert!(ft.avlen > 0.0, "average_length {} on {path}", ft.avlen);
            assert!(ft.candidates.iter().all(|c| c.doclen > 0.0));
            assert!(ft.candidates.iter().any(|c| c.wdfs.iter().any(|w| *w > 0)));
            assert_eq!(
                ft.terms.iter().map(|(t, w)| (t.as_str(), *w)).collect::<Vec<_>>(),
                [("acid", 2), ("base", 1)]
            );
            assert!(ft.candidates.len() >= 2, "expected several acid/base hits");

            let mut enquire = Enquire::new(&h.fulltext)?;
            enquire.set_query(query_state.ft_query.as_ref().unwrap())?;
            enquire.set_sort_by_relevance();
            let mset = enquire.get_mset(0, FT_FETCH_PER_ARCHIVE, 0)?;
            let mut weight_of = std::collections::HashMap::new();
            for i in 0..mset.size() {
                weight_of.insert(mset.document(i)?.data_str()?, mset.weight(i));
            }
            assert_eq!(weight_of.len(), ft.candidates.len());

            let payload = (arc, ft);
            let pool = FtPool::new(std::slice::from_ref(&payload));
            let mut worst = 0.0f64;
            for cand in &payload.1.candidates {
                let score = pool.score(&payload.1, cand);
                let expected = weight_of[&cand.path];
                worst = worst.max((score - expected).abs() / expected.abs().max(1e-12));
                assert!(
                    (score - expected).abs() <= 1e-9 * expected.abs().max(1e-12),
                    "{}: recomputed {score} against Xapian {expected}",
                    cand.path
                );
            }
            eprintln!(
                "real ZIM {path}: {} docs, avlen {:.2}, {} candidates, \
                 max relative difference {worst:.2e}, payload fetch {} ms",
                payload.1.docs,
                payload.1.avlen,
                payload.1.candidates.len(),
                started.elapsed().as_millis()
            );
            Ok(())
        })
        .unwrap()
        .unwrap();
    }

    #[test]
    fn e2e_search_single_archive_filter_scopes_every_tier() {
        let (_dir, library) = two_archive_library();
        let server = ZimMcpServer::new(library);

        // Unfiltered, each archive's one hit takes part in the merge.
        assert_eq!(search(&server, "banana cherry").len(), 2);
        // With the filter, only the named archive's per-archive work runs:
        // its exclusive article is the sole hit, the other archive nothing.
        let hits = search_one(&server, "a.zim", "banana cherry");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("a.zim", "C/Banana"));
        let hits = search_one(&server, "b.zim", "banana cherry");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("b.zim", "C/Cherry"));

        // A title both archives carry is not deduped across archives, so the
        // filter still reports the named archive's own copy - exactly one
        // hit, with none from the other archive.
        assert_eq!(search(&server, "apple").len(), 2);
        let hits = search_one(&server, "a.zim", "apple");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("a.zim", "C/Apple"));
        let hits = search_one(&server, "b.zim", "apple");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!((hits[0].zim.as_str(), hits[0].path.as_str()), ("b.zim", "C/Apple"));

        // A "./" prefix is trimmed like every archive name lookup.
        let hits = search_one(&server, "./a.zim", "banana cherry");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].zim, "a.zim");
    }

    #[test]
    fn e2e_search_single_archive_filter_rejects_outside_names() {
        let (_dir, library) = two_archive_library();
        let server = ZimMcpServer::new(library);

        // Unknown, traversal, and absolute names are all refused, with the
        // same NotFound message the article lookups report.
        for name in ["nope.zim", "../a.zim", "/etc/a.zim"] {
            let err = super::search(&server.library, Some(name), "apple").unwrap_err();
            let ToolError::NotFound(msg) = err else {
                panic!("expected NotFound for {name}: {err:?}");
            };
            assert!(
                msg.contains(&format!("ZIM file not found: {name} (loaded: a.zim, b.zim)")),
                "{msg}"
            );
        }
    }
    /// A filler markdown article.
    const BANANA_MD: &str = "\
# Banana

A banana is a tall herbaceous plant.
";

    /// An article in the shape wikizim_parser emits (`text/markdown`); the
    /// filler sections keep the History/India matches a minority under the
    /// coverage cap.
    const ZINC_MD: &str = "\
# Zinc

*This article is about the element. For other uses, see [[Zinc (disambiguation)]].*

**Zinc** is a [[Chemical element|chemical element]] with the symbol **Zn**.

## History

Zinc smelting is documented in ancient times.

### India

Ancient India smelted zinc early.

## Occurrence

The element occurs in several minerals.

## Uses

Brass alloys and batteries consume most of the supply.

## See also

Other transition metals are described elsewhere.
";

    #[test]
    fn e2e_search_and_section_markdown() {
        let dir = tempfile::tempdir().unwrap();
        let index = make_index(
            &[
                ("C/Zinc", "zinc chemic element symbol smelt ancient india", "Zinc"),
                ("C/Banana", "banana tree tall herbaceou plant growth", "Banana"),
                ("C/Glacier", "glacier ice dens movem weight flow", "Glacier"),
            ],
        );
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

        // Search: the preview is plain text derived from the Markdown, free
        // of markup, and is the lead's first sentence - the leading title
        // line and the hatnote are dropped. An exact match carries no
        // sections.
        let hits = search(&server, "zinc");
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

        // A query matching only a body section reports the matched sections
        // (nested one included); the preview stays the article's lead (the
        // intro holds no match, so _intro is absent). The filler sections
        // in ZINC_MD keep the article at six regions, so the two matching
        // ones stay a minority under the coverage cap.
        let hits = search(&server, "smelting");
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].sections,
            Some(vec!["History".to_string(), "India".to_string()]),
            "{:?}",
            hits[0]
        );
        assert_eq!(hits[0].preview, "Zinc is a chemical element with the symbol Zn.");
        // includes the subsection, reports the heading as written.
        let params = serde_json::from_value::<ZimGetSectionDirParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "history" }),
        )
        .unwrap();
        let result = block_on(ZimGetSectionDirTool::invoke(&server, params)).unwrap();
        assert_eq!(result.section, "History");
        assert!(result.content.contains("ancient times"), "{:?}", result.content);
        assert!(result.content.contains("Ancient India smelted zinc early"));

        // The reserved intro name: the raw Markdown between the leading
        // title line and the first heading.
        let params = serde_json::from_value::<ZimGetSectionDirParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "_intro" }),
        )
        .unwrap();
        let result = block_on(ZimGetSectionDirTool::invoke(&server, params)).unwrap();
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
        let params = serde_json::from_value::<ZimGetSectionDirParams>(
            serde_json::json!({ "zim": "md.zim", "path": "Zinc", "section": "Nope" }),
        )
        .unwrap();
        assert!(matches!(
            block_on(ZimGetSectionDirTool::invoke(&server, params)),
            Err(ToolError::SectionNotFound(_))
        ));
    }
}
