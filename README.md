# szmcp — Sultan's ZIM MCP

An [MCP](https://modelcontextprotocol.io) server and CLI that serve content
from [Kiwix ZIM files](https://www.kiwix.org/). The ZIM articles are exposed
through a small set of tools, either as an MCP server over the streamable
HTTP transport (with CORS headers so it can be used from browsers) or as
one-shot CLI subcommands. There is also functionality to convert ZIM archives
from HTML to Markdown for space efficiency and readability.

## Why I made this

One task that I've found LLMs useful for is answering questions about the
world. Without web search or RAG, generally you need to run a large model
with 100B+ parameters to get usable results, or at least a knowledge dense
model like Gemma 4 31B, but that is generally slow, prone to hallucination
for more obscure facts, and impossible on average personal computers or mobile
devices. I also don't like having the LLM search the web for my queries, as it
defeats the privacy and offline functionality goals of running local LLMs.

People (particlularly on Reddit) like to say that world knowledge in LLMs
is of little value or useless, due to the risk of hallucination. They always
say it's better give LLMs access to documents or RAG, but for offline setups
there tends to be a lot of "draw the rest of the owl" involved. There are a
huge range of different types of world knowledge that LLMs are trained on,
and Wikipedia is only one piece of that. It's hard to keep an offline copy
of the internet at home, and even harder to make it efficiently searchable.
Nonetheless, Wikipedia and other sites that Kiwix arhives and packages into
portable and easily searchable ZIM files are a good starting point.

One package I found that provides an MCP server for browsing ZIM files is
[openzim-mcp](https://github.com/cameronrye/openzim-mcp). I tried it out,
but found it to be overly complicated and bloated with features of limited
practical value (at least for my use cases), and its one-tool simplified mode
was near-useless with small models struggling to use it more than its advanced
tool set. I also found its complexity made it very context inefficient for
simple queries, and thus very slow on ordinary computers with slow prompt
processing. Other small basic ZIM MCP servers also exist, but the ones I found
seemed immature, unpolished, and lacking in functionality.

Two other similar projects are [Zimi](https://github.com/epheterson/Zimi)
and [zim-mcp-server](https://github.com/ThinkInAI-Hackathon/zim-mcp-server).
While I have not used either, Zimi appears to be polished and well maintained,
with a rich set of features, though one could say it has everything but the
kitchen sink. The zim-mcp-server project appears similar to this, though with
a more simplistic search mechanism that doesn't support searching across
multiple archives and combining the results, and it doesn't appear to be
actively maintained (not that it's necessary for a simple thing).

My goal was to build a ZIM MCP server that provides just the functionality
one really needs, which is easy to use (for LLMs and humans), and which is
context efficient, making usage with small local LLMs on not overly powerful
computers practical. It's also a single binary written in Rust, keeping
deployment simple and the code relatively efficient.

I also wanted good quality conversion of Wikipedia pages from HTML to Markdown,
in a manner that uses Markdown syntax effectively, makes infoboxes easily
readable, preserves complex tables as HTML while keeping simple tables pure
Markdown, and which generally produces nice clean-looking output on all pages.
I wrote a bunch of custom HTML to Markdown conversion logic to nicely handle
Wikipedia pages, while also providing a more generic HTML to Markdown conversion
path for non-wiki pages. The HTML to Markdown and ZIM rewriting code is
horrendously complicated and far from beautiful, but it works.

This project is unashamedly almost entirely vibe coded. I made it with GLM 5.3
Flash mostly. Even this README was all LLM generated, aside from this one section.
I won't pretend this is some masterpiece, but it does the job and seems solid.
You can connect it to the [llama.cpp](https://github.com/ggml-org/llama.cpp)
llama-server web UI's built in agentic loop, or connect it to your own agent
of choice. You can also just use the tools it exposes from the command line
interface to manually explore ZIM files.

## Tools

The tool set is decided by how the server was launched, not by the directory
contents: `serve` with a single ZIM file runs in **single mode**, with a
folder — holding any number of ZIM files, from none on up — in **directory
mode**.

- **`zim_search`** — search across the ZIM files in three tiers: an exact
  title/URL match comes first (a matching redirect reports the article it
  points to), then articles whose title contains every query word, then
  full-text matches ranked by BM25 relevance over all query words — partial
  matches still return. Each result has the ZIM file name (relative to the
  ZIM directory — this `zim` field is reported in both modes), the article
  path, the title, and `preview` — the article's first intro paragraph,
  truncated at a word boundary. A full-text match also carries
  `sections` — the names of the article regions whose text matches the query,
  BM25-scored, the intro listed as `_intro`. `sections` is omitted for title
  matches, when no region matches, and when so many regions match that the
  whole article is likely relevant. The same article is reported once even
  when several spellings of it match.
  Deduplication is within an archive only: the same title in different
  archives is reported once per archive, since different archives can hold
  different articles under one title. In single mode the tool takes only the
  query, plus an optional `limit` on the number of results (default 10). In
  directory mode an optional `zim` argument restricts the search
  to one file; without it all ZIM files are searched: their full-text hits
  are re-scored with one BM25 formula over pooled cross-archive statistics
  and merged best-first (a strong match in a small archive outranks a weak
  match in a big one), while the exact and title tiers keep their
  per-archive ranking.
- **`zim_get`** — get the full content of an article/page/object. Arguments:
  the article path — or an article title such as `"Beaconsfield, Quebec"`,
  converted to its `C/` path (Wikipedia ZIMs); a failed lookup reports the
  converted path, so the exact path from search results can be retried.
  Returns the title, final path (after redirects), MIME type, and all of the
  content (UTF-8 text, or base64 for binary objects). HTML pages are
  automatically converted to Markdown with the `wikizim_parser` conventions
  (`zim2zim --infobox` behavior: the infobox renders as a `## Key facts`
  section, boilerplate sections are dropped); pages without a wiki article
  body (scraped non-wiki sites, meta-refresh stubs) render from their
  `<body>` element, so every text/html entry comes back as Markdown.
  In single mode that is
  the whole argument list. In directory mode a `zim` argument is also
  required: the ZIM file name relative to the ZIM directory, as `zim_list`
  reports it.
- **`zim_get_section`** — get a single section of an article, identified by
  its heading text (e.g. `"History"`), or the special name `_intro` for the
  introduction (the region before the first heading). The article is
  addressed by its path or its title, as in `zim_get` (plus `zim` in
  directory mode). HTML wiki articles are converted to Markdown first, so
  the headings are those of the converted text (and `_intro` includes the
  `## Key facts` block). Returns the page title, the section name, and the
  section content.
- **`zim_list`** (directory mode only) — list the loaded ZIM files as
  `{"files": ["file1.zim", "xyx/file3.zim"]}`, names relative to the ZIM
  directory.

A `zim` argument must name a file inside the ZIM directory: `..` components
and absolute paths are refused, while symlinks are fine.

## HTML and Markdown archives

Both classic HTML Wikipedia ZIMs and Markdown ZIMs (as produced by
`wikizim_parser`, articles with MIME type `text/markdown`) are supported.
`zim_get`/`zim_get_section` convert HTML wiki articles to Markdown with the
same converter `wikizim_parser/zim2zim.py --infobox` uses (the Rust port in
`src/html2md.rs` + `src/infobox_html.rs`: paragraphs, headings, wikilinks,
fenced code, math, pipe/HTML tables, hatnotes, the infobox as a `## Key
facts` section, per-language boilerplate drops — byte-identical output on
the reference article sets), so both archive kinds read the same way; the
CLI's `--raw` flag returns the unconverted HTML. Search text and section
extraction use an HTML or a Markdown parser
depending on the article's MIME type, so Markdown articles yield clean
plain-text search text and Markdown section content; everything else
behaves the same.

## Converting an HTML ZIM to Markdown (`convert`)

`szmcp convert` is a Rust port of `wikizim_parser/zim2zim.py`: a two-pass
pipeline over the source archive converts every `text/html`
article to Markdown (`text/markdown`, same paths, infoboxes on), recreates
every redirect whose chain resolves to a source HTML article (targets
resolved transitively, cycle-safe), copies the core metadata and the 48x48
illustration, and builds fresh fulltext + title Xapian indexes exactly like
libzim 9.8.2 does (values slots, anchor-term title indexing with positions,
FLAG_CJK_NGRAM, stemmer chosen via the ICU primary language of the source's
`Language` metadata, DB_NO_TERMLIST + single-file compaction, index language
metadata = the raw code). Core metadata keys are copied verbatim; the accent
folding libzim expects is always applied to the indexed text.

Pass 1 is the only full walk: a pool of worker threads (= the CPU count,
override with `SZMCP_CONVERT_THREADS`) converts fixed 512-entry chunks
dynamically pulled from a shared counter, building both Xapian documents
OUTSIDE the database mutexes so that only the `add_document` calls
serialize; every entry leaves a 12-byte record (an article's blob
reference, or a redirect's resolved terminal). Membership is then pure
arithmetic over the record flags (a redirect is live iff its terminal was
converted), and a single-threaded finalize walks the members' dirent
headers exactly once to stream the dirents in source order, build the
title-ordered listing and rank redirect targets. RAM stays bounded:
content streams to disk as clusters close, dirents stream as they are
emitted, and the only per-entry state is the 12-byte record plus the
listing rows (~1 GB at 30M articles). Both embedded indexes commit every
10000 documents, which bounds the uncommitted Xapian buffers — the only
RSS term that scales with the converted-article count — at a few hundred
MB. Entry order, metadata, counter and listing stay deterministic; cluster
packing and the Xapian document ids follow worker completion order,
exactly like libzim's own racing workers. Measured with 4 threads: peak
RSS on the 8.3 GB `wikipedia_en_top_maxi_2026-06.zim` is ~0.7 GiB at
`--limit 50000` (153.9 s with 4 threads).

```
szmcp convert <input.zim> <output.zim> [--limit N] [--index-intro-only] [--index-redirect-titles]
```

- `--limit N` processes only the first N source entries (every entry counts
  one) — a development aid; articles beyond the cutoff are not converted, so
  the output may contain dangling redirects (dropped at finalization).
- `--index-intro-only` fulltext-indexes each article's intro (title line plus
  paragraphs before the first `## ` heading, hatnotes removed) instead of the
  whole Markdown.
- `--index-redirect-titles` gives recreated redirects the FRONT_ARTICLE hint
  so their titles enter the title index (default: excluded, zim2zim's
  `--no-redirect-titles` behavior; redirects resolve either way).
- Progress and the summary go to stderr; the output is a ZIM 6.x archive
  readable by libzim/Kiwix and this tool.

Verified against python `zim2zim.py` reference builds of the same input
(`--limit 3000`, with and without redirect titles, and with
`--index-intro-only`): header fields, checksum, mime list, the full dirent
sequence (paths, stored titles, mime strings, resolved redirect targets),
metadata values, `M/Counter`, the title-ordered listing (byte-equal), index
document counts, per-document data/value slots and term sets with posting
statistics all match — documents compared keyed by their data (the path),
since both indexes assign document ids in worker completion order, unlike
the python build's sorted order. The exceptions are the divergences below.

Divergences from the python reference, all deliberate or inherent:

- Fulltext documents are added in deterministic conversion order; libzim
  adds them from racing worker threads (the doc *sets* are equal).
- About 1.6% of articles (12 of 759 on the reference pair) carry the
  html2md deliberate fixes (recovered table content, deduplicated
  definition-list text, `<br>` paragraph splits), so their Markdown,
  fulltext `wordcount` value and the terms of that recovered text differ
  from the python build.
- No stopword lists are bundled — matching zim2zim's ZIMs, whose 3-letter
  language codes never load a libzim stopword resource.
- Accent folding is simplified: only Latin combining marks
  (U+0300..U+036F) are stripped. zim2zim strips every Unicode mark, which
  also normalizes Arabic-script words (`إ` → `ا`, kasras dropped), so
  Arabic-script etymology terms differ from the python build's.
- The stemmer is Xapian 2.0.0's bundled Snowball; the python reference
  bundles Xapian 1.4.23. Stem forms differ for some languages — including
  English (`university` → `universiti`, `internal` → `internal` vs
  `intern`), so a few Z-prefixed title terms and fulltext stems differ;
  surface forms, document sets and counts do not.

## Build

Requires a [Xapian 2.x](https://xapian.org/) installation (e.g. `brew install
xapian` on macOS, `apt install libxapian-dev` on Debian/Ubuntu; set
`XAPIAN_DIR` for non-standard prefixes — see `xapian2/README.md`).

```
cargo build --release
```

## Run

Every mode takes either a single ZIM file or a folder containing ZIM files as
its first argument (folders are scanned recursively; chunked archives
`*.zimaa…` are supported).

```
szmcp serve /path/to/zim-folder [--bind 127.0.0.1] [-p 3001]
szmcp search /path/to/zim-folder "query"
szmcp search /path/to/one-file.zim "query"
szmcp get /path/to/file.zim C/SomeArticle
szmcp get_section /path/to/file.zim C/SomeArticle "History"
```

- `serve` runs the MCP server. The endpoint is served at the root path
  (`http://127.0.0.1:3001`); `--bind` sets the bind address (default
  `127.0.0.1`), `-p`/`--port` the port (default `3001`). Whether the
  argument is a single ZIM file or a folder decides the tool set (single
  vs. directory mode, see Tools).
- `search`, `get` and `get_section` run the matching tool once and print its
  response JSON to stdout — the same JSON the MCP tool returns, without the
  MCP wrapper. `search` takes an optional `--limit` capping the number of
  results (default 10, matching the tool's `limit` argument). `get` and
  `get_section` take the ZIM file itself — the archive
  is identified by its own path; search results name ZIM files relative to
  the scanned directory. Both also take `--raw` to skip the HTML→Markdown
  conversion and print/extract from the raw HTML instead. Errors go to
  stderr and exit non-zero.

## Notes

- The Xapian full-text index is opened **in place** (memory-mapped by Xapian
  at its offset inside the ZIM file) — no multi-gigabyte copies. A copy to a
  temp file is used only if the index blob cannot be opened in place (e.g. it
  spans archive chunks).
- Searches run with Porter2/English stemming and OR semantics, matching how
  openZIM builds its indexes.
- Articles stored in LZMA2-compressed clusters (older ZIMs) are supported via
  `xz2`; zstd and uncompressed clusters are handled natively.
