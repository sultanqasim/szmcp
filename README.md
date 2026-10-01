# szmcp — Sultan's ZIM MCP

An [MCP](https://modelcontextprotocol.io) server and CLI that serve content
from [Kiwix ZIM files](https://www.kiwix.org/). The ZIM articles are exposed
through a small set of tools, either as an MCP server over the streamable
HTTP transport (with CORS headers so it can be used from browsers) or as
one-shot CLI subcommands. There is also functionality to convert ZIM archives
from HTML to Markdown for space efficiency and readability, with special
handling of MediaWiki features like infoboxes and formulas for clean,
well-formatted, and information-preserving wiki page conversion.

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
Flash mostly. Even this README was mostly LLM generated, aside from this section.
I won't pretend this is some masterpiece, but it does the job and seems solid.
You can connect it to the [llama.cpp](https://github.com/ggml-org/llama.cpp)
llama-server web UI's built in agentic loop, or connect it to your own agent
of choice. You can also just use the tools it exposes from the command line
interface to manually explore ZIM files.

## Tools

The tool set is decided by how the server was launched, not by the directory
contents: `serve` with a single ZIM file runs in **single mode**, with a
folder of ZIM files — **directory mode**.

- **`zim_search`** — search across the ZIM files in three tiers: exact
  title/URL matches first (a matching redirect reports the article it points
  to), then articles whose title contains every query word, then full-text
  matches ranked by relevance. Each result has the ZIM file name (`zim`,
  relative to the ZIM directory), the article path, the title, and a
  `preview` — the article's first intro paragraph. Full-text matches may
  also carry `sections` — the matching article regions, with the intro
  listed as `_intro`. In single mode the tool takes only the query plus an
  optional `limit` (default 10); in directory mode an optional `zim`
  argument restricts the search to one file, otherwise all ZIM files are
  searched and their results merged.
- **`zim_get`** — get the full content of an article/page/object, addressed
  by its path or by its title (titles like `"New York City"` are converted
  to their `C/` path; a failed lookup reports the path it tried).
  Returns the title, final path (after redirects), MIME type, and the
  content (UTF-8 text, or base64 for binary objects). HTML pages are
  automatically converted to Markdown. In directory mode a `zim` argument
  is also required: the ZIM file name relative to the ZIM directory, as
  `zim_list` reports it.
- **`zim_get_section`** — like `zim_get`, but returns a single section,
  identified by its heading text (e.g. `"History"`) or `_intro` for the
  introduction (the region before the first heading). Headings are those of
  the Markdown-converted article.
- **`zim_list`** (directory mode only) — list the loaded ZIM files, names
  relative to the ZIM directory.

A `zim` argument must name a file inside the ZIM directory: `..` components
and absolute paths are refused; symlinks are fine.

## HTML and Markdown archives

Both classic HTML Wikipedia ZIMs and Markdown ZIMs (as produced by
`szmcp convert`) are supported. `zim_get`/`zim_get_section` convert HTML
articles to Markdown on the fly. Wiki infoboxes render as a `## Key facts`
section, and boilerplate sections and references are dropped. The conversion
logic is tailored to give good quality Markdown conversion of Wikipedia
articles, but can also convert non-wiki pages. The CLI's `--raw` flag on
the `get` and `get_sections` subcommands returns the unconverted HTML.

## Converting an HTML ZIM to Markdown (`convert`)

The `convert` subcommand of `szmcp` converts every HTML article in a ZIM file
to Markdown. It maintains the same article paths, and uses the same conversion
logic as the `zim_get`/`zim_get_section` tools. It recreates redirects, copies
the core metadata, and rebuilds the full-text and title search indexes. Images
are excluded from the generated ZIM file. The output is a ZIM 6.x archive
readable by libzim/Kiwix and this tool. The converter is multi-threaded and
designed to be fairly memory efficient. Nonetheless, buulding a Xapian full-text
search index for the full English Wikipedia is memory intensive, so 24+ GB of
RAM is recommended if you are converting a `wikipedia_en_all` Kiwix archive.

```
szmcp convert <input.zim> <output.zim> [--limit N] [--index-intro-only] [--index-redirect-titles]
```

- `--limit N` — process only the first N source entries (a development aid;
  the output may then contain dangling redirects).
- `--index-intro-only` — fulltext-index only each article's intro instead of
  the whole Markdown.
- `--index-redirect-titles` — include recreated redirect titles in the title
  index (default: excluded; redirects resolve either way).

Progress and a summary go to stderr. The output is verified against Python
reference builds of the same input; a few deliberate divergences remain
(simplified accent folding, a newer Xapian stemmer, no bundled stopwords).

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

- `serve` — runs the MCP server at the root path (`http://127.0.0.1:3001` by
  default; `--bind` sets the bind address, `-p`/`--port` the port). A single
  file vs. a folder selects single vs. directory mode.
- `search`, `get`, `get_section` — run the matching tool once and print the
  same JSON the MCP tool returns. `search` takes an optional `--limit`
  (default 10). `get` and `get_section` take `--raw` to skip the
  HTML→Markdown conversion. Errors go to stderr and exit non-zero.

## Notes

- The Xapian full-text index is opened **in place** (memory-mapped inside the
  ZIM file) — no multi-gigabyte copies.
- Searches run with Porter2/English stemming and OR semantics, matching how
  openZIM builds its indexes.
- LZMA2-compressed clusters (older ZIMs) are supported via `xz2`; zstd and
  uncompressed clusters are handled natively.
