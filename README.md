# szmcp — Sultan's ZIM MCP

An [MCP](https://modelcontextprotocol.io) server and CLI that serve content
from [Kiwix ZIM files](https://www.kiwix.org/). The ZIM articles in a
directory are exposed through three tools, either as an MCP server over the
streamable HTTP transport (with CORS headers so it can be used from
browsers) or as one-shot CLI subcommands.

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

My goal was to build a ZIM MCP server that provides just the functionality
one really needs, which is easy to use (for LLMs and humans), and which is
context efficient, making usage with small local LLMs on not overly powerful
computers practical.

This project is unashamedly almost entirely vibe coded. I made it with GLM 5.3
Flash mostly. Even this README was all LLM generated, aside from this one section.
I won't pretend this is some masterpiece, but it does the job and seems solid.
You can connect it to the [llama.cpp](https://github.com/ggml-org/llama.cpp)
llama-server web UI's built in agentic loop, or connect it to your own agent
of choice. You can also just use the tools it exposes from the command line
interface to manually explore ZIM files.

This project **does not convert HTML into Markdown**. Content is presented to
the LLM in its original format. Nevertheless, given that Markdown is much more
token efficient than HTML, and perhaps easier for the LLM to parse too, I
recommend using Wikipedia ZIM files that have been converted to Markdown.
Such files are also more space efficient on disk. I made a Python
[script](https://github.com/sultanqasim/wikizim_parser) to do this conversion.

## Tools

- **`zim_search`** — search across all ZIM files in three tiers: an exact
  title/URL match comes first (a matching redirect reports the article it
  points to), then articles whose title contains every query word, then
  full-text matches ranked by BM25 relevance over all query words — partial
  matches still return. Each result has the ZIM file name (relative to the ZIM
  directory), the article path, the title, and `preview` — the article's first
  intro sentence when the query matches the title, otherwise
  the sentence with the most query matches (followed by the rest of its
  paragraph) together with `sections` (the matching regions' names, the
  intro listed as `_intro`). `sections` is omitted when the query matches
  the title or the first intro paragraph. The same article is reported once
  even when several spellings of it match.
  Use `zim` + `path` with `zim_get`/`zim_get_section`.
- **`zim_get`** — get the full content of an article/page/object. Arguments:
  the ZIM file name and the article path — or an article title such as
  `"Beaconsfield, Quebec"`, converted to its `C/` path (Wikipedia ZIMs);
  a failed lookup reports the converted path, so the exact path from search
  results can be retried. Returns the title, final path (after
  redirects), MIME type, and all of the content (UTF-8 text, or base64 for
  binary objects).
- **`zim_get_section`** — get a single section of an article, identified by
  its heading text (e.g. `"History"`), or the special name `_intro` for the
  introduction (the region before the first heading). The article is
  addressed by its path or its title, as in `zim_get`. Returns the page
  title, the section name, and the section content.

## HTML and Markdown archives

Both classic HTML Wikipedia ZIMs and Markdown ZIMs (as produced by
`wikizim_parser`, articles with MIME type `text/markdown`) are supported.
Search text and section extraction use an HTML or a Markdown parser
depending on the article's MIME type, so Markdown articles yield clean
plain-text search text and Markdown section content; everything else
behaves the same.

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
  `127.0.0.1`), `-p`/`--port` the port (default `3001`).
- `search`, `get` and `get_section` run the matching tool once and print its
  response JSON to stdout — the same JSON the MCP tool returns, without the
  MCP wrapper. `get` and `get_section` take the ZIM file itself — the archive
  is identified by its own path; search results name ZIM files relative to
  the scanned directory. Errors go to stderr and exit non-zero.

## Notes

- The Xapian full-text index is opened **in place** (memory-mapped by Xapian
  at its offset inside the ZIM file) — no multi-gigabyte copies. A copy to a
  temp file is used only if the index blob cannot be opened in place (e.g. it
  spans archive chunks).
- Searches run with Porter2/English stemming and OR semantics, matching how
  openZIM builds its indexes.
- Articles stored in LZMA2-compressed clusters (older ZIMs) are supported via
  `xz2`; zstd and uncompressed clusters are handled natively.
