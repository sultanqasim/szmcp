# szmcp — Sultan's ZIM MCP

An [MCP](https://modelcontextprotocol.io) server and CLI that serve content
from [Kiwix ZIM files](https://www.kiwix.org/). The ZIM articles in a
directory are exposed through three tools, either as an MCP server over the
streamable HTTP transport (with CORS headers so it can be used from
browsers) or as one-shot CLI subcommands.

## Tools

- **`zim_search`** — full-text search across all ZIM files. Multi-word queries
  rank articles containing all the words higher, but partial matches are
  still returned; an article whose title or redirect exactly matches the
  query comes first. Each result has the ZIM file name (relative to the ZIM
  directory), the article path, the title, and `preview` — the article's first
  paragraph when the query matches the title or that paragraph, otherwise
  the sentence with the most query matches (followed by the rest of its
  paragraph) together with `sections` (the matching regions' names, the
  intro listed as `_intro`). `sections` is omitted when the query matches
  the title or the first intro paragraph.
  Use `zim` + `path` with `zim_get`/`zim_get_section`.
- **`zim_get`** — get the full content of an article/page/object. Arguments:
  the ZIM file name and the article path. Returns the title, final path (after
  redirects), MIME type, and all of the content (UTF-8 text, or base64 for
  binary objects).
- **`zim_get_section`** — get a single section of an article, identified by
  its heading text (e.g. `"History"`), or the special name `_intro` for the
  introduction (the region before the first heading). Returns the page
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
szmcp get /path/to/zim-folder file.zim C/SomeArticle
szmcp get_section /path/to/zim-folder file.zim C/SomeArticle "History"
```

- `serve` runs the MCP server. The endpoint is served at the root path
  (`http://127.0.0.1:3001`); `--bind` sets the bind address (default
  `127.0.0.1`), `-p`/`--port` the port (default `3001`).
- `search`, `get` and `get_section` run the matching tool once and print its
  response JSON to stdout — the same JSON the MCP tool returns, without the
  MCP wrapper. The ZIM file name is relative to the ZIM directory, as given
  in search results. Errors go to stderr and exit non-zero.

## Notes

- The Xapian full-text index is opened **in place** (memory-mapped by Xapian
  at its offset inside the ZIM file) — no multi-gigabyte copies. A copy to a
  temp file is used only if the index blob cannot be opened in place (e.g. it
  spans archive chunks).
- Searches run with Porter2/English stemming and OR semantics, matching how
  openZIM builds its indexes.
- Articles stored in LZMA2-compressed clusters (older ZIMs) are supported via
  `xz2`; zstd and uncompressed clusters are handled natively.
