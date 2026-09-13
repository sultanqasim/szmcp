# szmcp — Sultan's ZIM MCP

An [MCP](https://modelcontextprotocol.io) server that serves content from
[Kiwix ZIM files](https://www.kiwix.org/). It exposes the ZIM articles in a
directory through three tools over the streamable HTTP transport, with CORS
headers so it can be used from browsers.

## Tools

- **`zim_search`** — full-text search through all articles in all ZIM files
  (Xapian, over the uncompressed `X/fulltext/xapian` index embedded in each
  archive). Returns a JSON array; each result has the ZIM file name (relative
  to the ZIM directory), the article path inside the ZIM file, the page title,
  and a short intro.
- **`zim_get`** — get the full content of an article/page/object. Arguments:
  the ZIM file name and the article path. Returns the title, final path (after
  redirects), MIME type, and all of the content (UTF-8 text, or base64 for
  binary objects).
- **`zim_get_section`** — get a single section of an article, identified by
  its heading text (e.g. `"History"`). Returns the page title, the section
  name, and the section content.

## HTML and Markdown archives

Both classic HTML Wikipedia ZIMs and Markdown ZIMs (as produced by
`wikizim_parser`, articles with MIME type `text/markdown`) are supported.
Search intros and section extraction use an HTML or a Markdown parser
depending on the article's MIME type, so Markdown articles yield clean plain
text intros and Markdown section content; everything else behaves the same.

## Build

Requires a [Xapian 2.x](https://xapian.org/) installation (e.g. `brew install
xapian` on macOS, `apt install libxapian-dev` on Debian/Ubuntu; set
`XAPIAN_DIR` for non-standard prefixes — see `xapian2/README.md`).

```
cargo build --release
```

## Run

```
./target/release/szmcp /path/to/zim-folder [--bind 127.0.0.1] [-p 3001]
```

- Positional argument: a folder containing ZIM files (scanned recursively;
  chunked archives `*.zimaa…` are supported).
- `--bind`: bind address (default `127.0.0.1`).
- `-p`/`--port`: port (default `3001`).

The MCP endpoint is served at the root path (`http://127.0.0.1:3001`).

## Notes

- The Xapian full-text index is opened **in place** (memory-mapped by Xapian
  at its offset inside the ZIM file) — no multi-gigabyte copies. A copy to a
  temp file is used only if the index blob cannot be opened in place (e.g. it
  spans archive chunks).
- Searches run with Porter2/English stemming and OR semantics, matching how
  openZIM builds its indexes.
- Articles stored in LZMA2-compressed clusters (older ZIMs) are supported via
  `xz2`; zstd and uncompressed clusters are handled natively.
