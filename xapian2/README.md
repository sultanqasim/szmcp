# xapian2

Minimal, hand-written Rust bindings for the **Xapian 2.x** search engine,
focused on serving full-text search over ZIM archives (e.g. Wikipedia).

No `autocxx`/`cxx`/`bindgen` - just a small C++ shim (`cpp/shim.cpp`)
compiled with the `cc` crate against the system Xapian 2.x, wrapped in
hand-written `extern "C"` bindings.

## Building

Requires a Xapian **2.x** installation (the 1.x API differs significantly -
there is no `open_memview`, the default query operator is `Or`, etc.).

- macOS: `brew install xapian`
- Debian/Ubuntu: `apt install libxapian-dev`
- Otherwise set `XAPIAN_DIR` to the prefix containing `include/xapian.h`
  and `lib/libxapian*`.

## API surface

| Type | Key methods | Send / Sync |
| --- | --- | --- |
| `Database` | `open(path)`, `open_at(path, offset)`, `doc_count()`, `average_length()`, `doc_length(id)`, `termfreq(t)`, `wdf(id, t)`, `get_document(id)`, `compact_to(dir)`, `compact_single_file(path)` | `Send` only |
| `Document` | `id()`, `data()`, `value(slot)`, `termlist_count()`, `set_data/add_term/set_value` (mutable) | `Send` only |
| `Query` | `term(t)`, `match_all()`, `combine(op, a, b)`, `terms()` | `Send + Sync` |
| `QueryParser` | `new()`, `parse_query(s)`, `set_default_op(op)`, `add_prefix`, `add_boolean_prefix`, `set_stemmer`, `set_database` | `Send` only |
| `Enquire` | `new(&db)`, `new_writable(&wdb)`, `set_query(&q)`, `set_sort_by_relevance()`, `set_weighting(scheme, params)`, `get_mset(first, max, atleast)` | `Send` only |
| `MSet` | `size()`, `docid/weight/percent/rank(i)`, `termfreq(t)`, `document(i)`, `iter()` | `Send + Sync` |
| `WritableDatabase` | `create(path)`, `in_memory()`, `add_document(&doc)`, `commit()` (minimal, for tests/tooling) | `Send` only |
| `bm25_weight` | free function: Xapian 2.0's default BM25 for one document from collection statistics (the pooled re-ranking behind `szmcp`'s cross-archive full-text merge) | pure `f64` math |
| `TermGenerator` | `new()`, `set_stemmer(lang)`, `set_stemming_strategy(strategy)`, `set_document(&doc)`, `index_text_without_positions(text)`, `get_document()` | `Send` only |

Errors: Xapian exceptions are captured per-thread and surfaced as
`xapian2::Error` (implements `std::error::Error`, so `?` works into
`anyhow::Error`). All functions are synchronous.

## Zero-copy over ZIM archives

ZIM archives store their fulltext index as a **single-file glass database**
at the `fulltext/xapian` entry. Xapian 2.x removed the 1.x
`Database::open_memview(addr, size)` constructor; the replacement is
`Xapian::Database(int fd, int flags)`, which opens a single-file glass DB
from a file descriptor positioned at the database's first byte and
memory-maps the region itself.

That is exactly what [`Database::open_at(path, offset, flags)`] does: open
the ZIM file read-only, seek to the index's offset, hand the fd to Xapian.
No bytes are copied; the OS page cache does the work, which is what you
want for a 15 GB Wikipedia index.

With the companion `zim` crate (which mmaps the archive and exposes
`Zim::fulltext_index() -> Content`), all that's missing is the on-disk
offset of the xapian blob within the archive file, so that `open_at` can be
called. If the blob's offset isn't exposed yet, add a small accessor to the
`zim` crate (the cluster start offset + blob offset are already computed
internally); for a single-file ZIM the two are in the same address space.

Notes:

- The database region must lie within **one** OS file. For chunked archives
  (`.zimaa`, `.zimab`, ...) point `open_at` at the chunk that contains the
  index, with the offset relative to that chunk.
- Xapian takes ownership of the fd and closes it; on failure the fd is
  already closed.
- Read-only glass databases do not lock, so opening a read-only ZIM file is
  safe.

## Typical search flow

```rust
use xapian2::{Database, Enquire, Operator, QueryParser};

let db = Database::open_at("/path/to/archive.zim", xapian_offset, Default::default())?;

let mut qp = QueryParser::new()?;
qp.set_stemmer("english")?;
qp.set_default_op(Operator::And)?;          // 2.x defaults to Or!
let query = qp.parse_query(user_input)?;    // parse errors come back as Err

let mut enquire = Enquire::new(&db)?;       // per-thread
enquire.set_query(&query)?;
enquire.set_sort_by_relevance();            // already the default

let mset = enquire.get_mset(0, 20, 0)?;     // independent snapshot
for m in mset.iter() {
    let mut doc = db.get_document(m.docid)?;
    let html = doc.data()?;                 // the ZIM page's data blob
    // serve (m.rank, m.weight, m.percent, html)
}
```

## Weighting

`Enquire::set_weighting(scheme, params)` switches the ranking scheme and its
parameters on an `Enquire` (the default is BM25 with Xapian's default
parameters, as if the call were never made). The scheme name is matched
case-insensitively and `params` are the weight class's constructor
parameters in Xapian's documented order; the count must match the scheme's
arity exactly, and an unknown scheme or wrong count returns an error
listing the valid schemes. Call it before `get_mset` - like `set_query`,
it persists on the `Enquire` and can be replaced by calling it again.

| Scheme | Weight class | Parameters (order) | Defaults |
| --- | --- | --- | --- |
| `bm25` | `Xapian::BM25Weight` | `(k1, k2, k3, b, min_normlen)` | `k1=1`, `k2=0`, `k3=1`, `b=0.5`, `min_normlen=0.5` (the unset default; `set_weighting("bm25", &[1.0, 0.0, 1.0, 0.5, 0.5])` reproduces it exactly) |
| `trad` | `Xapian::TradWeight` | `(k)` | `k=1`; equivalent to `BM25Weight(k, 0, 0, 1, 0)` - full document length normalisation, no normalisation floor |
| `bool` | `Xapian::BoolWeight` | none | every match scores weight 0 (pure boolean match set) |

Here `k1` scales how strongly within-document frequency counts, `k2` a
query-length correction factor, `k3` within-query frequency, `b` the
document length normalisation (0 = none, 1 = full), and `min_normlen` a
floor for the normalised document length (keeps very short documents from
dominating). Defaults were verified against `weight.h` of Xapian 2.0.0.

## Concurrency model

Per Xapian's documented thread-safety contract (`docs/overview.rst`,
"Thread safety" - there is no locking inside Xapian), concurrent calls on
*one* `Database` object are unsupported: glass databases mutate lazily cached
tables (postlist/position tables, value stats) even on read-only operations,
so sharing a `Database` across threads corrupts it. Instead:

- One `Database` object **per thread or per request**. Separate handles to
  the same database file are safe to use concurrently - Xapian itself says
  this is "no different to accessing the same database from two different
  processes" (read-only glass databases take no locks).
- One `Enquire` (and one `QueryParser`) per search - both are `Send` but not
  `Sync`.
- `MSet`s and `Document`s are free to move across threads.
