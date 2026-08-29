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
| `Database` | `open(path)`, `open_at(path, offset)`, `doc_count()`, `get_document(id)`, `compact_to(dir)`, `compact_single_file(path)` | `Send + Sync` |
| `Document` | `id()`, `data()`, `value(slot)`, `termlist_count()`, `set_data/add_term/set_value` (mutable) | `Send` only |
| `Query` | `term(t)`, `match_all()`, `combine(op, a, b)` | `Send + Sync` |
| `QueryParser` | `new()`, `parse_query(s)`, `set_default_op(op)`, `add_prefix`, `add_boolean_prefix`, `set_stemmer`, `set_database` | `Send` only |
| `Enquire` | `new(&db)`, `set_query(&q)`, `set_sort_by_relevance()`, `get_mset(first, max, atleast)` | `Send` only |
| `MSet` | `size()`, `docid/weight/percent/rank(i)`, `termfreq(t)`, `document(i)`, `iter()` | `Send + Sync` |
| `WritableDatabase` | `create(path)`, `add_document(&doc)`, `commit()` (minimal, for tests/tooling) | `Send` only |

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

## Concurrency model

- One shared `Database` (cheap to hold in a server struct).
- One `Enquire` (and one `QueryParser`) per request/thread - both are
  `Send` but not `Sync`.
- `MSet`s and `Document`s are free to move across threads.
