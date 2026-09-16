// Minimal C ABI over the Xapian 2.x C++ API.
//
// Design notes:
// - Every Xapian call is guarded: exceptions are captured into a per-thread
//   string, read back with xapian2_take_error().
// - Handles are opaque pointers to heap-allocated C++ objects.
// - XDoc wraps Xapian::Document together with a scratch std::string so that
//   get_data()/get_value() results can be handed across the FFI boundary
//   without a second C++-side copy. The returned byte range is valid until
//   the next call on the same XDoc, or until the handle is freed.
// - Xapian 2.x has no memview constructor. Single-file glass databases can
//   instead be opened from a file descriptor positioned at the database's
//   first byte (xapian2_db_open_fd); Xapian takes ownership of the fd and
//   memory-maps the region itself, which is the zero-copy path for
//   databases embedded inside a larger file (e.g. the fulltext index item
//   inside a ZIM archive).

#include <xapian.h>

#include <cstdint>
#include <string>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace {

// Set by the most recent failing call on this thread.
thread_local std::string g_error;

std::string describe(const Xapian::Error &e) {
    return e.get_msg() + " (" + e.get_description() + ")";
}

} // namespace

struct XDoc {
    Xapian::Document doc;
    std::string scratch;
};

// XStem wraps Xapian::Stem together with a scratch std::string so that the
// stem of a word can be handed across the FFI boundary without a second
// C++-side copy. The returned byte range is valid until the next call on
// the same XStem, or until the handle is freed.
struct XStem {
    Xapian::Stem stem;
    std::string scratch;
};

// XTermGen wraps Xapian::TermGenerator. Its get_document() result is handed
// across the FFI boundary as its own XDoc (which owns the scratch buffer).
struct XTermGen {
    Xapian::TermGenerator tg;
};

extern "C" {

// ---- Errors -------------------------------------------------------------

// NUL-terminated message from the most recent failed call on this thread.
// Empty string when the last call succeeded. Valid until the next call that
// may fail.
const char *xapian2_take_error(void) { return g_error.c_str(); }

// ---- Database (read-only) -----------------------------------------------

Xapian::Database *xapian2_db_open(const char *path, int flags) {
    try {
        return new Xapian::Database(path, flags);
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

// Open a single-file glass database from `fd`, positioned at the database's
// first byte. Xapian takes ownership of `fd` and closes it on success and on
// failure alike.
Xapian::Database *xapian2_db_open_fd(int fd, int flags) {
    try {
        return new Xapian::Database(fd, flags);
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

uint32_t xapian2_db_doccount(const Xapian::Database *db) {
    try {
        return static_cast<uint32_t>(db->get_doccount());
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

// The term's document frequency (`get_termfreq`); 0 for an absent term.
uint32_t xapian2_db_termfreq(const Xapian::Database *db, const char *term, uint32_t len) {
    try {
        return static_cast<uint32_t>(db->get_termfreq(std::string_view(term, len)));
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

XDoc *xapian2_db_get_document(const Xapian::Database *db, uint32_t did) {
    try {
        return new XDoc{db->get_document(did), {}};
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return nullptr;
}

// Compact to `output` as a glass *directory*.
int xapian2_db_compact(Xapian::Database *db, const char *output) {
    try {
        db->compact(output, 0, 0);
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

// Compact to a *single-file* glass database in the file `path`
// (created or truncated). Xapian takes ownership of the fd.
int xapian2_db_compact_single_file(Xapian::Database *db, const char *path) {
    // Xapian requires a readable, writable, seekable fd for compact().
    int fd = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        g_error = std::string("failed to open output file: ") + strerror(errno);
        return -1;
    }
    try {
        db->compact(fd, Xapian::DBCOMPACT_SINGLE_FILE, 0);
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

void xapian2_db_free(Xapian::Database *db) { delete db; }

// ---- Document ------------------------------------------------------------

XDoc *xapian2_doc_new(void) {
    try {
        return new XDoc{Xapian::Document(), {}};
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

int xapian2_doc_set_data(XDoc *d, const char *data, uint32_t len) {
    try {
        d->doc.set_data(std::string_view(data, len));
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

int xapian2_doc_add_term(XDoc *d, const char *term, uint32_t len, uint32_t increment) {
    try {
        d->doc.add_term(std::string_view(term, len), increment);
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

int xapian2_doc_set_value(XDoc *d, uint32_t slot, const char *value, uint32_t len) {
    try {
        d->doc.add_value(slot, std::string_view(value, len));
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

uint32_t xapian2_doc_id(const XDoc *d) {
    try {
        return static_cast<uint32_t>(d->doc.get_docid());
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

uint32_t xapian2_doc_termlist_count(const XDoc *d) {
    try {
        return static_cast<uint32_t>(d->doc.termlist_count());
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

const char *xapian2_doc_data(XDoc *d, uint32_t *out_len) {
    try {
        d->scratch = d->doc.get_data();
        *out_len = static_cast<uint32_t>(d->scratch.size());
        return d->scratch.data();
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
        *out_len = 0;
    }
    return nullptr;
}

const char *xapian2_doc_value(XDoc *d, uint32_t slot, uint32_t *out_len) {
    try {
        d->scratch = d->doc.get_value(slot);
        *out_len = static_cast<uint32_t>(d->scratch.size());
        return d->scratch.data();
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
        *out_len = 0;
    }
    return nullptr;
}

void xapian2_doc_free(XDoc *d) { delete d; }

// ---- Stem -------------------------------------------------------------------
//
// Standalone stemmer, e.g. to stem query words the same way libzim indexed
// them (the ZIM full-text indexes store unprefixed Porter2 stems).

XStem *xapian2_stem_new(const char *language) {
    try {
        return new XStem{Xapian::Stem(std::string_view(language), false), {}};
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

// Stem `word` (operator() is const, but the result goes through the scratch
// buffer, so the Rust side treats the call as exclusive access).
const char *xapian2_stem_apply(XStem *s, const char *word, uint32_t len, uint32_t *out_len) {
    try {
        s->scratch = s->stem(std::string(word, len));
        *out_len = static_cast<uint32_t>(s->scratch.size());
        return s->scratch.data();
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
        *out_len = 0;
    }
    return nullptr;
}

void xapian2_stem_free(XStem *s) { delete s; }

// ---- TermGenerator ---------------------------------------------------------
//
// Indexes text into a Document the way an indexer would (tokenize, fold,
// stem, count wdf) - the way libzim builds the full-text indexes embedded
// in ZIM archives (STEM_ALL over folded text, no positions).

XTermGen *xapian2_tg_new(void) {
    try {
        return new XTermGen{Xapian::TermGenerator()};
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

int xapian2_tg_set_stemmer(XTermGen *tg, const char *language) {
    try {
        tg->tg.set_stemmer(Xapian::Stem(std::string_view(language), false));
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return -1;
}

int xapian2_tg_set_stemming_strategy(XTermGen *tg, int strategy) {
    try {
        tg->tg.set_stemming_strategy(
            static_cast<Xapian::TermGenerator::stem_strategy>(strategy));
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

int xapian2_tg_set_document(XTermGen *tg, const XDoc *d) {
    try {
        tg->tg.set_document(d->doc);
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

// Index `text` into the current document without positional data (the ZIM
// full-text indexes carry none either, and positions cost memory).
int xapian2_tg_index_text_without_positions(XTermGen *tg, const char *text,
                                            uint32_t len) {
    try {
        tg->tg.index_text_without_positions(std::string_view(text, len));
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

// The document the generator has been filling (a copy; set_document starts
// a new one).
XDoc *xapian2_tg_get_document(XTermGen *tg) {
    try {
        return new XDoc{tg->tg.get_document(), {}};
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return nullptr;
}

void xapian2_tg_free(XTermGen *tg) { delete tg; }

// ---- WritableDatabase (minimal; used for building/test databases) --------

Xapian::WritableDatabase *xapian2_wdb_open(const char *path, int flags) {
    try {
        return new Xapian::WritableDatabase(path, flags, 0);
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

// Create an in-memory writable database (no file system footprint: the
// path is ignored and nothing is ever written to disk). The intended use is
// a throwaway index scored against and discarded within one request; a
// commit() makes the documents searchable through an Enquire over the same
// handle.
Xapian::WritableDatabase *xapian2_wdb_open_inmemory(void) {
    try {
        return new Xapian::WritableDatabase(std::string_view(),
                                            Xapian::DB_BACKEND_INMEMORY, 0);
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

uint32_t xapian2_wdb_add_document(Xapian::WritableDatabase *db, const XDoc *d) {
    try {
        return static_cast<uint32_t>(db->add_document(d->doc));
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

int xapian2_wdb_commit(Xapian::WritableDatabase *db) {
    try {
        db->commit();
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

void xapian2_wdb_free(Xapian::WritableDatabase *db) { delete db; }

// ---- QueryParser ----------------------------------------------------------

Xapian::QueryParser *xapian2_qp_new(void) {
    try {
        return new Xapian::QueryParser();
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

int xapian2_qp_set_default_op(Xapian::QueryParser *qp, int op) {
    try {
        qp->set_default_op(static_cast<Xapian::Query::op>(op));
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

int xapian2_qp_add_prefix(Xapian::QueryParser *qp, const char *field, const char *prefix) {
    try {
        qp->add_prefix(field, prefix);
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

int xapian2_qp_add_boolean_prefix(Xapian::QueryParser *qp, const char *field, const char *prefix) {
    try {
        qp->add_boolean_prefix(field, prefix);
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

int xapian2_qp_set_stemmer(Xapian::QueryParser *qp, const char *language) {
    try {
        qp->set_stemmer(Xapian::Stem(language, false));
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

int xapian2_qp_set_stemming_strategy(Xapian::QueryParser *qp, int strategy) {
    try {
        qp->set_stemming_strategy(
            static_cast<Xapian::QueryParser::stem_strategy>(strategy));
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

int xapian2_qp_set_database(Xapian::QueryParser *qp, const Xapian::Database *db) {
    try {
        qp->set_database(*db);
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

Xapian::Query *xapian2_qp_parse_query(Xapian::QueryParser *qp,
                                      const char *query,
                                      uint32_t flags,
                                      const char *default_prefix) {
    try {
        return new Xapian::Query(qp->parse_query(query, flags, default_prefix));
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return nullptr;
}

void xapian2_qp_free(Xapian::QueryParser *qp) { delete qp; }

// ---- Query -----------------------------------------------------------------

Xapian::Query *xapian2_query_term(const char *term, uint32_t len, uint32_t wqf) {
    try {
        return new Xapian::Query(std::string_view(term, len), wqf, 0);
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

// An empty term is a match-all query.
Xapian::Query *xapian2_query_match_all(void) {
    try {
        return new Xapian::Query(std::string_view());
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return nullptr;
}

Xapian::Query *xapian2_query_combine(int op, const Xapian::Query *a, const Xapian::Query *b) {
    try {
        return new Xapian::Query(static_cast<Xapian::Query::op>(op), *a, *b);
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return nullptr;
}

void xapian2_query_free(Xapian::Query *q) { delete q; }

// ---- Enquire ----------------------------------------------------------------

Xapian::Enquire *xapian2_enquire_new(const Xapian::Database *db) {
    try {
        return new Xapian::Enquire(*db);
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return nullptr;
}

// Enquire over a WritableDatabase (a subclass of Database): after commit(),
// the committed documents are searchable through this handle. Used for
// throwaway in-memory indexes, where a separate read-only handle would add
// nothing.
Xapian::Enquire *xapian2_enquire_new_writable(const Xapian::WritableDatabase *db) {
    try {
        return new Xapian::Enquire(*db);
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return nullptr;
}

int xapian2_enquire_set_query(Xapian::Enquire *e, const Xapian::Query *q, uint32_t query_length) {
    try {
        e->set_query(*q, query_length);
        return 0;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return -1;
}

// Does not throw; kept for explicitness and parity with the C++ API.
void xapian2_enquire_set_sort_by_relevance(Xapian::Enquire *e) { e->set_sort_by_relevance(); }

// Select the weighting scheme used to score matches, by scheme name and
// constructor parameters (Xapian's documented parameter order). The name is
// matched case-insensitively and the parameter count must match the scheme's
// arity exactly; anything else is an error through the per-thread buffer.
//
// Xapian's unset default is BM25Weight() (enquire.h), i.e. the "bm25" row
// with its default parameters.
int xapian2_enquire_set_weighting(Xapian::Enquire *e,
                                  const char *scheme,
                                  const double *params,
                                  uint32_t nparams) {
    try {
        std::string name(scheme);
        for (char &c : name) {
            if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        }
        if (name == "bm25") {
            if (nparams != 5) {
                g_error = "bm25 expects 5 parameters (k1, k2, k3, b, min_normlen), got "
                          + std::to_string(nparams) + " (valid schemes: bm25, trad, bool)";
                return -1;
            }
            e->set_weighting_scheme(
                Xapian::BM25Weight(params[0], params[1], params[2], params[3], params[4]));
            return 0;
        }
        if (name == "trad") {
            if (nparams != 1) {
                g_error = "trad expects 1 parameter (k), got " + std::to_string(nparams)
                          + " (valid schemes: bm25, trad, bool)";
                return -1;
            }
            e->set_weighting_scheme(Xapian::TradWeight(params[0]));
            return 0;
        }
        if (name == "bool") {
            if (nparams != 0) {
                g_error = "bool expects no parameters, got " + std::to_string(nparams)
                          + " (valid schemes: bm25, trad, bool)";
                return -1;
            }
            e->set_weighting_scheme(Xapian::BoolWeight());
            return 0;
        }
        g_error = "unknown weighting scheme '" + name + "' (valid schemes: bm25, trad, bool)";
        return -1;
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    } catch (const std::exception &e) {
        g_error = e.what();
    }
    return -1;
}

Xapian::MSet *xapian2_enquire_get_mset(const Xapian::Enquire *e,
                                       uint32_t first,
                                       uint32_t maxitems,
                                       uint32_t atleast) {
    try {
        return new Xapian::MSet(e->get_mset(first, maxitems, atleast, nullptr, nullptr));
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return nullptr;
}

void xapian2_enquire_free(Xapian::Enquire *e) { delete e; }

// ---- MSet ---------------------------------------------------------------------
//
// Xapian::MSet supports random access, so matches are addressed by position
// within the set (0-based, relative to `first`); `rank` is the absolute rank.

uint32_t xapian2_mset_size(const Xapian::MSet *m) {
    try {
        return static_cast<uint32_t>(m->size());
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

uint32_t xapian2_mset_docid(const Xapian::MSet *m, uint32_t i) {
    try {
        return static_cast<uint32_t>(*m->operator[](i));
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

double xapian2_mset_weight(const Xapian::MSet *m, uint32_t i) {
    try {
        return m->operator[](i).get_weight();
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0.0;
}

int32_t xapian2_mset_percent(const Xapian::MSet *m, uint32_t i) {
    try {
        return m->operator[](i).get_percent();
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

uint32_t xapian2_mset_rank(const Xapian::MSet *m, uint32_t i) {
    try {
        return static_cast<uint32_t>(m->operator[](i).get_rank());
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

uint32_t xapian2_mset_termfreq(const Xapian::MSet *m, const char *term, uint32_t len) {
    try {
        return static_cast<uint32_t>(m->get_termfreq(std::string_view(term, len)));
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

int32_t xapian2_mset_convert_to_percent(const Xapian::MSet *m, double weight) {
    try {
        return m->convert_to_percent(weight);
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return 0;
}

XDoc *xapian2_mset_get_document(const Xapian::MSet *m, uint32_t i) {
    try {
        return new XDoc{m->operator[](i).get_document(), {}};
    } catch (const Xapian::Error &e) {
        g_error = describe(e);
    }
    return nullptr;
}

void xapian2_mset_free(Xapian::MSet *m) { delete m; }

} // extern "C"
