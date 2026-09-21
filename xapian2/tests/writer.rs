//! Tests for the writer-side bindings added for the ZIM-writer port:
//! `WritableDatabase::create_with_flags` / `set_metadata` / `compact_to_path`,
//! `TermGenerator::set_flags` / `set_max_word_length`, `Document::remove_term`
//! and `Document::indexed_text_size`.

use xapian2::{
    wdb_flags, Database, DbFlags, Document, Enquire, QueryParser, StemStrategy, TermGenerator,
    WritableDatabase,
};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("xapian2-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `create_with_flags` with the libzim indexer's flag set
/// (`DB_CREATE_OR_OVERWRITE | DB_NO_TERMLIST` = 0x21): an existing database is
/// replaced, data/values/postings survive, but no term lists are stored.
#[test]
fn create_with_flags_overwrite_and_no_termlist() {
    let dir = temp_dir("wdbflags");
    let db_dir = dir.join("db");
    let flags = wdb_flags::DB_CREATE_OR_OVERWRITE | wdb_flags::DB_NO_TERMLIST;
    assert_eq!(flags, 0x21);

    {
        let mut wdb = WritableDatabase::create_with_flags(&db_dir, flags).unwrap();
        let mut doc = Document::new().unwrap();
        doc.set_data("old").unwrap();
        doc.add_term("stale", 1).unwrap();
        wdb.add_document(&doc).unwrap();
        wdb.commit().unwrap();
    }

    // Same path, same flags: the old database is replaced, not appended to.
    {
        let mut wdb = WritableDatabase::create_with_flags(&db_dir, flags).unwrap();
        let mut doc = Document::new().unwrap();
        doc.set_data("new").unwrap();
        doc.set_value(0, "en").unwrap();
        doc.add_term("fresh", 1).unwrap();
        wdb.add_document(&doc).unwrap();
        wdb.commit().unwrap();
    }

    let db = Database::open(&db_dir).unwrap();
    assert_eq!(db.doc_count(), 1);
    assert_eq!(db.termfreq("fresh"), 1);
    assert_eq!(db.termfreq("stale"), 0);
    let mut doc = db.get_document(1).unwrap();
    assert_eq!(doc.data_str().unwrap(), "new");
    assert_eq!(String::from_utf8(doc.value(0).unwrap()).unwrap(), "en");
    // DB_NO_TERMLIST: glass stores no per-document term list, so termlist
    // reads fail (posting lists, data and values above still work).
    assert!(doc.indexed_text_size().is_err());
    drop(doc);

    // The plain `create` (flags 0) still works.
    let plain = dir.join("plain");
    WritableDatabase::create(&plain).unwrap();
    Database::open(&plain).unwrap();

    let _ = std::fs::remove_dir_all(&dir);
}

/// Metadata written through the WritableDatabase is readable through the
/// read-only Database; re-setting a key replaces its value and setting an
/// empty value removes the key (Xapian's documented behaviour).
#[test]
fn set_metadata_roundtrip() {
    let dir = temp_dir("metadata");
    let db_dir = dir.join("db");
    {
        let mut wdb = WritableDatabase::create(&db_dir).unwrap();
        wdb.set_metadata("kind", "title").unwrap();
        wdb.set_metadata("valuesmap", "title:0;targetPath:1").unwrap();
        wdb.set_metadata("language", "eng").unwrap();
        wdb.commit().unwrap();
    }

    let db = Database::open(&db_dir).unwrap();
    assert_eq!(db.get_metadata("kind").unwrap(), "title");
    assert_eq!(db.get_metadata("valuesmap").unwrap(), "title:0;targetPath:1");
    assert_eq!(db.get_metadata("language").unwrap(), "eng");
    assert_eq!(db.get_metadata("absent").unwrap(), "");
    drop(db);

    // Replacement and the empty-value removal.
    {
        let mut wdb = WritableDatabase::create(&db_dir).unwrap();
        wdb.set_metadata("kind", "fulltext").unwrap();
        wdb.set_metadata("language", "").unwrap();
        wdb.commit().unwrap();
    }
    let db = Database::open(&db_dir).unwrap();
    assert_eq!(db.get_metadata("kind").unwrap(), "fulltext");
    assert_eq!(db.get_metadata("language").unwrap(), "");

    let _ = std::fs::remove_dir_all(&dir);
}

/// The libzim indexer flow: build a throwaway database at a temp path with
/// flags 0x21, record metadata, commit, then `compact_to_path` a single-file
/// FULL-compacted database that `Database::open_at(path, 0)` can open.
#[test]
fn compact_to_path_single_file_openable() {
    let dir = temp_dir("compact");
    let tmp = dir.join("index.tmp");
    let single = dir.join("index.xapian");

    {
        let mut wdb = WritableDatabase::create_with_flags(
            &tmp,
            wdb_flags::DB_CREATE_OR_OVERWRITE | wdb_flags::DB_NO_TERMLIST,
        )
        .unwrap();
        wdb.set_metadata("valuesmap", "title:0").unwrap();
        wdb.set_metadata("kind", "fulltext").unwrap();

        let mut tg = TermGenerator::new().unwrap();
        tg.set_stemmer("english").unwrap();
        tg.set_stemming_strategy(StemStrategy::All).unwrap();
        for (id, text) in [(0u32, "first radio telescope"), (1, "second radio drama")] {
            let mut doc = Document::new().unwrap();
            doc.set_data(format!("C/article{id}")).unwrap();
            tg.set_document(&doc).unwrap();
            tg.index_text_without_positions(text).unwrap();
            wdb.add_document(&tg.get_document().unwrap()).unwrap();
        }
        wdb.commit().unwrap();
        // Metadata survives the compaction (readers need it to interpret the
        // index).
        wdb.compact_to_path(&single).unwrap();
    }

    assert!(single.is_file());

    let db = Database::open_at(&single, 0, DbFlags::NONE).unwrap();
    assert_eq!(db.doc_count(), 2);
    assert_eq!(db.get_metadata("kind").unwrap(), "fulltext");
    assert_eq!(db.get_metadata("valuesmap").unwrap(), "title:0");
    assert_eq!(db.termfreq("radio"), 2);
    assert_eq!(db.termfreq("drama"), 1);

    let mut enquire = Enquire::new(&db).unwrap();
    let mut qp = QueryParser::new().unwrap();
    qp.set_stemmer("english").unwrap();
    // Match the index's STEM_ALL terms (the parser's default STEM_SOME would
    // ask for the Z-prefixed stem "Zdrama", which this index does not store).
    qp.set_stemming_strategy(StemStrategy::All).unwrap();
    let q = qp.parse_query("drama").unwrap();
    enquire.set_query(&q).unwrap();
    let mset = enquire.get_mset(0, 10, 0).unwrap();
    assert_eq!(mset.size(), 1);
    assert_eq!(mset.document(0).unwrap().data_str().unwrap(), "C/article1");

    let _ = std::fs::remove_dir_all(&dir);
}

/// `set_flags(FLAG_NGRAMS)` and `set_max_word_length` change what gets
/// indexed (values verified against Xapian 2.0.0 directly).
#[test]
fn termgenerator_flags_and_max_word_length() {
    // Baseline: CJK text is indexed as one whole-run term without the flag.
    let mut tg = TermGenerator::new().unwrap();
    let doc = Document::new().unwrap();
    tg.set_document(&doc).unwrap();
    tg.index_text_without_positions("日本語").unwrap();
    let doc = tg.get_document().unwrap();
    assert_eq!(doc.termlist_count(), 1);

    // With FLAG_NGRAMS (2048): per-character unigrams + bigrams instead
    // (日本語 -> 日, 日本, 本, 本語, 語).
    let mut tg = TermGenerator::new().unwrap();
    tg.set_flags(xapian2::tg_flags::FLAG_NGRAMS).unwrap();
    let doc = Document::new().unwrap();
    tg.set_document(&doc).unwrap();
    tg.index_text_without_positions("日本語").unwrap();
    let doc = tg.get_document().unwrap();
    assert_eq!(doc.termlist_count(), 5);

    // Words longer than the cap index nothing at all.
    let mut tg = TermGenerator::new().unwrap();
    tg.set_max_word_length(10).unwrap();
    let doc = Document::new().unwrap();
    tg.set_document(&doc).unwrap();
    tg.index_text_without_positions("short supercalifragilistic")
        .unwrap();
    let doc = tg.get_document().unwrap();
    assert_eq!(doc.termlist_count(), 1);
    assert_eq!(doc.indexed_text_size().unwrap(), 5); // "short" only
}

/// `remove_term` removes an existing term and errors on a missing one (the
/// libzim title indexer relies on the error to detect wordless titles).
#[test]
fn document_remove_term() {
    let mut doc = Document::new().unwrap();
    doc.add_term("0posanchor", 1).unwrap();
    doc.add_term("hello", 1).unwrap();
    assert_eq!(doc.termlist_count(), 2);

    doc.remove_term("0posanchor").unwrap();
    assert_eq!(doc.termlist_count(), 1);
    // Removing it again: the term is absent -> an error, not a silent no-op.
    let err = doc.remove_term("0posanchor").unwrap_err();
    assert!(!err.msg().is_empty());
    // ...and a term that was never there errors too.
    assert!(doc.remove_term("neveradded").is_err());
    // The document's remaining terms are unaffected.
    assert_eq!(doc.termlist_count(), 1);
}

/// `indexed_text_size` matches libzim's sizeOfIndexedText: the sum of
/// `wdf * term.len()` over non-Z-prefixed terms, hand-computed.
#[test]
fn indexed_text_size_hand_computed() {
    // Manually built document: "hello" wdf 2 (2*5=10), the Z-prefixed stem
    // "Zhello" (excluded), "world" wdf 2 (2*5=10) and the anchor term
    // "0posanchor" (1*10=10).
    let mut doc = Document::new().unwrap();
    doc.add_term("hello", 1).unwrap();
    doc.add_term("hello", 1).unwrap();
    doc.add_term("Zhello", 3).unwrap();
    doc.add_term("world", 2).unwrap();
    doc.add_term("0posanchor", 1).unwrap();
    assert_eq!(doc.indexed_text_size().unwrap(), 30);

    // An empty document indexes nothing.
    let empty = Document::new().unwrap();
    assert_eq!(empty.indexed_text_size().unwrap(), 0);

    // The same sum over a document a TermGenerator filled (STEM_SOME over
    // "0posanchor Hello World", english): surface terms 0posanchor/hello/world
    // (10+5+5) plus the Z-prefixed stems, which are excluded.
    let mut tg = TermGenerator::new().unwrap();
    tg.set_stemmer("english").unwrap();
    tg.set_stemming_strategy(StemStrategy::Some).unwrap();
    let doc = Document::new().unwrap();
    tg.set_document(&doc).unwrap();
    tg.index_text_without_positions("0posanchor Hello World").unwrap();
    let doc = tg.get_document().unwrap();
    assert_eq!(doc.termlist_count(), 5);
    assert_eq!(doc.indexed_text_size().unwrap(), 20);

    // The same document, read back from a (term-listed) database.
    let dir = temp_dir("idxsize");
    let db_dir = dir.join("db");
    {
        let mut wdb = WritableDatabase::create(&db_dir).unwrap();
        wdb.add_document(&doc).unwrap();
        wdb.commit().unwrap();
    }
    let db = Database::open(&db_dir).unwrap();
    assert_eq!(db.get_document(1).unwrap().indexed_text_size().unwrap(), 20);

    let _ = std::fs::remove_dir_all(&dir);
}
