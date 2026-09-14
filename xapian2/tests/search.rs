use xapian2::{Database, DbFlags, Document, Enquire, Match, Operator, Query, QueryParser, Stem, WritableDatabase};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("xapian2-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Index three documents and verify the full search path.
#[test]
fn search_roundtrip() {
    let dir = temp_dir("roundtrip");
    let db_dir = dir.join("db");

    {
        let mut wdb = WritableDatabase::create(&db_dir).unwrap();
        let docs = [
            ("alpha", "<p>first: alpha</p>"),
            ("beta", "<p>second: beta</p>"),
            // Four indexed terms so doc lengths differ: doc1 (1 term) must
            // outrank doc3 (4 terms) for the single-term query "alpha".
            ("alpha beta beta beta", "<p>third: alpha and beta</p>"),
        ];
        for (terms, data) in docs {
            let mut doc = Document::new().unwrap();
            doc.set_data(data.as_bytes()).unwrap();
            for t in terms.split_whitespace() {
                doc.add_term(t, 1).unwrap();
            }
            doc.set_value(0, "en").unwrap();
            wdb.add_document(&doc).unwrap();
        }
        wdb.commit().unwrap();
    }

    let db = Database::open(&db_dir).unwrap();
    assert_eq!(db.doc_count(), 3);

    let mut enquire = Enquire::new(&db).unwrap();
    let mut qp = QueryParser::new().unwrap();

    // Xapian 2.x default combining operator is OR: "alpha beta" matches all three.
    let q = qp.parse_query("alpha beta").unwrap();
    enquire.set_query(&q).unwrap();
    enquire.set_sort_by_relevance();
    let mset = enquire.get_mset(0, 10, 0).unwrap();
    assert_eq!(mset.size(), 3);
    assert_eq!(mset.docid(0), 3); // both terms -> strongest match
    assert_eq!(mset.termfreq("alpha"), 2);
    assert_eq!(mset.termfreq("zeta"), 0);

    let mut doc = mset.document(0).unwrap();
    assert_eq!(doc.id(), Some(3));
    assert_eq!(doc.termlist_count(), 2);
    assert!(String::from_utf8_lossy(&doc.data().unwrap()).contains("alpha and beta"));
    assert_eq!(doc.data_str().unwrap(), "<p>third: alpha and beta</p>");
    assert_eq!(String::from_utf8(doc.value(0).unwrap()).unwrap(), "en");
    assert!(doc.value(5).unwrap().is_empty());

    // Percent/weight consistency.
    let w = mset.weight(0);
    let p = mset.percent(0);
    assert_eq!(p, mset.convert_to_percent(w));
    assert!((0..=100).contains(&p));

    // Switch to AND.
    qp.set_default_op(Operator::And).unwrap();
    let q = qp.parse_query("alpha beta").unwrap();
    enquire.set_query(&q).unwrap();
    let mset = enquire.get_mset(0, 10, 0).unwrap();
    assert_eq!(mset.size(), 1);
    assert_eq!(mset.docid(0), 3);

    // Explicit boolean syntax also works.
    let q = qp.parse_query("alpha AND beta").unwrap();
    enquire.set_query(&q).unwrap();
    let mset = enquire.get_mset(0, 10, 0).unwrap();
    assert_eq!(mset.size(), 1);

    // Windowing + absolute rank.
    qp.set_default_op(Operator::Or).unwrap();
    let q = qp.parse_query("alpha").unwrap();
    enquire.set_query(&q).unwrap();
    let mset = enquire.get_mset(0, 1, 0).unwrap();
    assert_eq!(mset.size(), 1);
    let m = mset.iter().next().unwrap();
    assert_eq!(m.rank, 0);
    // Single term "alpha": doc1 (length 1) beats doc3 (length 4) under BM25.
    assert_eq!(m.docid, 1);

    // Iterator and manual access agree.
    let mset = enquire.get_mset(0, 10, 0).unwrap();
    let via_iter: Vec<u32> = mset.iter().map(|m| m.docid).collect();
    let via_index: Vec<u32> = (0..mset.size()).map(|i| mset.docid(i)).collect();
    assert_eq!(via_iter, via_index);

    // Match is Copy and self-contained.
    let first: Match = mset.iter().next().unwrap();
    let second = first; // copy
    assert_eq!(first, second);

    // Errors surface with messages.
    let err = db.get_document(99).unwrap_err();
    assert!(err.msg().contains("99"));
    assert!(db.get_document(0).is_err());
    // (Xapian auto-balances stray parentheses; a dangling boolean operator
    // is the reliable syntax error.)
    let err = qp.parse_query("alpha AND").unwrap_err();
    assert!(err.msg().contains("AND"));
    assert!(!err.msg().is_empty());
    let err = Database::open(dir.join("nope")).unwrap_err();
    assert!(!err.msg().is_empty());

    // Query combinators.
    let a = Query::term("alpha").unwrap();
    let b = Query::term("beta").unwrap();
    let q = Query::combine(Operator::And, &a, &b).unwrap();
    enquire.set_query(&q).unwrap();
    let mset = enquire.get_mset(0, 10, 0).unwrap();
    assert_eq!(mset.size(), 1);

    // Match-all.
    let q = Query::match_all().unwrap();
    enquire.set_query(&q).unwrap();
    let mset = enquire.get_mset(0, 10, 0).unwrap();
    assert_eq!(mset.size(), 3);

    let _ = std::fs::remove_dir_all(&dir);
}

/// The standalone stemmer: stemming, the empty word, and errors.
#[test]
fn stem_words() {
    let mut stem = Stem::new("english").unwrap();
    // The stems the ZIM full-text indexes store (libzim, STEM_ALL); the
    // QueryParser's STEM_ALL strategy produces the same terms.
    assert_eq!(stem.apply("elephants").unwrap(), "eleph");
    assert_eq!(stem.apply("dependent").unwrap(), "depend");
    assert_eq!(stem.apply("miami").unwrap(), "miami");
    assert_eq!(stem.apply("").unwrap(), "");
    // An unknown language is an error, not a silent no-op.
    assert!(Stem::new("notalanguage").is_err());
}

/// Single-file glass database embedded at an offset in a larger file -
/// the ZIM `open_at` scenario (zero-copy: Xapian mmaps the region).
#[test]
fn single_file_embedded_open_at() {
    let dir = temp_dir("singlefile");
    let db_dir = dir.join("db");

    {
        let mut wdb = WritableDatabase::create(&db_dir).unwrap();
        for i in 0..3u32 {
            let mut doc = Document::new().unwrap();
            doc.set_data(format!("doc {i}").as_bytes()).unwrap();
            doc.add_term("gamma", 1).unwrap();
            wdb.add_document(&doc).unwrap();
        }
        wdb.commit().unwrap();
    }

    let db = Database::open(&db_dir).unwrap();
    let single = dir.join("single.xdb");
    db.compact_single_file(&single).unwrap();
    assert!(single.is_file());
    // In Xapian 2.x a standalone single-file glass DB is opened via the
    // fd constructor (offset 0), not by path.
    let sanity = Database::open_at(&single, 0, DbFlags::NONE).unwrap();
    assert_eq!(sanity.doc_count(), 3);

    // Prepend padding to simulate embedding in a larger file.
    let glass = std::fs::read(&single).unwrap();
    let padded = dir.join("padded.bin");
    let mut out = vec![0xaa_u8; 7];
    out.extend_from_slice(&glass);
    std::fs::write(&padded, &out).unwrap();

    let db2 = Database::open_at(&padded, 7, DbFlags::NONE).unwrap();
    assert_eq!(db2.doc_count(), 3);

    let mut enquire = Enquire::new(&db2).unwrap();
    let mut qp = QueryParser::new().unwrap();
    let q = qp.parse_query("gamma").unwrap();
    enquire.set_query(&q).unwrap();
    let mset = enquire.get_mset(0, 10, 0).unwrap();
    assert_eq!(mset.size(), 3);
    for m in mset.iter() {
        let mut doc = mset.document(m.rank).unwrap();
        assert_eq!(doc.id(), Some(m.docid));
        assert_eq!(doc.data_str().unwrap(), format!("doc {}", m.docid - 1));
    }

    // A wrong offset must fail, not read garbage.
    assert!(Database::open_at(&padded, 4, DbFlags::NONE).is_err());

    let _ = std::fs::remove_dir_all(&dir);
}
