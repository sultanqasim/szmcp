//! Per-language strings for the HTML->Markdown converter, mirroring
//! wikizim_parser's wikil10n.py: the localized '## Key facts' heading and
//! the per-language boilerplate sections to drop. Unknown languages fall
//! back to the English defaults.

/// ZIM/BCP-47/ISO-639-3 language code -> the ISO 639-1 key used here
/// (first code of a comma/space list, BCP-47 subtags stripped, lowercased).
pub fn normalize_language(lang: Option<&str>) -> String {
    let Some(lang) = lang else { return String::new() };
    let first = lang.split([',', ';', ' ', '\t', '\n', '\r']).next().unwrap_or("");
    let mut tok = first.split('-').next().unwrap_or("").to_lowercase();
    if tok == "fra" || tok == "fre" {
        tok = "fr".into();
    }
    tok
}

fn string(key: &'static str, default: &'static str, lang: Option<&str>) -> &'static str {
    let norm = normalize_language(lang);
    match norm.as_str() {
        "fr" => match key {
            "key_facts" => "Données clés",
            "details" => "Détails",
            _ => default,
        },
        _ => default,
    }
}

/// The infobox section heading ('Key facts' / 'Données clés').
pub fn key_facts_title(lang: Option<&str>) -> &'static str {
    string("key_facts", "Key facts", lang)
}

/// Default sub-heading for a multi-infobox page ('Details'/'Détails').
pub fn details_title(lang: Option<&str>) -> &'static str {
    string("details", "Details", lang)
}

/// Extra (non-English) boilerplate section headings to drop, lower-case.
pub fn extra_drop_sections(lang: Option<&str>) -> &'static [&'static str] {
    match normalize_language(lang).as_str() {
        "fr" => &[
            "références",
            "notes et références",
            "notes et sources",
            "notes, sources et références",
            "sources et références",
            "liens externes",
            "bibliographie",
            "webographie",
        ],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_codes() {
        assert_eq!(normalize_language(Some("fra")), "fr");
        assert_eq!(normalize_language(Some("fre")), "fr");
        assert_eq!(normalize_language(Some("fr")), "fr");
        assert_eq!(normalize_language(Some("fr-FR")), "fr");
        assert_eq!(normalize_language(Some("fra, eng")), "fr");
        assert_eq!(normalize_language(Some("eng")), "eng");
        assert_eq!(normalize_language(None), "");
        assert_eq!(normalize_language(Some("")), "");
    }

    #[test]
    fn localized_strings() {
        assert_eq!(key_facts_title(Some("fra")), "Données clés");
        assert_eq!(details_title(Some("fr")), "Détails");
        assert_eq!(key_facts_title(Some("eng")), "Key facts");
        assert_eq!(key_facts_title(None), "Key facts");
        assert!(extra_drop_sections(Some("fra")).contains(&"liens externes"));
        assert!(extra_drop_sections(Some("eng")).is_empty());
    }
}
