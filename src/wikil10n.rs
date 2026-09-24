//! Per-language strings for the HTML->Markdown converter, mirroring
//! wikizim_parser's wikil10n.py: the localized '## Key facts' and
//! '## Categories' headings and the per-language boilerplate sections to
//! drop. Unknown languages fall back to the English defaults.

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

/// The infobox section heading ('Key facts' / 'Données clés').
pub fn key_facts_title(lang: Option<&str>) -> &'static str {
    match normalize_language(lang).as_str() {
        "fr" => "Données clés",
        _ => "Key facts",
    }
}

/// The category section heading ('Categories'/'Catégories').
///
/// Scaffold for more languages: add an arm above the default, keyed on
/// the normalized code like the "fr" arm (fra/fre already fold to "fr").
pub fn categories_title(lang: Option<&str>) -> &'static str {
    match normalize_language(lang).as_str() {
        "fr" => "Catégories",
        _ => "Categories",
    }
}

/// Default sub-heading for a multi-infobox page ('Details'/'Détails').
pub fn details_title(lang: Option<&str>) -> &'static str {
    match normalize_language(lang).as_str() {
        "fr" => "Détails",
        _ => "Details",
    }
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
    fn categories_title_localized() {
        assert_eq!(categories_title(Some("fra")), "Catégories");
        assert_eq!(categories_title(Some("fre")), "Catégories");
        assert_eq!(categories_title(Some("fr")), "Catégories");
        assert_eq!(categories_title(Some("eng")), "Categories");
        assert_eq!(categories_title(None), "Categories");
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
