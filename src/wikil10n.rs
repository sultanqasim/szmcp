//! Per-language strings for the HTML->Markdown converter, mirroring
//! wikizim_parser's wikil10n.py: the localized '## Key facts' and
//! '## Categories' headings, the per-language boilerplate sections to
//! drop and the per-language edit-link residue markers. Unknown
//! languages fall back to the English defaults.

/// ZIM/BCP-47/ISO-639-3 language code -> the ISO 639-1 key used here
/// (first code of a comma/space list, BCP-47 subtags stripped, lowercased;
/// the ISO 639-3 and 639-2/B codes of the supported languages fold to
/// their 639-1 key, like lang_map.rs does for the stemmer).
pub fn normalize_language(lang: Option<&str>) -> String {
    let Some(lang) = lang else { return String::new() };
    let first = lang.split([',', ';', ' ', '\t', '\n', '\r']).next().unwrap_or("");
    let tok = first.split('-').next().unwrap_or("").to_lowercase();
    match tok.as_str() {
        "fra" | "fre" => "fr".into(),
        "spa" | "esp" => "es".into(),
        _ => tok,
    }
}

/// The infobox section heading ('Key facts'/'Données clés'/'Datos clave').
pub fn key_facts_title(lang: Option<&str>) -> &'static str {
    match normalize_language(lang).as_str() {
        "fr" => "Données clés",
        "es" => "Datos clave",
        _ => "Key facts",
    }
}

/// The category section heading ('Categories'/'Catégories'/'Categorías').
///
/// Scaffold for more languages: add an arm above the default, keyed on
/// the normalized code like the "fr" arm (fra/fre already fold to "fr").
pub fn categories_title(lang: Option<&str>) -> &'static str {
    match normalize_language(lang).as_str() {
        "fr" => "Catégories",
        "es" => "Categorías",
        _ => "Categories",
    }
}

/// Default sub-heading for a multi-infobox page ('Details'/'Détails'/'Detalles').
pub fn details_title(lang: Option<&str>) -> &'static str {
    match normalize_language(lang).as_str() {
        "fr" => "Détails",
        "es" => "Detalles",
        _ => "Details",
    }
}

/// Extra (non-English) boilerplate section headings to drop, lower-case.
///
/// Spanish Wikipedia names its reference/external-link boilerplate
/// differently; 'Véase también' is kept (it mirrors the kept
/// 'See also'/'Voir aussi'). 'Notas' is the bare footnotes section the
/// English 'Notes' arm covers (Spanish uses the singular form).
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
        "es" => &[
            "referencias",
            "notas",
            "notas y referencias",
            "notas y fuentes",
            "notas, fuentes y referencias",
            "fuentes y referencias",
            "enlaces externos",
            "bibliografía",
            "webografía",
        ],
        _ => &[],
    }
}

/// The bracketed edit-link residue markers the cleanup pass strips,
/// per language. The English marker stays in every list so the safety
/// net keeps covering the English string on any archive.
pub fn edit_markers(lang: Option<&str>) -> &'static [&'static str] {
    match normalize_language(lang).as_str() {
        "fr" => &["[edit]", "[modifier]"],
        "es" => &["[edit]", "[editar]"],
        _ => &["[edit]"],
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
        assert_eq!(normalize_language(Some("spa")), "es");
        assert_eq!(normalize_language(Some("esp")), "es");
        assert_eq!(normalize_language(Some("es")), "es");
        assert_eq!(normalize_language(Some("es-ES")), "es");
        assert_eq!(normalize_language(Some("spa, eng")), "es");
        assert_eq!(normalize_language(Some("eng")), "eng");
        assert_eq!(normalize_language(None), "");
        assert_eq!(normalize_language(Some("")), "");
    }

    #[test]
    fn categories_title_localized() {
        assert_eq!(categories_title(Some("fra")), "Catégories");
        assert_eq!(categories_title(Some("fre")), "Catégories");
        assert_eq!(categories_title(Some("fr")), "Catégories");
        assert_eq!(categories_title(Some("spa")), "Categorías");
        assert_eq!(categories_title(Some("es-MX")), "Categorías");
        assert_eq!(categories_title(Some("eng")), "Categories");
        assert_eq!(categories_title(None), "Categories");
    }

    #[test]
    fn localized_strings() {
        assert_eq!(key_facts_title(Some("fra")), "Données clés");
        assert_eq!(details_title(Some("fr")), "Détails");
        assert_eq!(key_facts_title(Some("spa")), "Datos clave");
        assert_eq!(key_facts_title(Some("esp")), "Datos clave");
        assert_eq!(details_title(Some("es")), "Detalles");
        assert_eq!(key_facts_title(Some("eng")), "Key facts");
        assert_eq!(key_facts_title(None), "Key facts");
        assert!(extra_drop_sections(Some("fra")).contains(&"liens externes"));
        assert!(extra_drop_sections(Some("spa")).contains(&"enlaces externos"));
        assert!(extra_drop_sections(Some("spa")).contains(&"referencias"));
        assert!(extra_drop_sections(Some("spa")).contains(&"notas"));
        assert!(extra_drop_sections(Some("spa")).contains(&"bibliografía"));
        assert!(extra_drop_sections(Some("eng")).is_empty());
        assert!(extra_drop_sections(None).is_empty());
    }

    #[test]
    fn edit_markers_localized() {
        assert_eq!(edit_markers(None), &["[edit]"]);
        assert_eq!(edit_markers(Some("eng")), &["[edit]"]);
        assert_eq!(edit_markers(Some("fra")), &["[edit]", "[modifier]"]);
        assert_eq!(edit_markers(Some("spa")), &["[edit]", "[editar]"]);
        assert_eq!(edit_markers(Some("esp")), &["[edit]", "[editar]"]);
    }
}