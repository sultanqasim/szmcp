//! Shared helpers for the HTML->Markdown conversion: a global regex cache,
//! whitespace collapsing, URL splitting/decoding (urllib.parse equivalents)
//! and Python's `html.unescape`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A cached compiled `regex` (the plain, linear-time engine).
pub(crate) fn re(pat: &'static str) -> &'static regex::Regex {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, &'static regex::Regex>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&rx) = cache.get(pat) {
        return rx;
    }
    let rx: &'static regex::Regex = Box::leak(Box::new(regex::Regex::new(pat).unwrap()));
    cache.insert(pat, rx);
    rx
}

/// A cached compiled `fancy-regex` (needed for lookaround/backreferences).
pub(crate) fn fre(pat: &'static str) -> &'static fancy_regex::Regex {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, &'static fancy_regex::Regex>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&rx) = cache.get(pat) {
        return rx;
    }
    let rx: &'static fancy_regex::Regex =
        Box::leak(Box::new(fancy_regex::Regex::new(pat).unwrap()));
    cache.insert(pat, rx);
    rx
}

/// `re.sub` with a replacement function over a fancy-regex.
pub(crate) fn fre_sub(
    rx: &fancy_regex::Regex,
    text: &str,
    f: impl FnMut(&fancy_regex::Captures<'_, str>) -> String,
) -> String {
    let mut out = String::new();
    let mut last = 0usize;
    let mut f = f;
    for caps in rx.captures_iter(text) {
        let caps = match caps {
            Ok(caps) => caps,
            Err(_) => break,
        };
        let m = match caps.get(0) {
            Some(m) => m,
            None => break,
        };
        out.push_str(&text[last..m.start()]);
        out.push_str(&f(&caps));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

/// The render-path whitespace collapse: runs of whitespace become one
/// space (Python's `re.sub(r"\s+", " ", s)`).
pub(crate) fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            in_ws = true;
        } else {
            if in_ws && !out.is_empty() {
                out.push(' ');
            }
            in_ws = false;
            out.push(c);
        }
    }
    if in_ws && !out.is_empty() {
        out.push(' ');
    }
    out
}

/// Python's `[ \t]+ -> " "`.
pub(crate) fn collapse_space_tab(s: &str) -> String {
    if !s.contains("  ") && !s.contains('\t') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut in_ws = false;
    for c in s.chars() {
        if c == ' ' || c == '\t' {
            in_ws = true;
        } else {
            if in_ws && !out.is_empty() {
                out.push(' ');
            }
            in_ws = false;
            out.push(c);
        }
    }
    if in_ws && !out.is_empty() {
        out.push(' ');
    }
    out
}

/// `urllib.parse.unquote` (percent-decode as UTF-8, invalid escapes kept).
pub(crate) fn percent_decode(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && bytes[i + 1].is_ascii_hexdigit()
            && bytes[i + 2].is_ascii_hexdigit()
        {
            let hex = |b: u8| -> u8 {
                (b as char).to_digit(16).unwrap_or(0) as u8
            };
            out.push((hex(bytes[i + 1]) << 4) | hex(bytes[i + 2]));
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The parts of a URL that the conversion logic needs (a small
/// `urllib.parse.urlsplit`).
pub(crate) struct SplitUrl {
    // `scheme`/`fragment` are part of the urllib.parse.urlsplit mirror;
    // the converter reads them only occasionally.
    #[allow(dead_code)]
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub query: String,
    #[allow(dead_code)]
    pub fragment: String,
}

pub(crate) fn urlsplit(url: &str) -> SplitUrl {
    // CPython strips C0 controls + space at the ends and removes tab/CR/LF.
    let cleaned: String = url
        .chars()
        .filter(|&c| c != '\t' && c != '\r' && c != '\n')
        .collect();
    let cleaned = cleaned.trim_matches(|c: char| c <= '\u{20}');
    let mut rest = cleaned;
    let mut scheme = String::new();
    if let Some(i) = rest.find(':') {
        if i > 0 {
            let head = &rest[..i];
            let mut ok = head
                .chars()
                .next()
                .map_or(false, |c| c.is_ascii_alphanumeric());
            ok &= head
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
            if ok {
                scheme = head.to_lowercase();
                rest = &rest[i + 1..];
            }
        }
    }
    let mut netloc = String::new();
    if let Some(after) = rest.strip_prefix("//") {
        let mut delim = after.len();
        for c in ['/', '?', '#'] {
            if let Some(w) = after.find(c) {
                delim = delim.min(w);
            }
        }
        netloc = after[..delim].to_string();
        rest = &after[delim..];
    }
    let mut fragment = String::new();
    if let Some(i) = rest.find('#') {
        fragment = rest[i + 1..].to_string();
        rest = &rest[..i];
    }
    let mut query = String::new();
    if let Some(i) = rest.find('?') {
        query = rest[i + 1..].to_string();
        rest = &rest[..i];
    }
    SplitUrl {
        scheme,
        netloc,
        path: rest.to_string(),
        query,
        fragment,
    }
}

/// `urllib.parse.parse_qs(qs, keep_blank_values=True)`: the key/value pairs
/// in order of appearance ('+' means space; missing '=' yields an empty
/// value).
pub(crate) fn parse_query(qs: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for chunk in qs.split('&') {
        if chunk.is_empty() {
            continue;
        }
        let (k, v) = match chunk.split_once('=') {
            Some((k, v)) => (k, v),
            None => (chunk, ""),
        };
        out.push((percent_decode(&k.replace('+', " ")), percent_decode(&v.replace('+', " "))));
    }
    out
}

// ---------------------------------------------------------------------------
// Python's html.unescape (used by the final cleanup pass)
// ---------------------------------------------------------------------------

/// CPython's `html._invalid_charrefs` (windows-1252 remapping of numeric
/// references into the C1 range, plus NUL and CR).
const INVALID_CHARREFS: &[(u32, &str)] = &[
    (0x00, "\u{FFFD}"),
    (0x0d, "\r"),
    (0x80, "\u{20AC}"),
    (0x81, "\u{0081}"),
    (0x82, "\u{201A}"),
    (0x83, "\u{0192}"),
    (0x84, "\u{201E}"),
    (0x85, "\u{2026}"),
    (0x86, "\u{2020}"),
    (0x87, "\u{2021}"),
    (0x88, "\u{02C6}"),
    (0x89, "\u{2030}"),
    (0x8a, "\u{0160}"),
    (0x8b, "\u{2039}"),
    (0x8c, "\u{0152}"),
    (0x8d, "\u{008D}"),
    (0x8e, "\u{017D}"),
    (0x8f, "\u{008F}"),
    (0x90, "\u{0090}"),
    (0x91, "\u{2018}"),
    (0x92, "\u{2019}"),
    (0x93, "\u{201C}"),
    (0x94, "\u{201D}"),
    (0x95, "\u{2022}"),
    (0x96, "\u{2013}"),
    (0x97, "\u{2014}"),
    (0x98, "\u{02DC}"),
    (0x99, "\u{2122}"),
    (0x9a, "\u{0161}"),
    (0x9b, "\u{203A}"),
    (0x9c, "\u{0153}"),
    (0x9d, "\u{009D}"),
    (0x9e, "\u{017E}"),
    (0x9f, "\u{0178}"),
];

/// True for CPython's `html._invalid_codepoints` (removed when referenced).
fn is_invalid_codepoint(num: u32) -> bool {
    matches!(num, 0x01..=0x08 | 0x0b | 0x0e..=0x1f | 0x7f..=0x9f)
        || (0xfdd0..=0xfdef).contains(&num)
        || num == 0xfffe
        || num == 0xffff
        || (num & 0xfffe) == 0xfffe && num > 0xffff
        || num > 0x10ffff
}

fn lookup_entity(name: &str) -> Option<&'static str> {
    crate::entities::HTML5_ENTITIES
        .binary_search_by(|(k, _)| (*k).cmp(name))
        .ok()
        .map(|i| crate::entities::HTML5_ENTITIES[i].1)
}

/// Python's `html.unescape`: decode numeric and HTML5 named character
/// references, with the same longest-prefix fallback for names without a
/// full-table match ('&notit;' -> '¬it;').  The final cleanup pass calls
/// this only outside fenced code, inline code spans and link destinations,
/// where `&…;` text is verbatim content rather than parser residue.
pub(crate) fn html_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        match parse_charref(after) {
            Some((replacement, consumed)) => {
                out.push_str(&replacement);
                rest = &after[consumed..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// One reference after the '&'; returns (replacement, bytes consumed).
fn parse_charref(after: &str) -> Option<(String, usize)> {
    let bytes = after.as_bytes();
    if bytes.first() == Some(&b'#') {
        // Numeric: #[0-9]+;? or #[xX][0-9a-fA-F]+;?
        let hex = matches!(bytes.get(1), Some(b'x') | Some(b'X'));
        let start = if hex { 2 } else { 1 };
        let mut j = start;
        while j < bytes.len()
            && (if hex { bytes[j].is_ascii_hexdigit() } else { bytes[j].is_ascii_digit() })
        {
            j += 1;
        }
        if j == start {
            return None;
        }
        let mut consumed = j;
        if bytes.get(j) == Some(&b';') {
            consumed += 1;
        }
        let digits = &after[start..j];
        let num: u32 = if hex {
            u32::from_str_radix(digits, 16).ok()?
        } else {
            digits.parse().ok()?
        };
        if let Some((_, v)) = INVALID_CHARREFS.iter().find(|(n, _)| *n == num) {
            return Some((v.to_string(), consumed));
        }
        if (0xd800..=0xdfff).contains(&num) || num > 0x10ffff || is_invalid_codepoint(num) {
            let replacement = if (0xd800..=0xdfff).contains(&num)
                || num > 0x10ffff
            {
                "\u{FFFD}"
            } else {
                ""
            };
            return Some((replacement.to_string(), consumed));
        }
        return Some((char::from_u32(num)?.to_string(), consumed));
    }
    // Named: [^\t\n\f <&#;]{1,32};?  (the regex char class, minus '&')
    let mut len = 0usize;
    let mut count = 0usize;
    for c in after.chars() {
        if matches!(c, '\t' | '\n' | '\x0c' | ' ' | '<' | '&' | '#' | ';') {
            break;
        }
        len += c.len_utf8();
        count += 1;
        if count == 32 {
            break;
        }
    }
    if count == 0 {
        return None;
    }
    let mut consumed = len;
    if bytes.get(len) == Some(&b';') {
        consumed += 1;
    }
    let key = &after[..consumed];
    if let Some(v) = lookup_entity(key) {
        return Some((v.to_string(), consumed));
    }
    // Longest-prefix fallback over the matched text (including any ';'),
    // from len(key)-1 chars down to 2.
    let mut boundaries = key.char_indices().map(|(i, _)| i).collect::<Vec<_>>();
    boundaries.push(key.len());
    let nchars = boundaries.len() - 1;
    for x in (2..nchars).rev() {
        let prefix = &key[..boundaries[x]];
        if let Some(v) = lookup_entity(prefix) {
            return Some((format!("{}{}", v, &key[boundaries[x]..]), consumed));
        }
    }
    None
}
