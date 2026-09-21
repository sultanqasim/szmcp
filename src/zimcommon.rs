//! Format facts shared by the ZIM reader ([`crate::zim`]) and the ZIM
//! writer ([`crate::zimwrite`]): the magic number, the MIME sentinels, the
//! dirent ordering rule, and the 80-byte file header ([`ZimHeader`]) —
//! parsed and serialized side by side so the two sides cannot drift apart.

use std::cmp::Ordering;

/// ZIM magic number ("ZIM\x04" little-endian).
pub const ZIM_MAGIC: u32 = 0x044D_495A;

/// Directory-entry MIME sentinels (values `0xffff..0xfffd`).
pub const MIME_REDIRECT: u16 = 0xffff;
pub const MIME_LINKTARGET: u16 = 0xfffe;
pub const MIME_DELETED: u16 = 0xfffd;

/// Header size in bytes; also the position of the MIME type list
/// (`mimeListPos` is always this value).
pub const HEADER_SIZE: u64 = 80;

/// Cluster info byte layout (libzim's `Cluster` header): the low nibble is
/// the compression id, bit 4 marks 64-bit blob offsets.
pub const CLUSTER_COMPRESSION_MASK: u8 = 0x0f;
pub const CLUSTER_EXTENDED_BIT: u8 = 0x10;
/// Cluster compression ids (libzim's `compression` values; 0 is the
/// legacy "no compression" encoding readers still accept).
pub const CLUSTER_UNCOMPRESSED: u8 = 1;
pub const CLUSTER_LZMA: u8 = 4;
pub const CLUSTER_ZSTD: u8 = 5;

/// Old-scheme header fields the new namespace scheme leaves empty: the
/// title pointer list position and the layout page. Both are written so
/// that old readers see "no title index" / "no layout".
pub const NO_TITLE_PTR_POS: u64 = u64::MAX;
pub const NO_LAYOUT_PAGE: u32 = u32::MAX;

/// The canonical directory-entry order: the namespace byte first, then the
/// stored url bytes (`libzim`'s `comparePath` / `strcmp`). The reader's
/// directory binary search and the writer's `BTreeMap<(u8, String)>` keys
/// both follow exactly this rule — one definition, two consumers.
#[inline]
pub fn dirent_order(a: (u8, &[u8]), b: (u8, &[u8])) -> Ordering {
    a.cmp(&b)
}

/// The parsed 80-byte ZIM file header (libzim's `Fileheader`). Field names
/// follow the reader's historical vocabulary (`url_ptr_pos` = libzim's
/// `pathPtrPos`); `uuid` and `main_page` round out the fields the writer
/// produces and the `convert` pipeline consumes.
#[derive(Debug, Clone)]
pub struct ZimHeader {
    pub major: u16,
    pub minor: u16,
    pub uuid: [u8; 16],
    pub entry_count: u32,
    pub cluster_count: u32,
    /// Offset of the url (path) pointer list — libzim's `pathPtrPos`.
    pub url_ptr_pos: u64,
    /// Offset of the old-scheme title pointer list; `u64::MAX` in
    /// new-scheme archives (see [`NO_TITLE_PTR_POS`]). Parsed but unused.
    pub title_ptr_pos: u64,
    pub cluster_ptr_pos: u64,
    pub mime_list_pos: u64,
    pub main_page: u32,
    /// Offset of the old-scheme layout page; `u32::MAX` when absent
    /// (see [`NO_LAYOUT_PAGE`]). Parsed but unused.
    pub layout_page: u32,
    pub checksum_pos: u64,
}

impl ZimHeader {
    /// Parse the first 80 bytes of an archive. `b` must hold at least 80
    /// bytes; the caller checks the magic and version.
    pub fn parse(b: &[u8]) -> ZimHeader {
        ZimHeader {
            major: u16le(&b[4..6]),
            minor: u16le(&b[6..8]),
            uuid: b[8..24].try_into().unwrap(),
            entry_count: u32le(&b[24..28]),
            cluster_count: u32le(&b[28..32]),
            url_ptr_pos: u64le(&b[32..40]),
            title_ptr_pos: u64le(&b[40..48]),
            cluster_ptr_pos: u64le(&b[48..56]),
            mime_list_pos: u64le(&b[56..64]),
            main_page: u32le(&b[64..68]),
            layout_page: u32le(&b[68..72]),
            checksum_pos: u64le(&b[72..80]),
        }
    }

    /// Serialize the header exactly as libzim's `Fileheader::write` lays it
    /// out. The old-scheme title/layout fields carry their "absent" values.
    pub fn serialize(&self) -> [u8; 80] {
        let mut h = [0u8; HEADER_SIZE as usize];
        h[0..4].copy_from_slice(&ZIM_MAGIC.to_le_bytes());
        h[4..6].copy_from_slice(&self.major.to_le_bytes());
        h[6..8].copy_from_slice(&self.minor.to_le_bytes());
        h[8..24].copy_from_slice(&self.uuid);
        h[24..28].copy_from_slice(&self.entry_count.to_le_bytes());
        h[28..32].copy_from_slice(&self.cluster_count.to_le_bytes());
        h[32..40].copy_from_slice(&self.url_ptr_pos.to_le_bytes());
        h[40..48].copy_from_slice(&self.title_ptr_pos.to_le_bytes());
        h[48..56].copy_from_slice(&self.cluster_ptr_pos.to_le_bytes());
        h[56..64].copy_from_slice(&self.mime_list_pos.to_le_bytes());
        h[64..68].copy_from_slice(&self.main_page.to_le_bytes());
        h[68..72].copy_from_slice(&self.layout_page.to_le_bytes());
        h[72..80].copy_from_slice(&self.checksum_pos.to_le_bytes());
        h
    }
}

#[inline]
pub(crate) fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
#[inline]
pub(crate) fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
#[inline]
pub(crate) fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().unwrap())
}
