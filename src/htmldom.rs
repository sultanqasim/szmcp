//! A minimal arena-backed DOM over html5ever, shaped like lxml's trees (the
//! shape wikizim_parser's html2md.py was written against): text lives in
//! per-gap text nodes, so an element's `.text` is its first text child and
//! an element's `.tail` is the text node following it, parents are
//! reachable, and subtrees can be copied, unwrapped and removed.

use html5ever::tendril::TendrilSink;
use html5ever::{parse_document, ParseOpts};
use markup5ever_rcdom::{NodeData, RcDom};

pub type NodeId = usize;

#[derive(Debug, Clone)]
pub enum NodeKind {
    Element {
        /// Local tag name, lower-cased (libxml2, unlike the HTML spec,
        /// lower-cases everything; the converter matches lower-case tags).
        tag: String,
        /// Attributes in document order (first wins on duplicates).
        attrs: Vec<(String, String)>,
    },
    Text(String),
    Comment,
    ProcessingInstruction,
    Document,
}

struct Node {
    kind: NodeKind,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
}

/// A parsed HTML document. `root` is the `<html>` element.
pub struct Dom {
    nodes: Vec<Node>,
    root: NodeId,
}

/// A read-only reference to one node of a [`Dom`].
#[derive(Clone, Copy)]
pub struct NodeRef<'a> {
    pub dom: &'a Dom,
    pub id: NodeId,
}

impl Dom {
    /// Parse an HTML document (a fragment also parses: html5ever always
    /// wraps it in `<html>`, and every walk of the converter is
    /// position-independent, so the wrapping is invisible to it).
    pub fn parse(html: &str) -> Dom {
        let mut opts = ParseOpts::default();
        // lxml parses <noscript> contents as ordinary elements; without
        // this they would collapse into one text node.
        opts.tree_builder.scripting_enabled = false;
        let rcdom: RcDom = parse_document(RcDom::default(), opts)
            .from_utf8()
            .one(html.as_bytes());
        let mut nodes = Vec::new();
        let doc = build(&mut nodes, &rcdom.document);
        // The spec's tree builder always inserts an <html> element; a
        // fragment without one would still get it.
        let root = nodes[doc]
            .children
            .iter()
            .copied()
            .find(|&c| matches!(&nodes[c].kind, NodeKind::Element { tag, .. } if tag == "html"))
            .unwrap_or(doc);
        Dom { nodes, root }
    }

    pub fn root(&self) -> NodeRef<'_> {
        self.ref_(self.root)
    }

    pub fn ref_(&self, id: NodeId) -> NodeRef<'_> {
        NodeRef { dom: self, id }
    }

    pub fn kind(&self, id: NodeId) -> &NodeKind {
        &self.nodes[id].kind
    }

    pub fn parent_of(&self, id: NodeId) -> Option<NodeId> {
        self.nodes[id].parent
    }

    fn children_of(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id].children
    }

    /// The text of the text node before `id`'s first element child, if any
    /// (lxml's `.text`).
    pub fn text_of(&self, id: NodeId) -> Option<&str> {
        match self.children_of(id).first().map(|&c| &self.nodes[c].kind) {
            Some(NodeKind::Text(t)) => Some(t),
            _ => None,
        }
    }

    /// The text of the text node following `id` before the next element
    /// sibling, if any (lxml's `.tail`).
    pub fn tail_of(&self, id: NodeId) -> Option<&str> {
        match self.next_sibling_kind(id) {
            Some(NodeKind::Text(t)) => Some(t),
            _ => None,
        }
    }

    fn next_sibling_kind(&self, id: NodeId) -> Option<&NodeKind> {
        let parent = self.parent_of(id)?;
        let children = self.children_of(parent);
        let pos = children.iter().position(|&c| c == id)?;
        children.get(pos + 1).map(|&n| &self.nodes[n].kind)
    }

    fn prev_element(&self, id: NodeId) -> Option<NodeId> {
        let parent = self.parent_of(id)?;
        let children = self.children_of(parent);
        let pos = children.iter().position(|&c| c == id)?;
        children[..pos]
            .iter()
            .rev()
            .copied()
            .find(|&c| matches!(self.nodes[c].kind, NodeKind::Element { .. }))
    }

    fn next_element(&self, id: NodeId) -> Option<NodeId> {
        let parent = self.parent_of(id)?;
        let children = self.children_of(parent);
        let pos = children.iter().position(|&c| c == id)?;
        children[pos + 1..]
            .iter()
            .copied()
            .find(|&c| matches!(self.nodes[c].kind, NodeKind::Element { .. }))
    }

    /// Detach `id` from its parent (its tail text node stays in place).
    pub fn detach(&mut self, id: NodeId) {
        let Some(parent) = self.parent_of(id) else { return };
        self.nodes[parent].children.retain(|&c| c != id);
        self.nodes[id].parent = None;
    }

    /// lxml's `drop_tag`: replace the element with its children, merging
    /// its own text into the surrounding text (no-op when detached).
    pub fn drop_tag(&mut self, id: NodeId) {
        let Some(parent) = self.parent_of(id) else { return };
        let children = std::mem::take(&mut self.nodes[id].children);
        if children.is_empty() {
            // Nothing inside: the element vanishes, its tail stays.
            self.nodes[parent].children.retain(|&c| c != id);
            self.nodes[id].parent = None;
            merge_sibling_text(&mut self.nodes, parent);
            return;
        }
        if let Some(pos) = self.nodes[parent].children.iter().position(|&c| c == id) {
            for (offset, &child) in children.iter().enumerate() {
                self.nodes[child].parent = Some(parent);
                if offset == 0 {
                    self.nodes[parent].children[pos] = child;
                } else {
                    self.nodes[parent].children.insert(pos + offset, child);
                }
            }
        }
        self.nodes[id].parent = None;
        merge_sibling_text(&mut self.nodes, parent);
    }

    /// Deep-copy the subtree rooted at `id` into a fresh standalone `Dom`
    /// (the copy's tail is not copied — it is a sibling, not a child).
    pub fn copy_subtree(&self, id: NodeId) -> Dom {
        let mut nodes = Vec::new();
        let root = clone_into(&self.nodes, id, &mut nodes);
        Dom { nodes, root }
    }

    /// Merge runs of adjacent text children of `parent` into single text
    /// nodes (keeps the one-text-node-per-gap invariant after splices).
    pub fn merge_text(&mut self, parent: NodeId) {
        merge_sibling_text(&mut self.nodes, parent);
    }

    /// lxml splice semantics: replace the element with its children (used
    /// to flatten Parsoid <section> wrappers); the surrounding text nodes
    /// stay in place and are re-merged.
    pub fn replace_with_children(&mut self, id: NodeId) {
        let Some(parent) = self.parent_of(id) else { return };
        let children = std::mem::take(&mut self.nodes[id].children);
        if let Some(pos) = self.nodes[parent].children.iter().position(|&c| c == id) {
            self.nodes[parent].children.splice(pos..pos + 1, children.iter().copied());
            for &c in &children {
                self.nodes[c].parent = Some(parent);
            }
        }
        self.nodes[id].parent = None;
        merge_sibling_text(&mut self.nodes, parent);
    }
}

fn build(nodes: &mut Vec<Node>, rcdom: &markup5ever_rcdom::Handle) -> NodeId {
    let id = match &rcdom.data {
        NodeData::Document => {
            push_node(nodes, NodeKind::Document)
        }
        NodeData::Text { contents } => {
            push_node(nodes, NodeKind::Text(contents.borrow().to_string()))
        }
        NodeData::Comment { .. } => push_node(nodes, NodeKind::Comment),
        NodeData::ProcessingInstruction { .. } => {
            push_node(nodes, NodeKind::ProcessingInstruction)
        }
        NodeData::Doctype { .. } => push_node(nodes, NodeKind::Comment), // not renderable
        NodeData::Element { name, attrs, .. } => {
            let tag = name.local.to_string().to_lowercase();
            let attrs = attrs
                .borrow()
                .iter()
                .map(|a| {
                    (
                        a.name.local.to_string().to_lowercase(),
                        a.value.to_string(),
                    )
                })
                .collect();
            push_node(nodes, NodeKind::Element { tag, attrs })
        }
    };
    for child in rcdom.children.borrow().iter() {
        let cid = build(nodes, child);
        nodes[cid].parent = Some(id);
        nodes[id].children.push(cid);
    }
    merge_sibling_text(nodes, id);
    id
}

fn push_node(nodes: &mut Vec<Node>, kind: NodeKind) -> NodeId {
    let id = nodes.len();
    nodes.push(Node { kind, parent: None, children: Vec::new() });
    id
}

/// Merge adjacent text children (html5ever already merges, this is a
/// belt-and-braces pass over the materialized arena).
fn merge_sibling_text(nodes: &mut Vec<Node>, parent: NodeId) {
    let children = std::mem::take(&mut nodes[parent].children);
    let mut merged: Vec<NodeId> = Vec::with_capacity(children.len());
    for child in children {
        if let NodeKind::Text(t) = &nodes[child].kind {
            let t = t.clone();
            match merged.last().copied() {
                Some(last) if matches!(nodes[last].kind, NodeKind::Text(_)) => {
                    if let NodeKind::Text(prev) = &mut nodes[last].kind {
                        prev.push_str(&t);
                    }
                    nodes[child].parent = None; // folded away, unreachable
                }
                _ => merged.push(child),
            }
        } else {
            merged.push(child);
        }
    }
    nodes[parent].children = merged;
}

fn clone_into(source: &[Node], id: NodeId, nodes: &mut Vec<Node>) -> NodeId {
    let kind = match &source[id].kind {
        NodeKind::Element { tag, attrs } => NodeKind::Element {
            tag: tag.clone(),
            attrs: attrs.clone(),
        },
        NodeKind::Text(t) => NodeKind::Text(t.clone()),
        NodeKind::Comment => NodeKind::Comment,
        NodeKind::ProcessingInstruction => NodeKind::ProcessingInstruction,
        NodeKind::Document => NodeKind::Document,
    };
    let copy = push_node(nodes, kind);
    for &child in &source[id].children {
        let cid = clone_into(source, child, nodes);
        nodes[cid].parent = Some(copy);
        nodes[copy].children.push(cid);
    }
    copy
}

impl<'a> NodeRef<'a> {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn kind(&self) -> &'a NodeKind {
        self.dom.kind(self.id)
    }

    pub fn is_element(&self) -> bool {
        matches!(self.kind(), NodeKind::Element { .. })
    }

    pub fn is_text(&self) -> bool {
        matches!(self.kind(), NodeKind::Text(_))
    }

    /// The element's lower-cased tag name, or `None` for non-elements.
    pub fn tag(&self) -> Option<&'a str> {
        match self.kind() {
            NodeKind::Element { tag, .. } => Some(tag),
            _ => None,
        }
    }

    pub fn attr(&self, name: &str) -> Option<&'a str> {
        match self.kind() {
            NodeKind::Element { attrs, .. } => attrs
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.as_str()),
            _ => None,
        }
    }

    /// The element's class tokens, in attribute order.
    pub fn class_tokens(&self) -> impl Iterator<Item = &'a str> {
        self.attr("class").unwrap_or("").split_whitespace()
    }

    pub fn has_class(&self, token: &str) -> bool {
        self.class_tokens().any(|c| c == token)
    }

    /// True when any class token is in `set`.
    pub fn has_any_class(&self, set: &[&str]) -> bool {
        self.class_tokens().any(|c| set.contains(&c))
    }

    /// The element's text before its first child (lxml's `.text`).
    pub fn text(&self) -> Option<&'a str> {
        self.dom.text_of(self.id)
    }

    /// The element's text after its end tag (lxml's `.tail`).
    pub fn tail(&self) -> Option<&'a str> {
        self.dom.tail_of(self.id)
    }

    pub fn parent(&self) -> Option<NodeRef<'a>> {
        self.dom.parent_of(self.id).map(|p| self.dom.ref_(p))
    }

    /// The previous element/comment sibling (text nodes are tails).
    pub fn prev_element(&self) -> Option<NodeRef<'a>> {
        self.dom.prev_element(self.id).map(|p| self.dom.ref_(p))
    }

    /// The next element/comment sibling (text nodes are tails).
    pub fn next_element(&self) -> Option<NodeRef<'a>> {
        self.dom.next_element(self.id).map(|p| self.dom.ref_(p))
    }

    /// Element and text children, in document order.
    pub fn children(&self) -> impl Iterator<Item = NodeRef<'a>> + 'a {
        let dom = self.dom;
        dom.children_of(self.id).iter().map(move |&c| dom.ref_(c))
    }

    /// Element children only, in document order.
    pub fn element_children(&self) -> impl Iterator<Item = NodeRef<'a>> + 'a {
        self.children().filter(|c| c.is_element())
    }

    /// Collapsed text content: this element's leading text plus every
    /// descendant's text and tail (lxml's `text_content()`).
    pub fn text_content(&self) -> String {
        let mut out = String::new();
        collect_text(self, &mut out);
        out
    }

    /// This node and all descendants, pre-order (lxml's `iter()`).
    pub fn self_and_descendants(&self) -> Vec<NodeRef<'a>> {
        let mut out = Vec::new();
        collect_preorder(self.dom, self.id, true, &mut out);
        out
    }

    /// All descendants, pre-order, self excluded (lxml's
    /// `iterdescendants()`).
    pub fn descendants(&self) -> Vec<NodeRef<'a>> {
        let mut out = Vec::new();
        collect_preorder(self.dom, self.id, false, &mut out);
        out
    }

    /// The first descendant (self excluded) with the given tag.
    pub fn find(&self, tag: &str) -> Option<NodeRef<'a>> {
        self.descendants().into_iter().find(|n| n.tag() == Some(tag))
    }

    /// All descendants (self excluded) with the given tag.
    pub fn find_all(&self, tag: &str) -> Vec<NodeRef<'a>> {
        self.descendants()
            .into_iter()
            .filter(|n| n.tag() == Some(tag))
            .collect()
    }
}

fn collect_text(node: &NodeRef<'_>, out: &mut String) {
    // The leading text is one of the text children; walking the children
    // covers it and every tail exactly once.
    for child in node.children() {
        match child.kind() {
            NodeKind::Text(t) => out.push_str(t),
            NodeKind::Element { .. } => collect_text(&child, out),
            _ => {}
        }
    }
}

fn collect_preorder<'a>(
    dom: &'a Dom,
    id: NodeId,
    include_self: bool,
    out: &mut Vec<NodeRef<'a>>,
) {
    if include_self {
        out.push(dom.ref_(id));
    }
    for &child in dom.children_of(id) {
        collect_preorder(dom, child, true, out);
    }
}
