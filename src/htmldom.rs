//! A minimal arena-backed DOM over html5ever, shaped like lxml's trees (the
//! shape wikizim_parser's html2md.py was written against): text lives in
//! per-gap text nodes, so an element's `.text` is its first text child and
//! an element's `.tail` is the text node following it, parents are
//! reachable, and subtrees can be copied, unwrapped and removed.

use html5ever::interface::tree_builder::{ElementFlags, NodeOrText, QuirksMode, TreeSink};
use html5ever::tendril::{StrTendril, TendrilSink};
use html5ever::{parse_document, Attribute, ExpandedName, ParseOpts, QualName};
use std::cell::UnsafeCell;
use std::collections::HashMap;

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
    /// Comments, processing instructions and doctypes: never rendered.
    Comment,
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
        // ~40 bytes of HTML per arena node on real pages; a single
        // capacity guess avoids most of the growth reallocations.
        let sink = ArenaSink::with_capacity(html.len() / 40 + 16);
        let nodes = parse_document(sink, opts).from_utf8().one(html.as_bytes());
        // The spec's tree builder always inserts an <html> element; a
        // fragment without one would still get it.
        let root = nodes[0]
            .children
            .iter()
            .copied()
            .find(|&c| matches!(&nodes[c].kind, NodeKind::Element { tag, .. } if tag == "html"))
            .unwrap_or(0);
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
        let parent = self.parent_of(id);
        if self.nodes[id].children.is_empty() {
            // Nothing inside: the element vanishes, its tail stays.
            self.detach(id);
        } else {
            self.replace_with_children(id);
        }
        if let Some(parent) = parent {
            self.merge_text(parent);
        }
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

/// A tag/attribute name as a String, ASCII-lower-cased only when it has
/// uppercase (html5ever already lower-cases HTML names; only foreign
/// names arrive mixed-case and the converter matches lower-case tags).
/// One allocation, versus `to_string().to_lowercase()`'s two.
fn lower_name(name: &str) -> String {
    let mut s = name.to_string();
    if s.bytes().any(|b| b.is_ascii_uppercase()) {
        s.make_ascii_lowercase();
    }
    s
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
        let merges = matches!(&nodes[child].kind, NodeKind::Text(_))
            && merged.last().is_some_and(|&last| matches!(&nodes[last].kind, NodeKind::Text(_)));
        if !merges {
            merged.push(child);
            continue;
        }
        // Move the folded child's text out (its node leaves the tree
        // below), so no second borrow of `nodes` is held for the write.
        let text = match &mut nodes[child].kind {
            NodeKind::Text(t) => std::mem::take(t),
            _ => unreachable!(),
        };
        let last = *merged.last().unwrap();
        let NodeKind::Text(prev) = &mut nodes[last].kind else { unreachable!() };
        prev.push_str(&text);
        nodes[child].parent = None; // folded away, unreachable
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
        other @ (NodeKind::Comment | NodeKind::Document) => other.clone(),
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

    /// The ancestor chain (immediate parent upwards), self excluded.
    pub fn ancestors(&self) -> Ancestors<'a> {
        Ancestors { dom: self.dom, cur: self.dom.parent_of(self.id) }
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
    pub fn self_and_descendants(&self) -> impl Iterator<Item = NodeRef<'a>> + 'a {
        std::iter::once(*self).chain(self.descendants())
    }

    /// All descendants, pre-order, self excluded (lxml's
    /// `iterdescendants()`). Lazy: [`Self::find`] early-exits on the walk
    /// and [`Self::find_all`] collects only its matches — no whole-subtree
    /// materialization.
    pub fn descendants(&self) -> Descendants<'a> {
        Descendants::new(self.dom, self.id)
    }

    /// The first descendant (self excluded) with the given tag.
    pub fn find(&self, tag: &str) -> Option<NodeRef<'a>> {
        self.descendants().find(|n| n.tag() == Some(tag))
    }

    /// All descendants (self excluded) with the given tag.
    pub fn find_all(&self, tag: &str) -> Vec<NodeRef<'a>> {
        self.descendants().filter(|n| n.tag() == Some(tag)).collect()
    }
}

/// Iterator over a node's ancestors (immediate parent upwards).
pub struct Ancestors<'a> {
    dom: &'a Dom,
    cur: Option<NodeId>,
}

impl<'a> Iterator for Ancestors<'a> {
    type Item = NodeRef<'a>;

    fn next(&mut self) -> Option<NodeRef<'a>> {
        let id = self.cur?;
        self.cur = self.dom.parent_of(id);
        Some(self.dom.ref_(id))
    }
}

/// Explicit-stack pre-order walk of a node's descendants (self excluded);
/// same visit order as the recursive collect it replaced.
pub struct Descendants<'a> {
    dom: &'a Dom,
    /// Siblings still to visit (children are pushed in reverse, so the
    /// first child pops first).
    stack: Vec<NodeId>,
}

impl<'a> Descendants<'a> {
    fn new(dom: &'a Dom, id: NodeId) -> Self {
        Descendants {
            dom,
            stack: dom.children_of(id).iter().rev().copied().collect(),
        }
    }
}

impl<'a> Iterator for Descendants<'a> {
    type Item = NodeRef<'a>;

    fn next(&mut self) -> Option<NodeRef<'a>> {
        let id = self.stack.pop()?;
        let children = self.dom.children_of(id);
        self.stack.extend(children.iter().rev().copied());
        Some(self.dom.ref_(id))
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



// ---------------------------------------------------------------------------
// html5ever TreeSink that builds the arena directly: no markup5ever_rcdom
// tree, no copy pass. Semantics mirror RcDom exactly: text merged into a
// preceding text node on append/append_before_sibling, adjacent text runs
// re-merged after remove_from_parent/reparent_children (what the removed
// copy pass used to normalize), template contents kept in a detached
// fragment invisible to the converter, comments/PIs/doctypes as ignorable
// nodes, element/attr names lower-cased for the consumer while the original
// QualNames are kept beside the arena for the tree builder (foreign content
// needs the case, e.g. svg foreignObject integration points).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
struct H(NodeId);

struct SinkInner {
    nodes: Vec<Node>,
    /// original QualName per node (placeholder for non-elements, never read)
    names: Vec<QualName>,
    /// element id -> detached template-contents fragment (rare)
    tcontents: HashMap<NodeId, NodeId>,
    /// element id -> MathML annotation-xml integration point (rare)
    mip: HashMap<NodeId, bool>,
}

impl SinkInner {
    fn push_node(&mut self, kind: NodeKind) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node { kind, parent: None, children: Vec::new() });
        self.names.push(QualName::new(None, html5ever::ns!(), html5ever::local_name!("")));
        id
    }

    fn push_text(&mut self, text: StrTendril) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node { kind: NodeKind::Text(text.to_string()), parent: None, children: Vec::new() });
        self.names.push(QualName::new(None, html5ever::ns!(), html5ever::local_name!("")));
        id
    }
}

/// The html5ever 0.39 tree builder drives every [`TreeSink`] method through
/// `&self` while mutating the tree. That is sound here because the sink is
/// a `parse()`-local object whose only client is the parser's single-threaded
/// driver — no handle, reference or clone of it ever escapes, and `finish()`
/// consumes it — so the `UnsafeCell` just provides the `&mut` access that a
/// `&self` API hides; a `RefCell` would add per-access borrow bookkeeping to
/// a borrow pattern that is statically exclusive anyway.
struct ArenaSink {
    inner: UnsafeCell<SinkInner>,
}

impl ArenaSink {
    fn with_capacity(cap: usize) -> ArenaSink {
        let mut nodes = Vec::with_capacity(cap);
        nodes.push(Node { kind: NodeKind::Document, parent: None, children: Vec::new() });
        let mut names = Vec::with_capacity(cap);
        names.push(QualName::new(None, html5ever::ns!(), html5ever::local_name!("")));
        ArenaSink {
            inner: UnsafeCell::new(SinkInner {
                nodes,
                names,
                tcontents: HashMap::new(),
                mip: HashMap::new(),
            }),
        }
    }

    fn inner(&self) -> &SinkInner {
        unsafe { &*self.inner.get() }
    }

    fn mut_inner(&self) -> &mut SinkInner {
        unsafe { &mut *self.inner.get() }
    }
}

impl TreeSink for ArenaSink {
    type Output = Vec<Node>;
    type Handle = H;

    type ElemName<'a>
        = ExpandedName<'a>
    where
        Self: 'a;

    fn finish(self) -> Vec<Node> {
        self.inner.into_inner().nodes
    }

    fn parse_error(&self, _msg: std::borrow::Cow<'static, str>) {}

    fn get_document(&self) -> H {
        H(0)
    }

    fn set_quirks_mode(&self, _mode: QuirksMode) {}

    fn same_node(&self, x: &H, y: &H) -> bool {
        x == y
    }

    fn elem_name<'a>(&'a self, target: &'a H) -> ExpandedName<'a> {
        self.inner().names[target.0].expanded()
    }

    fn create_element(&self, name: QualName, attrs: Vec<Attribute>, flags: ElementFlags) -> H {
        let inner = self.mut_inner();
        let tag = lower_name(&name.local);
        let attrs: Vec<(String, String)> = attrs
            .into_iter()
            .map(|a| (lower_name(&a.name.local), a.value.to_string()))
            .collect();
        let id = inner.push_node(NodeKind::Element { tag, attrs });
        inner.names[id] = name;
        if flags.template {
            let tid = inner.push_node(NodeKind::Document);
            inner.tcontents.insert(id, tid);
        }
        if flags.mathml_annotation_xml_integration_point {
            inner.mip.insert(id, true);
        }
        H(id)
    }

    fn create_comment(&self, _text: StrTendril) -> H {
        H(self.mut_inner().push_node(NodeKind::Comment))
    }

    fn create_pi(&self, _target: StrTendril, _data: StrTendril) -> H {
        H(self.mut_inner().push_node(NodeKind::Comment))
    }

    fn append_doctype_to_document(
        &self,
        _name: StrTendril,
        _public_id: StrTendril,
        _system_id: StrTendril,
    ) {
        let inner = self.mut_inner();
        let id = inner.push_node(NodeKind::Comment);
        inner.nodes[id].parent = Some(0);
        inner.nodes[0].children.push(id);
    }

    fn append(&self, parent: &H, child: NodeOrText<H>) {
        let inner = self.mut_inner();
        let pid = parent.0;
        match child {
            NodeOrText::AppendText(text) => {
                if let Some(&last) = inner.nodes[pid].children.last() {
                    if let NodeKind::Text(t) = &mut inner.nodes[last].kind {
                        t.push_str(&text);
                        return;
                    }
                }
                let id = inner.push_text(text);
                inner.nodes[id].parent = Some(pid);
                inner.nodes[pid].children.push(id);
            }
            NodeOrText::AppendNode(h) => {
                debug_assert!(inner.nodes[h.0].parent.is_none());
                inner.nodes[h.0].parent = Some(pid);
                inner.nodes[pid].children.push(h.0);
            }
        }
    }

    fn append_before_sibling(&self, sibling: &H, child: NodeOrText<H>) {
        let inner = self.mut_inner();
        let sid = sibling.0;
        let (pid, i) = {
            let p = inner.nodes[sid].parent.expect("append_before_sibling: no parent");
            let idx = inner.nodes[p].children.iter().position(|&c| c == sid).expect("not a child");
            (p, idx)
        };
        match child {
            NodeOrText::AppendText(text) => {
                if i > 0 {
                    let prev = inner.nodes[pid].children[i - 1];
                    if let NodeKind::Text(t) = &mut inner.nodes[prev].kind {
                        t.push_str(&text);
                        return;
                    }
                }
                let id = inner.push_text(text);
                inner.nodes[id].parent = Some(pid);
                inner.nodes[pid].children.insert(i, id);
            }
            NodeOrText::AppendNode(h) => {
                if let Some(oldp) = inner.nodes[h.0].parent.take() {
                    if let Some(pos) = inner.nodes[oldp].children.iter().position(|&c| c == h.0) {
                        inner.nodes[oldp].children.remove(pos);
                    }
                }
                inner.nodes[h.0].parent = Some(pid);
                inner.nodes[pid].children.insert(i, h.0);
            }
        }
    }

    fn append_based_on_parent_node(&self, element: &H, prev_element: &H, child: NodeOrText<H>) {
        let has_parent = self.inner().nodes[element.0].parent.is_some();
        if has_parent {
            self.append_before_sibling(element, child);
        } else {
            self.append(prev_element, child);
        }
    }

    fn get_template_contents(&self, target: &H) -> H {
        H(*self.inner().tcontents.get(&target.0).expect("not a template element!"))
    }

    fn add_attrs_if_missing(&self, target: &H, attrs: Vec<Attribute>) {
        let inner = self.mut_inner();
        let NodeKind::Element { attrs: existing, .. } = &mut inner.nodes[target.0].kind else {
            panic!("not an element");
        };
        for a in attrs {
            let lname = lower_name(&a.name.local);
            if existing.iter().any(|(k, _)| *k == lname) {
                continue;
            }
            existing.push((lname, a.value.to_string()));
        }
    }

    fn remove_from_parent(&self, target: &H) {
        let inner = self.mut_inner();
        let Some(oldp) = inner.nodes[target.0].parent.take() else { return };
        if let Some(pos) = inner.nodes[oldp].children.iter().position(|&c| c == target.0) {
            inner.nodes[oldp].children.remove(pos);
        }
        merge_sibling_text(&mut inner.nodes, oldp);
    }

    fn reparent_children(&self, node: &H, new_parent: &H) {
        let inner = self.mut_inner();
        let children = std::mem::take(&mut inner.nodes[node.0].children);
        for &c in &children {
            inner.nodes[c].parent = Some(new_parent.0);
        }
        inner.nodes[new_parent.0].children.extend(children);
        merge_sibling_text(&mut inner.nodes, new_parent.0);
    }

    fn is_mathml_annotation_xml_integration_point(&self, handle: &H) -> bool {
        *self.inner().mip.get(&handle.0).unwrap_or(&false)
    }
}
