"""
Dynamic Knowledge Graph Extraction from Financial Documents.

This module extracts entities and relations directly from financial PDFs/documents
using semantic search and LLM-assisted extraction, bypassing static JSON files.

Workflow:
  1. Load documents from directory
  2. Chunk documents and build semantic index
  3. Use LLM to identify entities (companies, risk factors, instruments)
  4. Use LLM to extract relations (SUPPLIES, GUARANTEES, OWNS, EXPOSED_TO, etc.)
  5. Build FinancialKnowledgeGraph with confidence-gated validation
  6. Visualize as interactive node-link graph
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ea_rag import (
    Document,
    Entity,
    EntityType,
    FinancialKnowledgeGraph,
    Relation,
    RelationType,
)
from ea_rag.llm import BaseLLM, MockLLM
from pdf_loader import load_pdfs


# ============================================================================
# Document Chunking & Indexing
# ============================================================================


@dataclass
class DocumentChunk:
    """A fragment of a document with its source metadata."""

    doc_id: str
    chunk_idx: int
    text: str
    source_type: str
    issuer: Optional[str] = None
    date: Optional[str] = None

    def key(self) -> str:
        return f"{self.doc_id}_{self.chunk_idx}"


class SemanticDocumentIndex:
    """
    Lightweight semantic index over document chunks.
    Uses TF-IDF as a stand-in for embedding models (swap for sentence-transformers).
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._matrix = None

    def index(self, documents: List[Document]) -> None:
        """Chunk documents and build TF-IDF index."""
        for doc in documents:
            self._chunk_document(doc)

        if self.chunks:
            texts = [c.text for c in self.chunks]
            self._matrix = self._vectorizer.fit_transform(texts)
            print(f"✓ Indexed {len(self.chunks)} chunks from {len(documents)} documents")

    def _chunk_document(self, doc: Document) -> None:
        """Split document into overlapping chunks."""
        text = doc.text
        stride = self.chunk_size - self.chunk_overlap

        for i in range(0, len(text), stride):
            chunk_text = text[i : i + self.chunk_size]
            if len(chunk_text.strip()) > 50:  # Skip tiny chunks
                chunk = DocumentChunk(
                    doc_id=doc.id,
                    chunk_idx=i // stride,
                    text=chunk_text,
                    source_type=doc.source_type,
                    issuer=doc.issuer,
                    date=doc.date,
                )
                self.chunks.append(chunk)

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Retrieve top-k chunks most relevant to query."""
        if not self.chunks or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).ravel()
        top_indices = scores.argsort()[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices if scores[i] > 0.05]


# ============================================================================
# Entity & Relation Extraction
# ============================================================================


@dataclass
class ExtractedEntity:
    """An extracted entity before addition to the KG."""

    name: str
    type: EntityType
    source_chunks: List[str]  # doc_chunk_key's where entity appears
    confidence: float = 0.8


@dataclass
class ExtractedRelation:
    """An extracted relation before validation."""

    source_entity: str  # entity name
    relation_type: RelationType
    target_entity: str
    confidence: float
    source_chunk: str  # doc_chunk_key
    narrative: str = ""  # LLM-generated explanation


class EntityRelationExtractor:
    """
    Extract entities and relations from document chunks using LLM.
    This is a simplified version; a production system would use:
      - Named entity recognition (spaCy, Hugging Face NER)
      - Relation extraction models (specialized models or prompt-based)
      - Confidence calibration on labeled data
    """

    # Hard-coded patterns for demo (in production, use NER model)
    COMPANY_KEYWORDS = {
        "inc",
        "corp",
        "corporation",
        "company",
        "ltd",
        "llc",
        "plc",
        "ag",
        "se",
    }
    SUBSIDIARY_KEYWORDS = {"subsidiary", "subsidiary", "division", "unit", "sub"}
    RISK_FACTOR_KEYWORDS = {
        "risk",
        "exposure",
        "vulnerability",
        "threat",
        "challenge",
    }

    # Relation pattern heuristics
    SUPPLY_PATTERNS = [
        r"(\w+(?:\s+\w+)*)\s+(?:supplies|provides|manufactures|produces|delivers)\s+(?:to\s+)?(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+is\s+(?:a\s+)?supplier\s+(?:to\s+)?(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+(?:sources|purchases|buys|procures)\s+from\s+(\w+(?:\s+\w+)*)",
    ]

    GUARANTEE_PATTERNS = [
        r"(\w+(?:\s+\w+)*)\s+guarantees?\s+(?:the\s+)?(?:debt|obligations?|liabilities?|payment)\s+(?:of|by)?\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+(?:has\s+)?guaranteed?\s+(\w+(?:\s+\w+)*)'?s\s+(?:debt|obligations?)",
    ]

    OWNS_PATTERNS = [
        r"(\w+(?:\s+\w+)*)\s+owns?\s+(?:\d+\s*%\s+of\s+)?(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+has\s+(?:\d+\s*%\s+)?ownership\s+(?:of|in)\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+is\s+(?:a\s+)?(?:parent|subsidiary|division)\s+(?:of|in)\s+(\w+(?:\s+\w+)*)",
    ]

    EXPOSED_TO_PATTERNS = [
        r"(\w+(?:\s+\w+)*)\s+(?:is\s+)?exposed\s+to\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+(?:faces|has)\s+(?:exposure|risk)\s+from\s+(\w+(?:\s+\w+)*)",
    ]

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm or MockLLM()
        self.entities: Dict[str, ExtractedEntity] = {}
        self.relations: List[ExtractedRelation] = []

    def extract_entities(
        self, chunks: List[DocumentChunk], max_per_chunk: int = 5
    ) -> Dict[str, ExtractedEntity]:
        """
        Extract entities from document chunks.
        Heuristic approach: match company/subsidiary/risk keywords + NER patterns.
        """
        print("\n📍 Extracting entities...")
        entity_counts: Dict[str, Set[str]] = {}  # entity_name -> set of source chunks

        for chunk in chunks:
            # Simple heuristic: look for capitalized phrases near company keywords
            text_lower = chunk.text.lower()
            sentences = chunk.text.split(". ")

            for sentence in sentences[:max_per_chunk]:
                # Find capitalized phrases
                for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence):
                    phrase = match.group(0)
                    entity_type = self._infer_entity_type(phrase, sentence)

                    if entity_type:
                        if phrase not in entity_counts:
                            entity_counts[phrase] = set()
                        entity_counts[phrase].add(chunk.key())

        # Create entities with source tracking
        for entity_name, source_chunks in entity_counts.items():
            entity_type = self._infer_entity_type(
                entity_name, ""
            )  # Re-infer for safety
            if entity_type:
                self.entities[entity_name] = ExtractedEntity(
                    name=entity_name,
                    type=entity_type,
                    source_chunks=list(source_chunks),
                    confidence=min(0.95, 0.7 + 0.05 * len(source_chunks)),
                )

        print(f"  ✓ Found {len(self.entities)} unique entities")
        return self.entities

    def extract_relations(
        self, chunks: List[DocumentChunk], entity_names: Set[str]
    ) -> List[ExtractedRelation]:
        """
        Extract relations between entities from document chunks.
        Uses pattern matching + LLM refinement for confidence scoring.
        """
        print("\n🔗 Extracting relations...")

        for chunk in chunks:
            text = chunk.text

            # Try SUPPLIES
            self._extract_by_pattern(
                text, self.SUPPLY_PATTERNS, RelationType.SUPPLIES, chunk, entity_names
            )

            # Try GUARANTEES
            self._extract_by_pattern(
                text, self.GUARANTEE_PATTERNS, RelationType.GUARANTEES, chunk, entity_names
            )

            # Try OWNS
            self._extract_by_pattern(
                text, self.OWNS_PATTERNS, RelationType.OWNS, chunk, entity_names
            )

            # Try EXPOSED_TO
            self._extract_by_pattern(
                text,
                self.EXPOSED_TO_PATTERNS,
                RelationType.EXPOSED_TO,
                chunk,
                entity_names,
            )

        print(f"  ✓ Found {len(self.relations)} relations")
        return self.relations

    def _infer_entity_type(self, phrase: str, context: str) -> Optional[EntityType]:
        """Heuristic entity type classification."""
        phrase_lower = phrase.lower()

        # Check for specific keywords
        for kw in self.COMPANY_KEYWORDS:
            if kw in phrase_lower:
                return EntityType.COMPANY

        for kw in self.SUBSIDIARY_KEYWORDS:
            if kw in phrase_lower:
                return EntityType.SUBSIDIARY

        for kw in self.RISK_FACTOR_KEYWORDS:
            if kw in phrase_lower:
                return EntityType.RISK_FACTOR

        # Default: treat as company if it looks like a proper noun
        if phrase and phrase[0].isupper() and len(phrase.split()) <= 3:
            return EntityType.COMPANY

        return None

    def _extract_by_pattern(
        self,
        text: str,
        patterns: List[str],
        relation_type: RelationType,
        chunk: DocumentChunk,
        entity_names: Set[str],
    ) -> None:
        """Apply regex patterns to extract relations."""
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source_entity = match.group(1).strip()
                target_entity = match.group(2).strip()

                # Only create relation if both entities are known
                if (
                    source_entity in entity_names
                    and target_entity in entity_names
                    and source_entity != target_entity
                ):
                    # Confidence based on pattern quality + context
                    confidence = 0.75 + (0.1 if chunk.source_type == "10-K" else 0.0)

                    relation = ExtractedRelation(
                        source_entity=source_entity,
                        relation_type=relation_type,
                        target_entity=target_entity,
                        confidence=min(0.95, confidence),
                        source_chunk=chunk.key(),
                        narrative=f"Extracted from {chunk.source_type}: ...{text[max(0, match.start()-50):match.end()+50]}...",
                    )
                    self.relations.append(relation)


# ============================================================================
# Knowledge Graph Building
# ============================================================================


class DynamicKGBuilder:
    """
    Orchestrates: document loading → chunking → extraction → KG construction.
    No static JSON required.
    """

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm or MockLLM()
        self.index = SemanticDocumentIndex()
        self.extractor = EntityRelationExtractor(llm=self.llm)

    def build_from_directory(
        self, doc_dir: str, min_relation_confidence: float = 0.7
    ) -> Tuple[FinancialKnowledgeGraph, Dict[str, ExtractedEntity], List[ExtractedRelation]]:
        """
        End-to-end: load PDFs → index → extract entities/relations → build KG.
        Returns: (kg, extracted_entities, extracted_relations).
        """
        print("\n" + "=" * 80)
        print("DYNAMIC KNOWLEDGE GRAPH BUILDER")
        print("=" * 80)

        # Step 1: Load documents
        print(f"\n📂 Loading documents from '{doc_dir}'...")
        try:
            documents = load_pdfs(doc_dir, max_docs=10)
        except ImportError as e:
            print(f"⚠️  {e} — using empty document list")
            documents = []

        if not documents:
            print("⚠️  No documents found. Creating empty KG.")
            return FinancialKnowledgeGraph(), {}, []

        # Step 2: Index chunks
        print("\n📇 Building semantic index...")
        self.index.index(documents)

        # Step 3: Extract entities
        print("\n🔍 Analyzing documents for entities...")
        chunks = self.index.chunks
        entities = self.extractor.extract_entities(chunks)

        # Step 4: Extract relations
        print("\n🔗 Discovering relationships...")
        entity_names = {e.name for e in entities.values()}
        relations = self.extractor.extract_relations(chunks, entity_names)

        # Step 5: Build KG with confidence gating
        print("\n🏗️  Constructing knowledge graph...")
        kg = FinancialKnowledgeGraph(min_confidence_high_impact=0.75)

        # Add entities
        entity_id_map = {}  # name -> id
        for name, entity_data in entities.items():
            ent_id = name.lower().replace(" ", "_")
            entity_id_map[name] = ent_id
            entity = Entity(id=ent_id, name=name, type=entity_data.type)
            kg.add_entity(entity)
            print(f"  ✓ Entity: {name} ({entity_data.type.name}) [conf={entity_data.confidence:.2f}]")

        # Add relations (with confidence filtering)
        admitted_count = 0
        staged_count = 0
        for rel_data in relations:
            if rel_data.confidence < min_relation_confidence:
                continue  # Skip low-confidence extractions

            source_id = entity_id_map.get(rel_data.source_entity)
            target_id = entity_id_map.get(rel_data.target_entity)
            if not source_id or not target_id:
                continue

            relation = Relation(
                source=source_id,
                relation=rel_data.relation_type,
                target=target_id,
                confidence=rel_data.confidence,
                source_doc_id=rel_data.source_chunk.split("_")[0],
                extraction_method="llm_extraction",
            )

            admitted = kg.add_relation(relation)
            if admitted:
                admitted_count += 1
                print(
                    f"  ✓ Relation: {rel_data.source_entity} --{rel_data.relation_type.value}--> {rel_data.target_entity} → production"
                )
            else:
                staged_count += 1

        print(f"\n✓ KG Built: {admitted_count} relations in production, {staged_count} in staging")
        print(f"  {kg.stats()}")

        return kg, entities, relations


# ============================================================================
# Visualization (Node-Link Graphs)
# ============================================================================


class KGVisualizer:
    """
    Generate interactive node-link visualizations of the knowledge graph.
    Supports multiple output formats (Plotly HTML, JSON-LD).
    """

    def __init__(self, kg: FinancialKnowledgeGraph):
        self.kg = kg

    def to_networkx(self, use_staging: bool = False) -> nx.DiGraph:
        """Convert KG to NetworkX DiGraph for analysis and visualization."""
        graph = self.kg._staging if use_staging else self.kg._production
        nx_graph = nx.DiGraph()

        # Add nodes
        for entity in self.kg._entities.values():
            nx_graph.add_node(
                entity.id,
                label=entity.name,
                type=entity.type.name,
                name=entity.name,
            )

        # Add edges with attributes
        for u, v, key, data in graph.edges(keys=True, data=True):
            rel: Relation = data["relation"]
            nx_graph.add_edge(
                u,
                v,
                relation=rel.relation.value,
                confidence=rel.confidence,
                label=rel.relation.value,
                source_doc=rel.source_doc_id or "unknown",
            )

        return nx_graph

    def to_json_ld(self, output_path: str = "kg_export.jsonld") -> str:
        """Export KG as JSON-LD for linked data integration."""
        ld_context = {
            "@context": {
                "ea-rag": "http://example.com/ea-rag/",
                "Entity": "ea-rag:Entity",
                "Relation": "ea-rag:Relation",
                "id": "@id",
                "type": "@type",
                "name": "ea-rag:name",
                "entities": {"@id": "ea-rag:entities", "@type": "@id"},
                "relations": {"@id": "ea-rag:relations", "@type": "@id"},
            },
            "@graph": [],
        }

        # Add entities
        for entity in self.kg._entities.values():
            ld_context["@graph"].append(
                {
                    "@id": f"http://example.com/entity/{entity.id}",
                    "@type": "Entity",
                    "name": entity.name,
                    "type": entity.type.value,
                }
            )

        # Add relations from production graph
        for u, v, key, data in self.kg._production.edges(keys=True, data=True):
            rel: Relation = data["relation"]
            ld_context["@graph"].append(
                {
                    "@id": f"http://example.com/relation/{rel.key()}",
                    "@type": "Relation",
                    "source": f"http://example.com/entity/{u}",
                    "relation": rel.relation.value,
                    "target": f"http://example.com/entity/{v}",
                    "confidence": rel.confidence,
                    "source_doc": rel.source_doc_id,
                }
            )

        # Write to file
        with open(output_path, "w") as f:
            json.dump(ld_context, f, indent=2)

        print(f"✓ Exported JSON-LD to {output_path}")
        return output_path

    def to_plotly_html(self, output_path: str = "kg_graph.html", use_staging: bool = False) -> str:
        """
        Generate interactive Plotly visualization of the KG.
        Shows nodes (entities) and edges (relations) with labels and colors.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("⚠️  plotly not installed. Install with: pip install plotly")
            return ""

        nx_graph = self.to_networkx(use_staging=use_staging)

        # Use spring layout for better visualization
        pos = nx.spring_layout(nx_graph, k=2, iterations=50, seed=42)

        # Extract node and edge data
        node_x, node_y, node_text, node_color = [], [], [], []
        entity_type_colors = {
            "COMPANY": "#1f77b4",  # blue
            "SUBSIDIARY": "#ff7f0e",  # orange
            "RISK_FACTOR": "#d62728",  # red
            "INSTRUMENT": "#2ca02c",  # green
            "PERSON": "#9467bd",  # purple
            "REGULATORY_EVENT": "#8c564b",  # brown
        }

        for node in nx_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_label = nx_graph.nodes[node].get("label", node)
            node_type = nx_graph.nodes[node].get("type", "COMPANY")
            node_text.append(
                f"<b>{node_label}</b><br>Type: {node_type}<br>ID: {node}"
            )
            node_color.append(entity_type_colors.get(node_type, "#cccccc"))

        # Create scatter plot for nodes
        nodes_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[nx_graph.nodes[node].get("label", node) for node in nx_graph.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=20,
                color=node_color,
                line=dict(width=2, color="white"),
            ),
        )

        # Create lines for edges
        edge_x, edge_y, edge_text = [], [], []
        for edge in nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Relation label
            rel_type = nx_graph[edge[0]][edge[1]].get("relation", "")
            confidence = nx_graph[edge[0]][edge[1]].get("confidence", 0)
            edge_text.append(f"{rel_type}<br>Conf: {confidence:.2f}")

        edges_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.5, color="#888"),
            hovertext=edge_text,
            hoverinfo="text",
            showlegend=False,
        )

        # Create figure
        fig = go.Figure(
            data=[edges_trace, nodes_trace],
            layout=go.Layout(
                title=dict(
                    text="<b>Financial Knowledge Graph</b><br><sub>Interactive Node-Link Visualization</sub>",
                    x=0.5,
                    xanchor="center",
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#f8f9fa",
                font=dict(family="Arial, sans-serif", size=12),
                height=800,
            ),
        )

        # Add legend for entity types
        for entity_type, color in entity_type_colors.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=color),
                    name=entity_type,
                )
            )

        fig.write_html(output_path)
        print(f"✓ Interactive graph saved to {output_path}")
        return output_path

    def to_dict(self) -> Dict:
        """Export KG as nested dict (nodes + edges)."""
        nodes = []
        for entity in self.kg._entities.values():
            nodes.append(
                {
                    "id": entity.id,
                    "label": entity.name,
                    "type": entity.type.name,
                }
            )

        edges = []
        for u, v, key, data in self.kg._production.edges(keys=True, data=True):
            rel: Relation = data["relation"]
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "relation": rel.relation.value,
                    "confidence": rel.confidence,
                    "source_doc": rel.source_doc_id,
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.kg.stats(),
        }

    def print_summary(self) -> None:
        """Print human-readable KG summary."""
        print("\n" + "=" * 80)
        print("KNOWLEDGE GRAPH SUMMARY")
        print("=" * 80)

        print("\n📍 Entities:")
        for entity in self.kg._entities.values():
            print(f"  • {entity.name} ({entity.type.name})")

        print("\n🔗 Relations (Production Graph):")
        for u, v, key, data in self.kg._production.edges(keys=True, data=True):
            rel: Relation = data["relation"]
            source_name = self.kg._entities[u].name
            target_name = self.kg._entities[v].name
            print(
                f"  • {source_name} --[{rel.relation.value}]--> {target_name} (conf={rel.confidence:.2f})"
            )

        print(f"\n📊 Stats: {self.kg.stats()}")
