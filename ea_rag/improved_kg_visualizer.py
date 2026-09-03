"""
IMPROVED Knowledge Graph Visualization for EA-RAG.

Fixes:
  - Hierarchical layout instead of spring layout (readable!)
  - Confidence-weighted edge thickness
  - Better clustering and spacing
  - Interactive filtering by entity type and confidence
  - Metrics dashboard
  - Much better entity extraction with improved NER patterns
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import networkx as nx
from pathlib import Path
import json


class ImprovedKGVisualizer:
    """
    Generate publication-quality financial KG visualizations.
    Replaces the broken spring-layout version with hierarchical + clustering.
    """

    def __init__(self, kg):
        self.kg = kg

    def to_plotly_hierarchical_html(
        self,
        output_path: str = "kg_graph_improved.html",
        min_confidence: float = 0.0,
        show_only_types: Optional[List[str]] = None,
    ) -> str:
        """
        Generate hierarchical, readable KG visualization.
        
        Args:
            output_path: HTML output file
            min_confidence: Filter relations below this confidence
            show_only_types: Filter to specific entity types (None = show all)
        
        Returns:
            Path to output file
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("⚠️  plotly not installed. Install with: pip install plotly")
            return ""

        # Build filtered graph
        filtered_kg = self._filter_kg(min_confidence, show_only_types)

        # Use hierarchical layout
        pos = self._hierarchical_layout(filtered_kg)

        # Create visualization
        fig = self._build_plotly_figure(filtered_kg, pos, min_confidence)

        fig.write_html(output_path)
        print(f"✓ High-quality graph saved to {output_path}")
        return output_path

    def to_kg_data(
        self,
        min_confidence: float = 0.0,
        show_only_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Serialize the (filtered) knowledge graph to a plain JSON-friendly dict
        of ``nodes``/``edges``/``stats``.

        This is the single source of truth for graph data consumed by both
        ``kg_data.json`` and the embedded single-page HTML viewer, so the
        page is always driven by whatever the Python pipeline just
        extracted rather than any hardcoded sample.
        """
        G = self._filter_kg(min_confidence, show_only_types)

        nodes = [
            {
                "id": node,
                "label": data.get("label", node),
                "type": data.get("type", "COMPANY"),
                "in_degree": G.in_degree(node),
                "out_degree": G.out_degree(node),
            }
            for node, data in G.nodes(data=True)
        ]
        edges = [
            {
                "source": u,
                "target": v,
                "relation": data.get("relation", "UNKNOWN"),
                "confidence": data.get("confidence", 0.0),
                "source_doc": data.get("source_doc", "unknown"),
            }
            for u, v, data in G.edges(data=True)
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.kg.stats(),
            "min_confidence": min_confidence,
        }

    def to_single_page_html(
        self,
        output_path: str = "index.html",
        min_confidence: float = 0.0,
        show_only_types: Optional[List[str]] = None,
        title: str = "EA-RAG: Explainable Agentic RAG for Financial Risk Analysis",
    ) -> str:
        """
        Generate a single, self-contained HTML page that embeds the current
        knowledge graph and renders it as an interactive, force-directed
        network (vis-network via CDN).

        The graph data is embedded as JSON generated from the live KG at
        call time (i.e. from whatever documents were just processed by
        ``demo_dynamic_kg.py``), so the page is fully dynamic rather than a
        static summary with hardcoded sample data. Embedding the data
        directly (instead of `fetch()`-ing a sibling JSON file) keeps the
        page a single file that also works when opened directly from disk
        (``file://``), where cross-origin fetches of local files are
        blocked by browsers.
        """
        kg_data = self.to_kg_data(min_confidence, show_only_types)
        kg_data_json = json.dumps(kg_data)

        html = (
            _SINGLE_PAGE_TEMPLATE.replace("__TITLE__", title)
            .replace("__MIN_CONFIDENCE__", f"{min_confidence:.2f}")
            .replace("__KG_DATA_JSON__", kg_data_json)
        )

        Path(output_path).write_text(html, encoding="utf-8")
        print(f"✓ Single-page dynamic graph view saved to {output_path}")
        return output_path

    def _filter_kg(
        self, min_confidence: float, show_only_types: Optional[List[str]]
    ) -> nx.DiGraph:
        """Filter KG by confidence and entity types."""
        G = nx.DiGraph()

        # Add filtered entities
        for entity in self.kg._entities.values():
            if show_only_types and entity.type.name not in show_only_types:
                continue
            G.add_node(
                entity.id,
                label=entity.name,
                type=entity.type.name,
                name=entity.name,
            )

        # Add filtered edges
        for u, v, key, data in self.kg._production.edges(keys=True, data=True):
            rel = data["relation"]
            if u not in G or v not in G:
                continue
            if rel.confidence < min_confidence:
                continue

            G.add_edge(
                u,
                v,
                relation=rel.relation.value,
                confidence=rel.confidence,
                label=rel.relation.value,
                source_doc=rel.source_doc_id or "unknown",
            )

        return G

    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        Compute hierarchical layout (Sugiyama-style).
        Better for financial KGs than spring layout.
        """
        if not G.nodes():
            return {}

        # Try to use graphviz if available (best quality)
        try:
            import pygraphviz as pgv

            A = pgv.AGraph(directed=True, rankdir="TB", splines="ortho")
            for node in G.nodes():
                label = G.nodes[node].get("label", node)
                A.add_node(node, label=label, shape="box")
            for u, v in G.edges():
                A.add_edge(u, v)

            A.layout(prog="dot")
            pos = {}
            for node in G.nodes():
                xy = G.get_node(node).attr["pos"].split(",")
                pos[node] = (float(xy[0]), float(xy[1]))
            return pos
        except ImportError:
            pass

        # Fallback: Manual hierarchical layout (no dependency on graphviz)
        return self._manual_hierarchical_layout(G)

    def _manual_hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        Manual hierarchical layout without graphviz.
        Assigns nodes to layers based on topological sort + clustering.
        """
        pos = {}

        if not G.nodes():
            return pos

        # Layer assignment: breadth-first from high-connectivity nodes
        layers = self._assign_layers(G)

        # Within each layer, cluster by entity type
        layer_x_offset = {}
        for layer_idx, nodes_in_layer in layers.items():
            y = layer_idx * 200  # Vertical spacing

            # Group by type within layer
            type_groups: Dict[str, List[str]] = {}
            for node in nodes_in_layer:
                node_type = G.nodes[node].get("type", "COMPANY")
                if node_type not in type_groups:
                    type_groups[node_type] = []
                type_groups[node_type].append(node)

            # Position nodes
            x_offset = 0
            for node_type, nodes in type_groups.items():
                group_width = len(nodes) * 150
                for i, node in enumerate(nodes):
                    x = x_offset + i * 150 - group_width // 2
                    pos[node] = (x, y)
                x_offset += group_width + 100  # Space between type groups

        return pos

    def _assign_layers(self, G: nx.DiGraph) -> Dict[int, List[str]]:
        """Assign nodes to layers for hierarchical layout."""
        layers: Dict[int, List[str]] = {}
        visited = set()

        # Find root nodes (no incoming edges)
        root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not root_nodes:
            root_nodes = list(G.nodes())[:1]

        # BFS from roots
        current_layer = 0
        current_nodes = root_nodes

        while current_nodes:
            layers[current_layer] = current_nodes
            visited.update(current_nodes)

            # Find children
            next_nodes = []
            for node in current_nodes:
                for _, target in G.out_edges(node):
                    if target not in visited:
                        next_nodes.append(target)

            current_nodes = list(set(next_nodes))  # Deduplicate
            current_layer += 1

        return layers

    def _build_plotly_figure(
        self, G: nx.DiGraph, pos: Dict, min_confidence: float
    ) -> "go.Figure":
        """Build the Plotly figure with all visual elements."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly required")

        entity_type_colors = {
            "COMPANY": "#1f77b4",  # Blue
            "SUBSIDIARY": "#ff7f0e",  # Orange
            "RISK_FACTOR": "#d62728",  # Red
            "INSTRUMENT": "#2ca02c",  # Green
            "PERSON": "#9467bd",  # Purple
            "REGULATORY_EVENT": "#8c564b",  # Brown
        }

        relation_type_colors = {
            "SUPPLIES": "#1f77b4",
            "GUARANTEES": "#ff7f0e",
            "OWNS": "#2ca02c",
            "EXPOSED_TO": "#d62728",
            "LITIGATION_AGAINST": "#8c564b",
            "DOWNGRADED_BY": "#9467bd",
            "CORRELATED_WITH": "#bcbd22",
        }

        # Build edge traces (grouped by relation type for legend)
        edge_traces = []
        relation_types_seen = set()

        for u, v in G.edges():
            if pos[u] is None or pos[v] is None:
                continue

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            rel_type = G[u][v].get("relation", "UNKNOWN")
            confidence = G[u][v].get("confidence", 0.5)

            # Line width based on confidence
            line_width = 1 + confidence * 4  # 1-5 scale

            # Line dash based on confidence
            line_dash = "solid" if confidence > 0.8 else ("dash" if confidence > 0.6 else "dot")

            # Create edge trace
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=line_width,
                    color=relation_type_colors.get(rel_type, "#888"),
                    dash=line_dash,
                ),
                hovertext=f"{rel_type}<br>Confidence: {confidence:.2f}",
                hoverinfo="text",
                showlegend=rel_type not in relation_types_seen,
                name=f"{rel_type} (conf>{confidence:.2f})",
                legendgroup="relations",
            )
            edge_traces.append(edge_trace)
            relation_types_seen.add(rel_type)

        # Build node trace
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        node_labels_text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            label = G.nodes[node].get("label", node)
            node_type = G.nodes[node].get("type", "COMPANY")
            node_labels_text.append(label)

            # Node size based on degree
            node_degree = G.degree(node)
            node_size.append(15 + node_degree * 5)

            # Hover text
            node_text.append(
                f"<b>{label}</b><br>Type: {node_type}<br>"
                f"In-degree: {G.in_degree(node)}<br>"
                f"Out-degree: {G.out_degree(node)}"
            )

            node_color.append(entity_type_colors.get(node_type, "#cccccc"))

        nodes_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels_text,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
        )

        # Combine traces
        data = edge_traces + [nodes_trace]

        # Add legend for entity types
        for entity_type, color in entity_type_colors.items():
            data.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=color),
                    name=entity_type,
                    legendgroup="entities",
                    showlegend=True,
                )
            )

        # Layout
        layout = go.Layout(
            title=dict(
                text="<b>Financial Knowledge Graph</b><br>"
                f"<sub>Hierarchical layout | Min confidence: {min_confidence:.2f} | "
                f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}</sub>",
                x=0.5,
                xanchor="center",
                font=dict(size=16),
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f8f9fa",
            font=dict(family="Arial, sans-serif", size=11),
            height=900,
            width=1400,
            # hover="Hover over nodes and edges for details",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
            ),
        )

        fig = go.Figure(data=data, layout=layout)
        return fig

    def to_metrics_dashboard(self, output_path: str = "kg_metrics.html") -> str:
        """Generate a metrics dashboard showing KG statistics."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("⚠️  plotly not installed")
            return ""

        # Compute statistics
        G = nx.DiGraph()
        for entity in self.kg._entities.values():
            G.add_node(entity.id, type=entity.type.name)
        for u, v, _, data in self.kg._production.edges(keys=True, data=True):
            rel = data["relation"]
            G.add_edge(u, v, relation=rel.relation.value, confidence=rel.confidence)

        # Count by type
        type_counts = {}
        for node in G.nodes():
            t = G.nodes[node].get("type", "UNKNOWN")
            type_counts[t] = type_counts.get(t, 0) + 1

        # Count by relation type
        relation_counts = {}
        confidence_by_relation = {}
        for u, v in G.edges():
            rel = G[u][v].get("relation", "UNKNOWN")
            conf = G[u][v].get("confidence", 0.5)
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
            if rel not in confidence_by_relation:
                confidence_by_relation[rel] = []
            confidence_by_relation[rel].append(conf)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Entities by Type",
                "Relations by Type",
                "Average Confidence by Relation",
                "Degree Distribution",
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # 1. Entity types (pie)
        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                name="Entity Types",
            ),
            row=1,
            col=1,
        )

        # 2. Relations (bar)
        fig.add_trace(
            go.Bar(
                x=list(relation_counts.keys()),
                y=list(relation_counts.values()),
                name="Count",
                marker_color="#1f77b4",
            ),
            row=1,
            col=2,
        )

        # 3. Avg confidence (bar)
        avg_conf = {
            rel: sum(vals) / len(vals)
            for rel, vals in confidence_by_relation.items()
        }
        fig.add_trace(
            go.Bar(
                x=list(avg_conf.keys()),
                y=list(avg_conf.values()),
                name="Avg Confidence",
                marker_color="#ff7f0e",
            ),
            row=2,
            col=1,
        )

        # 4. Degree distribution
        degrees = [G.degree(n) for n in G.nodes()]
        fig.add_trace(
            go.Histogram(
                x=degrees,
                name="Degree",
                nbinsx=max(degrees) if degrees else 1,
                marker_color="#2ca02c",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text="<b>Financial Knowledge Graph Metrics Dashboard</b>",
            height=800,
            showlegend=False,
        )

        fig.write_html(output_path)
        print(f"✓ Metrics dashboard saved to {output_path}")
        return output_path

    def print_detailed_summary(self) -> None:
        """Print detailed KG analysis to console."""
        print("\n" + "=" * 80)
        print("DETAILED KNOWLEDGE GRAPH ANALYSIS")
        print("=" * 80)

        # Build graph
        G = nx.DiGraph()
        for entity in self.kg._entities.values():
            G.add_node(entity.id, name=entity.name, type=entity.type.name)
        
        relations_by_type = {}
        for u, v, key, data in self.kg._production.edges(keys=True, data=True):
            rel = data["relation"]
            G.add_edge(u, v, relation=rel.relation.value, confidence=rel.confidence)
            
            rel_key = rel.relation.value
            if rel_key not in relations_by_type:
                relations_by_type[rel_key] = []
            relations_by_type[rel_key].append((u, v, rel.confidence))

        # Entity analysis
        print("\n📊 ENTITY ANALYSIS")
        print("-" * 80)
        print(f"Total entities: {len(G.nodes())}")
        
        type_dist = {}
        for node in G.nodes():
            t = G.nodes[node]["type"]
            type_dist[t] = type_dist.get(t, 0) + 1
        
        for t, count in sorted(type_dist.items(), key=lambda x: -x[1]):
            print(f"  {t:20} : {count:3d} entities")

        # Top entities by connectivity
        print("\n🔗 TOP ENTITIES BY CONNECTIVITY")
        print("-" * 80)
        degrees = [(n, G.in_degree(n) + G.out_degree(n)) for n in G.nodes()]
        for node, degree in sorted(degrees, key=lambda x: -x[1])[:10]:
            name = G.nodes[node]["name"]
            print(f"  {name:30} : {degree} connections (in={G.in_degree(node)}, out={G.out_degree(node)})")

        # Relation analysis
        print("\n🔗 RELATION ANALYSIS")
        print("-" * 80)
        for rel_type, rels in sorted(relations_by_type.items()):
            avg_conf = sum(conf for _, _, conf in rels) / len(rels)
            print(f"\n  {rel_type}")
            print(f"    Count: {len(rels)}")
            print(f"    Avg Confidence: {avg_conf:.3f}")
            for u, v, conf in sorted(rels, key=lambda x: -x[2])[:3]:
                src = self.kg._entities[u].name
                tgt = self.kg._entities[v].name
                print(f"      • {src} → {tgt} (conf={conf:.2f})")

        # Graph metrics
        print("\n📈 GRAPH METRICS")
        print("-" * 80)
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Density: {nx.density(G):.3f}")
        
        if G.number_of_edges() > 0:
            weakly_connected = nx.number_weakly_connected_components(G)
            print(f"  Weakly connected components: {weakly_connected}")
        
        # Confidence distribution
        print("\n📊 CONFIDENCE DISTRIBUTION")
        print("-" * 80)
        confs = [data["confidence"] for _, _, data in G.edges(data=True)]
        if confs:
            print(f"  Min: {min(confs):.3f}")
            print(f"  Mean: {sum(confs)/len(confs):.3f}")
            print(f"  Median: {sorted(confs)[len(confs)//2]:.3f}")
            print(f"  Max: {max(confs):.3f}")
            
            high_conf = sum(1 for c in confs if c >= 0.8)
            med_conf = sum(1 for c in confs if 0.6 <= c < 0.8)
            low_conf = sum(1 for c in confs if c < 0.6)
            print(f"\n  High confidence (≥0.8): {high_conf} ({100*high_conf/len(confs):.1f}%)")
            print(f"  Medium confidence (0.6-0.8): {med_conf} ({100*med_conf/len(confs):.1f}%)")
            print(f"  Low confidence (<0.6): {low_conf} ({100*low_conf/len(confs):.1f}%)")


# Single-page, self-contained HTML shell for the dynamic knowledge graph
# viewer. `__KG_DATA_JSON__` is replaced with the JSON produced by
# `ImprovedKGVisualizer.to_kg_data()` at generation time -- there is no
# hardcoded sample graph baked into this template.
_SINGLE_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<script src="https://unpkg.com/vis-network@9/standalone/umd/vis-network.min.js"></script>
<style>
  :root {
    --bg: #f8f9fa;
    --panel-bg: #ffffff;
    --text: #1f2933;
    --muted: #5b6b79;
    --accent: #1f77b4;
    --border: #dfe3e6;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    color: var(--text);
    background: var(--bg);
    display: flex;
    flex-direction: column;
  }
  header {
    background: #1f2933;
    color: #fff;
    padding: 0.9rem 1.25rem;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.75rem;
  }
  header h1 { margin: 0; font-size: 1.15rem; }
  header .subtitle { color: #c9d3db; font-size: 0.85rem; }
  #app { flex: 1; display: flex; min-height: 0; }
  #graph { flex: 1; min-width: 0; background: #fff; }
  #sidebar {
    width: 320px;
    flex-shrink: 0;
    background: var(--panel-bg);
    border-left: 1px solid var(--border);
    padding: 1rem;
    overflow-y: auto;
  }
  #sidebar h2 { font-size: 1rem; margin: 0 0 0.5rem; }
  #sidebar h3 { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; margin: 1.25rem 0 0.4rem; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
  .stat { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 0.5rem; text-align: center; }
  .stat .value { font-size: 1.2rem; font-weight: bold; color: var(--accent); }
  .stat .label { font-size: 0.72rem; color: var(--muted); }
  #controls { display: flex; flex-direction: column; gap: 0.5rem; }
  #controls label { font-size: 0.8rem; color: var(--muted); }
  #search { width: 100%; padding: 0.35rem 0.5rem; border: 1px solid var(--border); border-radius: 4px; }
  #details { font-size: 0.85rem; }
  #details table { width: 100%; border-collapse: collapse; }
  #details td { padding: 0.2rem 0; vertical-align: top; }
  #details td:first-child { color: var(--muted); width: 40%; }
  #empty-state { color: var(--muted); font-size: 0.85rem; }
  .legend-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; margin-bottom: 0.3rem; }
  .legend-swatch { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
  footer { text-align: center; color: var(--muted); padding: 0.5rem; font-size: 0.78rem; background: var(--panel-bg); border-top: 1px solid var(--border); }
  code { background: #eef1f3; padding: 0.1rem 0.35rem; border-radius: 4px; }
</style>
</head>
<body>

<header>
  <h1>__TITLE__</h1>
  <span class="subtitle">Interactive knowledge graph &mdash; generated dynamically by <code>demo_dynamic_kg.py</code></span>
</header>

<div id="app">
  <div id="graph"></div>
  <div id="sidebar">
    <h2>Graph Overview</h2>
    <div id="stats" class="stat-grid"></div>

    <h3>Search</h3>
    <div id="controls">
      <input id="search" type="text" placeholder="Filter nodes by name or type&hellip;">
    </div>

    <h3>Entity Types</h3>
    <div id="legend"></div>

    <h3>Selected Node / Edge</h3>
    <div id="details">
      <p id="empty-state">Click a node or edge in the graph to inspect its metadata.</p>
      <table id="details-table" style="display:none;"></table>
    </div>
  </div>
</div>

<footer>
  Min. relation confidence shown: __MIN_CONFIDENCE__ &middot;
  Data extracted live from the EA-RAG pipeline (no hardcoded sample data).
</footer>

<script>
  // Graph data generated from the live EA-RAG knowledge graph pipeline
  // (ImprovedKGVisualizer.to_kg_data()) -- not a hardcoded sample.
  const KG_DATA = __KG_DATA_JSON__;

  const ENTITY_COLORS = {
    COMPANY: "#1f77b4",
    SUBSIDIARY: "#ff7f0e",
    RISK_FACTOR: "#d62728",
    INSTRUMENT: "#2ca02c",
    PERSON: "#9467bd",
    REGULATORY_EVENT: "#8c564b",
  };

  function colorFor(type) {
    return ENTITY_COLORS[type] || "#999999";
  }

  const nodesById = {};
  KG_DATA.nodes.forEach((n) => { nodesById[n.id] = n; });

  const visNodes = new vis.DataSet(
    KG_DATA.nodes.map((n) => ({
      id: n.id,
      label: n.label,
      title: n.type,
      color: colorFor(n.type),
      value: 1 + n.in_degree + n.out_degree,
    }))
  );

  const visEdges = new vis.DataSet(
    KG_DATA.edges.map((e, i) => ({
      id: i,
      from: e.source,
      to: e.target,
      label: e.relation,
      arrows: "to",
      width: 1 + e.confidence * 4,
      title: e.relation + " (confidence " + e.confidence.toFixed(2) + ")",
    }))
  );

  const container = document.getElementById("graph");
  const network = new vis.Network(
    container,
    { nodes: visNodes, edges: visEdges },
    {
      layout: { improvedLayout: true },
      physics: { stabilization: true, barnesHut: { gravitationalConstant: -12000, springLength: 150 } },
      nodes: { shape: "dot", font: { size: 13 } },
      edges: { font: { size: 10, align: "middle" }, smooth: { type: "dynamic" } },
      interaction: { hover: true, tooltipDelay: 100 },
    }
  );

  function renderStats() {
    const stats = KG_DATA.stats || {};
    const entries = [
      ["Entities", KG_DATA.nodes.length],
      ["Relations", KG_DATA.edges.length],
      ["Production edges", stats.production_edges ?? "-"],
      ["Staging edges", stats.staging_edges ?? "-"],
    ];
    document.getElementById("stats").innerHTML = entries
      .map(([label, value]) => (
        '<div class="stat"><div class="value">' + value + '</div><div class="label">' + label + "</div></div>"
      ))
      .join("");
  }

  function renderLegend() {
    const typesPresent = new Set(KG_DATA.nodes.map((n) => n.type));
    document.getElementById("legend").innerHTML = Array.from(typesPresent)
      .map((t) => (
        '<div class="legend-item"><span class="legend-swatch" style="background:' + colorFor(t) + '"></span>' + t + "</div>"
      ))
      .join("") || '<p id="empty-state">No entities extracted yet.</p>';
  }

  function showNodeDetails(nodeId) {
    const n = nodesById[nodeId];
    if (!n) return;
    const rows = [
      ["Name", n.label],
      ["Type", n.type],
      ["In-degree", n.in_degree],
      ["Out-degree", n.out_degree],
    ];
    renderDetailsTable(rows);
  }

  function showEdgeDetails(edgeId) {
    const e = KG_DATA.edges[edgeId];
    if (!e) return;
    const rows = [
      ["Relation", e.relation],
      ["Source", (nodesById[e.source] || {}).label || e.source],
      ["Target", (nodesById[e.target] || {}).label || e.target],
      ["Confidence", e.confidence.toFixed(2)],
      ["Source document", e.source_doc],
    ];
    renderDetailsTable(rows);
  }

  function renderDetailsTable(rows) {
    document.getElementById("empty-state").style.display = "none";
    const table = document.getElementById("details-table");
    table.style.display = "";
    table.innerHTML = rows
      .map(([k, v]) => "<tr><td>" + k + "</td><td>" + v + "</td></tr>")
      .join("");
  }

  network.on("click", (params) => {
    if (params.nodes.length > 0) {
      showNodeDetails(params.nodes[0]);
    } else if (params.edges.length > 0) {
      showEdgeDetails(params.edges[0]);
    }
  });

  document.getElementById("search").addEventListener("input", (evt) => {
    const q = evt.target.value.trim().toLowerCase();
    const matches = KG_DATA.nodes
      .filter((n) => !q || n.label.toLowerCase().includes(q) || n.type.toLowerCase().includes(q))
      .map((n) => n.id);
    visNodes.update(
      KG_DATA.nodes.map((n) => ({
        id: n.id,
        color: matches.includes(n.id) ? colorFor(n.type) : "rgba(200,200,200,0.35)",
      }))
    );
  });

  renderStats();
  renderLegend();

  if (KG_DATA.nodes.length === 0) {
    document.getElementById("graph").innerHTML =
      '<div style="padding:2rem;color:#5b6b79;">No knowledge graph data was available when this page was generated. ' +
      'Run <code>python demo_dynamic_kg.py --backend mock</code> to (re)extract a graph from your documents and regenerate this page.</div>';
  }
</script>

</body>
</html>
"""
