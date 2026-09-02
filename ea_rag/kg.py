"""
Financial Knowledge Graph (FKG) with confidence-gated staging, implementing
Section IV.B of the paper:

    1. Structured seeding      -> add_entity / add_relation(..., extraction_method="structured_seed")
    2. LLM-assisted extraction -> add_relation(..., extraction_method="llm_extraction")
    3. Confidence-gated validation ->
         high-impact relation types (GUARANTEES, EXPOSED_TO, OWNS) are only
         admitted to the *production* graph above `min_confidence_high_impact`
         (or once `reviewed_by` is set); everything else lands in the
         *staging* graph and is available for candidate generation only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import networkx as nx

from .data_models import Entity, GraphEvidence, Relation, HIGH_IMPACT_RELATION_TYPES


class FinancialKnowledgeGraph:
    def __init__(self, min_confidence_high_impact: float = 0.75) -> None:
        self.min_confidence_high_impact = min_confidence_high_impact
        # production graph: only validated, audit-ready edges
        self._production = nx.MultiDiGraph()
        # staging graph: everything (superset of production), used for
        # candidate generation but never cited as final evidence
        self._staging = nx.MultiDiGraph()
        self._entities: Dict[str, Entity] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity
        for graph in (self._production, self._staging):
            if not graph.has_node(entity.id):
                graph.add_node(entity.id, entity=entity)

    def add_relation(self, relation: Relation) -> bool:
        """Add a relation to the staging graph, and to the production graph
        if it clears the confidence-gated validation step.

        Returns True iff the relation was admitted to the production graph.
        """
        for entity_id in (relation.source, relation.target):
            if entity_id not in self._entities:
                raise KeyError(f"Unknown entity id '{entity_id}': add_entity() first")

        self._add_relation_to_graph(self._staging, relation)

        admitted = self._passes_validation_gate(relation)
        if admitted:
            self._add_relation_to_graph(self._production, relation)
        return admitted

    @staticmethod
    def _add_relation_to_graph(graph: nx.MultiDiGraph, relation: Relation) -> None:
        graph.add_edge(relation.source, relation.target, key=relation.key(), relation=relation)

    def _passes_validation_gate(self, relation: Relation) -> bool:
        if relation.relation not in HIGH_IMPACT_RELATION_TYPES:
            # Lower-impact / exploratory edges are admitted directly.
            return True
        if relation.reviewed_by is not None:
            # Human-in-the-loop sign-off always clears the gate.
            return True
        return relation.confidence >= self.min_confidence_high_impact

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        name_lower = name.lower()
        for e in self._entities.values():
            if e.name.lower() == name_lower:
                return e
        return None

    def bounded_traversal(
        self,
        seed_entity_ids: List[str],
        max_hops: int = 3,
        use_production_only: bool = True,
    ) -> GraphEvidence:
        """Breadth-first, bounded-depth traversal from a set of seed
        entities, as described for the graph retriever in Section IV.C.

        Returns the union of all nodes/edges visited within `max_hops`,
        which the caller (GraphRetriever) may further prune to the minimal
        path relevant to a specific claim (see FinancialKnowledgeGraph.
        shortest_path_evidence for that narrower use case).
        """
        graph = self._select_graph(use_production_only)
        visited_nodes: Set[str] = set()
        visited_edges: Dict[str, Relation] = {}

        frontier = [eid for eid in seed_entity_ids if graph.has_node(eid)]
        visited_nodes.update(frontier)

        for _ in range(max_hops):
            next_frontier = self._expand_frontier(
                graph, frontier, visited_nodes, visited_edges
            )
            frontier = next_frontier
            if not frontier:
                break

        entities = [self._entities[eid] for eid in visited_nodes if eid in self._entities]
        return GraphEvidence(
            entities=entities,
            relations=list(visited_edges.values()),
            seed_entities=seed_entity_ids,
        )

    def shortest_path_evidence(
        self, source_id: str, target_id: str, use_production_only: bool = True
    ) -> Optional[GraphEvidence]:
        """Minimal supporting subgraph between two entities -- used by the
        explainer agent to build the *minimal* provenance subgraph for a
        claim (Section IV.D, "Structural (provenance subgraph)").
        """
        graph = self._select_graph(use_production_only)
        undirected = graph.to_undirected(as_view=True)
        try:
            node_path = nx.shortest_path(undirected, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        relations: List[Relation] = []
        for a, b in zip(node_path[:-1], node_path[1:]):
            relation = self._best_relation_between(graph, a, b)
            if relation is not None:
                relations.append(relation)

        entities = [self._entities[n] for n in node_path if n in self._entities]
        return GraphEvidence(entities=entities, relations=relations, seed_entities=[source_id, target_id])

    def remove_relation(self, relation_key: str) -> bool:
        """Remove an edge from the production graph (used to compute
        counterfactual sensitivity explanations, e.g. 'if this GUARANTEES
        edge did not exist, would the conclusion change?')."""
        removed = False
        for graph in (self._production, self._staging):
            for u, v, k in list(graph.edges(keys=True)):
                if k == relation_key:
                    graph.remove_edge(u, v, key=k)
                    removed = True
        return removed

    def stats(self) -> Dict[str, int]:
        return {
            "entities": len(self._entities),
            "production_edges": self._production.number_of_edges(),
            "staging_edges": self._staging.number_of_edges(),
        }

    def _select_graph(self, use_production_only: bool) -> nx.MultiDiGraph:
        return self._production if use_production_only else self._staging

    def _expand_frontier(
        self,
        graph: nx.MultiDiGraph,
        frontier: List[str],
        visited_nodes: Set[str],
        visited_edges: Dict[str, Relation],
    ) -> List[str]:
        next_frontier: List[str] = []
        for node in frontier:
            self._collect_neighbors(
                graph.out_edges(node, data=True), 1, visited_nodes, visited_edges, next_frontier
            )
            # also traverse guarantee/exposure chains backwards, since
            # e.g. "who guarantees me" matters as much as "who I guarantee"
            self._collect_neighbors(
                graph.in_edges(node, data=True), 0, visited_nodes, visited_edges, next_frontier
            )
        return next_frontier

    @staticmethod
    def _collect_neighbors(
        edges,
        candidate_index: int,
        visited_nodes: Set[str],
        visited_edges: Dict[str, Relation],
        next_frontier: List[str],
    ) -> None:
        for edge in edges:
            candidate = edge[candidate_index]
            relation: Relation = edge[2]["relation"]
            visited_edges[relation.key()] = relation
            if candidate not in visited_nodes:
                visited_nodes.add(candidate)
                next_frontier.append(candidate)

    def _best_relation_between(
        self, graph: nx.MultiDiGraph, source_id: str, target_id: str
    ) -> Optional[Relation]:
        edge_data = None
        if graph.has_edge(source_id, target_id):
            edge_data = graph.get_edge_data(source_id, target_id)
        elif graph.has_edge(target_id, source_id):
            edge_data = graph.get_edge_data(target_id, source_id)
        if not edge_data:
            return None

        # take the first (or highest-confidence) parallel edge
        best = max(edge_data.values(), key=lambda data: data["relation"].confidence)
        return best["relation"]
