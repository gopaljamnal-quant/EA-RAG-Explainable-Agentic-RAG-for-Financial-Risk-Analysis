"""
Agent roles from Section IV.C of the paper.

(The Dense retriever and Graph retriever agents are implemented directly in
retrieval.py since they are thin, tool-like wrappers; the Quantitative agent
lives in quant.py for the same reason. This module holds the three agents
that actually reason over evidence: Planner, Graph-Reasoner, and
Critic/Verifier, plus the Explainer.)
"""

from __future__ import annotations

import re
from typing import List, Optional

from .data_models import (
    Claim,
    Document,
    GraphEvidence,
    Provenance,
    QuantResult,
    RetrievedPassage,
    SubTask,
)
from .kg import FinancialKnowledgeGraph
from .llm import BaseLLM

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "if",
    "what", "our", "we", "for", "on", "by", "with", "as", "at", "this",
    "that", "under", "would", "will", "be",
}


def _keywords(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

class PlannerAgent:
    """Decomposes a risk query into an ordered list of SubTasks.

    Real deployments would prompt an LLM to output a structured plan (see the
    commented `_llm_plan` path below); the default heuristic path splits a
    compound query on conjunctions and resolves entity mentions against the
    knowledge graph so the demo is fully runnable without an API key while
    still producing genuine multi-sub-task plans for compound queries.
    """

    def __init__(self, kg: FinancialKnowledgeGraph, llm: Optional[BaseLLM] = None) -> None:
        self.kg = kg
        self.llm = llm

    def decompose(self, query: str, quant_spec: Optional[dict] = None) -> List[SubTask]:
        segments = re.split(r"\bif\b|;|\. ", query, flags=re.IGNORECASE)
        segments = [s.strip(" ?.") for s in segments if s.strip(" ?.")]
        if not segments:
            segments = [query]

        subtasks: List[SubTask] = []
        for i, segment in enumerate(segments):
            seeds = self._match_entities(segment)
            needs_quant = bool(quant_spec) and i == len(segments) - 1  # attach quant to final sub-task
            subtasks.append(
                SubTask(
                    id=f"t{i+1}",
                    description=segment,
                    needs_dense=True,
                    needs_graph=bool(seeds),
                    needs_quant=needs_quant,
                    seed_entities=seeds,
                    quant_spec=quant_spec if needs_quant else None,
                )
            )
        return subtasks

    def _match_entities(self, text: str) -> List[str]:
        text_lower = text.lower()
        matches = []
        for entity in self.kg._entities.values():  # noqa: SLF001 - internal read-only use
            if entity.name.lower() in text_lower:
                matches.append(entity.id)
        return matches


# --------------------------------------------------------------------------
# Graph-Reasoning agent
# --------------------------------------------------------------------------

class GraphReasonerAgent:
    """Synthesizes dense passages + graph evidence + quantitative output into
    a candidate claim with explicit citations, per Section IV.C."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def synthesize(
        self,
        subtask: SubTask,
        passages: List[RetrievedPassage],
        graph_evidence: GraphEvidence,
        quant_result: Optional[QuantResult],
    ) -> Claim:
        clauses = []
        cited_passages = [p.doc_id for p in passages]
        cited_relations = [r.key() for r in graph_evidence.relations]

        if graph_evidence.relations:
            clauses.append(f"Graph evidence indicates: {graph_evidence.path_description()}.")
        if passages:
            top = passages[0]
            clauses.append(
                f"This is corroborated by a {top.source_type or 'retrieved'} passage "
                f"[{top.doc_id}]: \"{top.text[:160].strip()}...\""
            )
        if quant_result:
            clauses.append(quant_result.narrative)

        if not clauses:
            claim_text = f"No sufficient evidence was retrieved to answer: '{subtask.description}'."
        else:
            claim_text = " ".join(clauses)

        prompt = (
            "You are the Graph-Reasoning agent in a financial risk analysis system. "
            "Rewrite the following evidence synthesis as a single fluent analytical "
            "statement. Preserve every citation marker in square brackets exactly "
            "as given, and do not introduce any fact not present below.\n\n"
            f"Sub-task: {subtask.description}\n"
            f"Evidence synthesis:\n{claim_text}"
        )
        final_text = self.llm.generate(prompt, system="Preserve citations; do not hallucinate.") or claim_text

        return Claim(
            id=f"claim_{subtask.id}",
            subtask_id=subtask.id,
            text=final_text,
            cited_passage_ids=cited_passages,
            cited_relation_keys=cited_relations,
            used_quant=quant_result,
        )


# --------------------------------------------------------------------------
# Critic / Verifier agent
# --------------------------------------------------------------------------

class CriticVerifierAgent:
    """Computes a faithfulness/entailment score c_i in [0, 1] for a claim
    against its cited evidence (Section IV.C, "Critic/verifier agent").

    IMPLEMENTATION NOTE: this reference implementation uses a transparent,
    dependency-free heuristic (structural coverage + lexical overlap)
    instead of a trained NLI / faithfulness model, so the whole pipeline
    runs offline. For a production system, replace `_textual_entailment`
    with a call to a cross-encoder NLI model or an automated faithfulness
    metric such as RAGAS (Es et al., 2024) -- the rest of the orchestration
    loop (Algorithm 1) is agnostic to how c_i is computed.
    """

    def __init__(
        self,
        structural_weight: float = 0.5,
        textual_weight: float = 0.5,
    ) -> None:
        self.structural_weight = structural_weight
        self.textual_weight = textual_weight

    def verify(
        self,
        claim: Claim,
        passages: List[RetrievedPassage],
        graph_evidence: GraphEvidence,
    ) -> float:
        structural_score = self._structural_coverage(claim, graph_evidence)
        textual_score = self._textual_entailment(claim, passages)

        if not claim.cited_relation_keys and not claim.cited_passage_ids:
            return 0.0  # a claim with no citations at all cannot be verified

        if not claim.cited_relation_keys:
            return textual_score
        if not claim.cited_passage_ids:
            return structural_score

        return self.structural_weight * structural_score + self.textual_weight * textual_score

    @staticmethod
    def _structural_coverage(claim: Claim, graph_evidence: GraphEvidence) -> float:
        """Fraction of the claim's cited relation keys that are actually
        present in the retrieved graph evidence (a hallucinated graph fact
        would score 0 here)."""
        if not claim.cited_relation_keys:
            return 1.0
        available = {r.key() for r in graph_evidence.relations}
        matched = sum(1 for k in claim.cited_relation_keys if k in available)
        return matched / len(claim.cited_relation_keys)

    @staticmethod
    def _textual_entailment(claim: Claim, passages: List[RetrievedPassage]) -> float:
        """Lexical-overlap proxy for entailment: fraction of the claim's
        salient keywords that also appear in the union of cited passages.
        This is a coarse stand-in for a real NLI entailment score -- see the
        class docstring."""
        if not passages:
            return 0.0
        claim_keywords = _keywords(claim.text)
        if not claim_keywords:
            return 1.0
        passage_keywords = set()
        for p in passages:
            passage_keywords |= _keywords(p.text)
        overlap = claim_keywords & passage_keywords
        return len(overlap) / len(claim_keywords)


# --------------------------------------------------------------------------
# Explainer agent
# --------------------------------------------------------------------------

class ExplainerAgent:
    """Builds the three-tier explanation (structural / attributional /
    counterfactual) described in Section IV.D for an accepted claim."""

    def __init__(self, kg: FinancialKnowledgeGraph) -> None:
        self.kg = kg

    def explain(
        self,
        claim: Claim,
        passages: List[RetrievedPassage],
        graph_evidence: GraphEvidence,
        quant_result: Optional[QuantResult] = None,
        counterfactual_narrative: Optional[str] = None,
    ) -> Provenance:
        # (1) Structural: minimal provenance subgraph is exactly the graph
        # evidence actually cited by the claim (not the full bounded
        # traversal), so a reviewer sees only what supports this claim.
        cited_keys = set(claim.cited_relation_keys)
        minimal_relations = [r for r in graph_evidence.relations if r.key() in cited_keys] or graph_evidence.relations
        involved_entity_ids = {r.source for r in minimal_relations} | {r.target for r in minimal_relations}
        minimal_entities = [e for e in graph_evidence.entities if e.id in involved_entity_ids] or graph_evidence.entities

        # (2) Attributional: span-level citations are simply the retrieved
        # passages actually cited in the claim.
        cited_passage_ids = set(claim.cited_passage_ids)
        cited_passages = [p for p in passages if p.doc_id in cited_passage_ids] or passages

        return Provenance(
            claim_id=claim.id,
            entities=minimal_entities,
            relations=minimal_relations,
            passages=cited_passages,
            quant=quant_result,
            # (3) Counterfactual: supplied by the orchestrator when a
            # counterfactual probe was requested for this sub-task (see
            # orchestrator.py / demo.py for a worked example).
            counterfactual=counterfactual_narrative,
        )
