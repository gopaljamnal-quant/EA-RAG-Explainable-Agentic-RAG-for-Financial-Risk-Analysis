"""
Shared data structures for EA-RAG.

These correspond directly to the formalization in Section III (Problem
Formulation) and Section IV (Proposed Framework) of the paper:

    G = (V, E)                     -> Entity, Relation
    D = {d_1, ..., d_n}             -> Document
    f(q, D, G) = (y_hat, P, c)      -> Claim (y_i), Provenance (p_i), confidence (c_i)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Knowledge graph primitives
# --------------------------------------------------------------------------

class EntityType(str, Enum):
    COMPANY = "Company"
    SUBSIDIARY = "Subsidiary"
    INSTRUMENT = "Instrument"
    PERSON = "Person"
    RISK_FACTOR = "RiskFactor"
    REGULATORY_EVENT = "RegulatoryEvent"


class RelationType(str, Enum):
    SUPPLIES = "SUPPLIES"
    GUARANTEES = "GUARANTEES"
    OWNS = "OWNS"
    EXPOSED_TO = "EXPOSED_TO"
    LITIGATION_AGAINST = "LITIGATION_AGAINST"
    DOWNGRADED_BY = "DOWNGRADED_BY"
    CORRELATED_WITH = "CORRELATED_WITH"


# Relation types that materially affect a risk decision and are therefore
# subject to confidence-gated validation before entering the production
# graph (Section IV.B, "Confidence-gated validation").
HIGH_IMPACT_RELATION_TYPES = {
    RelationType.GUARANTEES,
    RelationType.EXPOSED_TO,
    RelationType.OWNS,
}


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    type: EntityType

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.type.value})"


@dataclass
class Relation:
    source: str            # Entity.id
    relation: RelationType
    target: str             # Entity.id
    confidence: float
    source_doc_id: Optional[str] = None
    extraction_method: str = "structured_seed"   # or "llm_extraction"
    reviewed_by: Optional[str] = None

    def key(self) -> str:
        return f"{self.source}-{self.relation.value}-{self.target}"


@dataclass
class Document:
    id: str
    text: str
    source_type: str        # "10-K", "8-K", "news", "transcript"
    issuer: Optional[str] = None
    date: Optional[str] = None


# --------------------------------------------------------------------------
# Orchestration primitives
# --------------------------------------------------------------------------

@dataclass
class SubTask:
    id: str
    description: str
    needs_dense: bool = True
    needs_graph: bool = True
    needs_quant: bool = False
    seed_entities: List[str] = field(default_factory=list)   # entity ids to seed graph traversal
    quant_spec: Optional[Dict[str, Any]] = None               # parameters for the quant tool, if any


@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    score: float
    source_type: str = ""


@dataclass
class GraphEvidence:
    """A bounded-hop subgraph returned by the graph retriever for one sub-task."""
    entities: List[Entity]
    relations: List[Relation]
    seed_entities: List[str]

    def path_description(self) -> str:
        if not self.relations:
            return "(no graph path found)"
        id_to_name = {e.id: e.name for e in self.entities}
        return "; ".join(
            f"{id_to_name.get(relation.source, relation.source)} "
            f"--[{relation.relation.value}]--> "
            f"{id_to_name.get(relation.target, relation.target)}"
            for relation in self.relations
        )


@dataclass
class QuantResult:
    tool: str
    inputs: Dict[str, float]
    outputs: Dict[str, float]
    narrative: str = ""


@dataclass
class Claim:
    id: str
    subtask_id: str
    text: str
    cited_passage_ids: List[str] = field(default_factory=list)
    cited_relation_keys: List[str] = field(default_factory=list)
    used_quant: Optional[QuantResult] = None


@dataclass
class Provenance:
    claim_id: str
    entities: List[Entity]
    relations: List[Relation]
    passages: List[RetrievedPassage]
    quant: Optional[QuantResult]
    counterfactual: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "entities": [str(e) for e in self.entities],
            "relations": [self._format_relation(r) for r in self.relations],
            "passages": [f"[{p.doc_id}] {p.text[:120]}..." for p in self.passages],
            "quant": self.quant.narrative if self.quant else None,
            "counterfactual": self.counterfactual,
        }

    @staticmethod
    def _format_relation(relation: Relation) -> str:
        return relation.key() + f" (conf={relation.confidence:.2f}, src={relation.source_doc_id})"


@dataclass
class VerifiedClaim:
    claim: Claim
    confidence: float
    supported: bool
    provenance: Optional[Provenance] = None
    retries_used: int = 0
