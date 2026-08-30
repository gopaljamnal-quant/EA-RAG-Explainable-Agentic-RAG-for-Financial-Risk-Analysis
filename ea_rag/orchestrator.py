"""
EA-RAG Verification-Gated Orchestration (Algorithm 1 in the paper).

    Require: query q, corpus D, graph G, threshold tau, max retries R
    Ensure:  claims y_hat, provenance P, scores c

    T <- Planner(q)
    for each sub-task t in T:
        repeat
            D_t <- DenseRetrieve(t, D)
            G_t <- GraphRetrieve(t, G)
            Q_t <- QuantAgent(t)                      # if applicable
            y_t <- GraphReasoner(t, D_t, G_t, Q_t)
            c_t <- Critic(y_t, D_t, G_t, Q_t)
        until c_t >= tau or retries == R
        if c_t >= tau:
            p_t <- Explainer(y_t, D_t, G_t, Q_t)
            append y_t, p_t, c_t
        else:
            flag y_t as unsupported; append with c_t retained
    return y_hat, P, c

This module is a line-for-line implementation of that loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .agents import CriticVerifierAgent, ExplainerAgent, GraphReasonerAgent, PlannerAgent
from .data_models import Document, VerifiedClaim
from .kg import FinancialKnowledgeGraph
from .llm import BaseLLM
from .quant import QuantitativeAgent
from .retrieval import DenseRetriever, GraphRetriever


@dataclass
class OrchestratorConfig:
    tau: float = 0.7          # faithfulness/confidence threshold
    max_retries: int = 2      # R in Algorithm 1
    top_k_passages: int = 3
    max_hops: int = 3


class EARAGOrchestrator:
    def __init__(
        self,
        kg: FinancialKnowledgeGraph,
        documents: List[Document],
        llm: BaseLLM,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.kg = kg
        self.config = config or OrchestratorConfig()

        self.dense_retriever = DenseRetriever(documents)
        self.graph_retriever = GraphRetriever(kg, max_hops=self.config.max_hops)
        self.quant_agent = QuantitativeAgent()

        self.planner = PlannerAgent(kg, llm=llm)
        self.reasoner = GraphReasonerAgent(llm=llm)
        self.critic = CriticVerifierAgent()
        self.explainer = ExplainerAgent(kg)

    def run(self, query: str, quant_spec: Optional[dict] = None) -> List[VerifiedClaim]:
        subtasks = self.planner.decompose(query, quant_spec=quant_spec)
        results: List[VerifiedClaim] = []

        for subtask in subtasks:
            retries = 0
            claim = None
            confidence = 0.0
            passages, graph_evidence, quant_result = [], None, None

            while True:
                passages = (
                    self.dense_retriever.retrieve(subtask.description, top_k=self.config.top_k_passages)
                    if subtask.needs_dense
                    else []
                )
                graph_evidence = (
                    self.graph_retriever.retrieve(subtask)
                    if subtask.needs_graph
                    else self.graph_retriever.retrieve(subtask)  # empty seeds -> empty evidence
                )
                quant_result = (
                    self.quant_agent.run(subtask.quant_spec)
                    if subtask.needs_quant and subtask.quant_spec
                    else None
                )

                claim = self.reasoner.synthesize(subtask, passages, graph_evidence, quant_result)
                confidence = self.critic.verify(claim, passages, graph_evidence)

                retries += 1
                if confidence >= self.config.tau or retries > self.config.max_retries:
                    break
                # A real re-retrieval step would widen top_k / hop budget or
                # reformulate the sub-query here; for this reference
                # implementation, widening top_k is enough to demonstrate
                # the retry mechanism.
                self.config.top_k_passages += 2

            if confidence >= self.config.tau:
                provenance = self.explainer.explain(claim, passages, graph_evidence, quant_result)
                results.append(VerifiedClaim(claim=claim, confidence=confidence, supported=True,
                                              provenance=provenance, retries_used=retries - 1))
            else:
                results.append(VerifiedClaim(claim=claim, confidence=confidence, supported=False,
                                              provenance=None, retries_used=retries - 1))

        return results

    def explain_with_counterfactual(
        self, verified_claim: VerifiedClaim, counterfactual_narrative: str
    ) -> None:
        """Attach a counterfactual sensitivity statement (Section IV.D,
        "Counterfactual (sensitivity)") to an already-accepted claim's
        provenance, e.g. after the caller has recomputed a quant tool with a
        perturbed graph edge or input -- see demo.py for a worked example."""
        if verified_claim.provenance is not None:
            verified_claim.provenance.counterfactual = counterfactual_narrative
