"""
EA-RAG: Explainable Agentic Retrieval-Augmented Generation for Financial Risk Analysis
========================================================================================

Reference implementation of the architecture described in:

    "Explainable Agentic RAG for Financial Risk Analysis: A Knowledge-Graph-
    Grounded Multi-Agent Framework for Trustworthy LLM-Based Risk Intelligence"

This package implements, end to end and runnably:

  - a confidence-gated Financial Knowledge Graph (kg.py)
  - a dense (TF-IDF) retriever and a bounded-hop graph retriever (retrieval.py)
  - deterministic quantitative risk tools: Merton distance-to-default and
    parametric Value-at-Risk (quant.py)
  - the six agent roles from the paper: Planner, (Dense/Graph) Retriever,
    Graph-Reasoning, Critic/Verifier, and Explainer (agents.py)
  - the verification-gated orchestration loop of Algorithm 1 (orchestrator.py)

The LLM used by the agents is pluggable (llm.py). By default the package ships
with a dependency-free ``MockLLM`` so the whole pipeline runs offline without
any API key -- this is what ``demo.py`` uses unless you pass ``--backend``.
Two open-weight backends are included for real experiments: ``OllamaLLM``
(talks to a local Ollama server -- easiest setup, no CUDA/quantization
config) and ``HuggingFaceLLM`` (loads a model directly via ``transformers``,
with optional 4-bit quantization). A thin ``AnthropicLLM`` wrapper is also
included for using a closed-source model instead. See README.md, section
"Running with a real open-source LLM".

NOTE ON SCOPE: this is a reference / teaching implementation meant to make the
paper's architecture concrete and runnable, not a production system. The
critic/verifier's faithfulness scoring, in particular, uses a transparent
lexical + structural heuristic rather than a trained NLI model -- see
``agents.CriticVerifierAgent`` for exactly where to plug in a real entailment
model (e.g. a cross-encoder NLI model or the RAGAS faithfulness metric).
"""

from .data_models import (
    Entity,
    EntityType,
    Relation,
    RelationType,
    Document,
    SubTask,
    RetrievedPassage,
    GraphEvidence,
    QuantResult,
    Claim,
    Provenance,
    VerifiedClaim,
)
from .kg import FinancialKnowledgeGraph
from .retrieval import DenseRetriever, GraphRetriever
from .quant import QuantitativeAgent
from .llm import BaseLLM, MockLLM, OllamaLLM, HuggingFaceLLM, AnthropicLLM
from .agents import PlannerAgent, GraphReasonerAgent, CriticVerifierAgent, ExplainerAgent
from .orchestrator import EARAGOrchestrator

__all__ = [
    "Entity",
    "EntityType",
    "Relation",
    "RelationType",
    "Document",
    "SubTask",
    "RetrievedPassage",
    "GraphEvidence",
    "QuantResult",
    "Claim",
    "Provenance",
    "VerifiedClaim",
    "FinancialKnowledgeGraph",
    "DenseRetriever",
    "GraphRetriever",
    "QuantitativeAgent",
    "BaseLLM",
    "MockLLM",
    "OllamaLLM",
    "HuggingFaceLLM",
    "AnthropicLLM",
    "PlannerAgent",
    "GraphReasonerAgent",
    "CriticVerifierAgent",
    "ExplainerAgent",
    "EARAGOrchestrator",
]

__version__ = "0.1.0"
