"""
Minimal unit tests covering the load-bearing pieces of the pipeline:
knowledge-graph confidence gating, bounded traversal, the Merton model, and
the end-to-end orchestrator on both a supported and an unsupported query.

Run with:  python -m pytest tests/ -v
"""

import math

import pytest

from ea_rag import (
    Document,
    Entity,
    EntityType,
    FinancialKnowledgeGraph,
    MockLLM,
    Relation,
    RelationType,
)
from ea_rag.llm import OllamaLLM, HuggingFaceLLM, _strip_thinking
from ea_rag.orchestrator import EARAGOrchestrator, OrchestratorConfig
from ea_rag.quant import MertonInputs, merton_distance_to_default, parametric_var


@pytest.fixture
def toy_kg():
    kg = FinancialKnowledgeGraph(min_confidence_high_impact=0.75)
    a = Entity(id="a", name="Firm A", type=EntityType.COMPANY)
    b = Entity(id="b", name="Firm B", type=EntityType.COMPANY)
    kg.add_entity(a)
    kg.add_entity(b)
    return kg, a, b


def test_high_impact_relation_below_threshold_is_staged_only(toy_kg):
    kg, a, b = toy_kg
    rel = Relation(source="a", relation=RelationType.GUARANTEES, target="b", confidence=0.5)
    admitted = kg.add_relation(rel)
    assert admitted is False
    stats = kg.stats()
    assert stats["production_edges"] == 0
    assert stats["staging_edges"] == 1


def test_high_impact_relation_above_threshold_is_admitted(toy_kg):
    kg, a, b = toy_kg
    rel = Relation(source="a", relation=RelationType.GUARANTEES, target="b", confidence=0.9)
    admitted = kg.add_relation(rel)
    assert admitted is True
    assert kg.stats()["production_edges"] == 1


def test_human_review_overrides_low_confidence(toy_kg):
    kg, a, b = toy_kg
    rel = Relation(source="a", relation=RelationType.GUARANTEES, target="b", confidence=0.2, reviewed_by="analyst_1")
    assert kg.add_relation(rel) is True


def test_low_impact_relation_always_admitted(toy_kg):
    kg, a, b = toy_kg
    rel = Relation(source="a", relation=RelationType.SUPPLIES, target="b", confidence=0.1)
    assert kg.add_relation(rel) is True


def test_bounded_traversal_finds_multi_hop_path():
    kg = FinancialKnowledgeGraph()
    x = Entity(id="x", name="X", type=EntityType.COMPANY)
    y = Entity(id="y", name="Y", type=EntityType.COMPANY)
    z = Entity(id="z", name="Z", type=EntityType.SUBSIDIARY)
    for e in (x, y, z):
        kg.add_entity(e)
    kg.add_relation(Relation(source="y", relation=RelationType.SUPPLIES, target="x", confidence=0.9))
    kg.add_relation(Relation(source="x", relation=RelationType.GUARANTEES, target="z", confidence=0.95))

    evidence = kg.bounded_traversal(["y"], max_hops=2)
    entity_ids = {e.id for e in evidence.entities}
    assert entity_ids == {"x", "y", "z"}
    assert len(evidence.relations) == 2


def test_merton_distance_to_default_matches_closed_form():
    result = merton_distance_to_default(
        MertonInputs(asset_value=500.0, debt_face_value=350.0, asset_volatility=0.25, risk_free_rate=0.04)
    )
    d2 = (math.log(500.0 / 350.0) + (0.04 - 0.5 * 0.25 ** 2) * 1.0) / (0.25 * math.sqrt(1.0))
    assert result.outputs["distance_to_default"] == pytest.approx(d2, rel=1e-9)
    assert 0.0 < result.outputs["probability_of_default"] < 1.0


def test_higher_leverage_increases_probability_of_default():
    low_leverage = merton_distance_to_default(
        MertonInputs(asset_value=500.0, debt_face_value=200.0, asset_volatility=0.25, risk_free_rate=0.04)
    )
    high_leverage = merton_distance_to_default(
        MertonInputs(asset_value=500.0, debt_face_value=450.0, asset_volatility=0.25, risk_free_rate=0.04)
    )
    assert high_leverage.outputs["probability_of_default"] > low_leverage.outputs["probability_of_default"]


def test_parametric_var_is_positive_for_typical_inputs():
    result = parametric_var(portfolio_value=1_000_000, expected_return=0.0, volatility=0.2, confidence=0.99)
    assert result.outputs["var_amount"] > 0


def test_orchestrator_supported_claim_end_to_end():
    kg = FinancialKnowledgeGraph()
    y = Entity(id="supplier_y", name="Supplier Y", type=EntityType.COMPANY)
    x = Entity(id="firm_x", name="Firm X", type=EntityType.COMPANY)
    for e in (y, x):
        kg.add_entity(e)
    kg.add_relation(Relation(source="supplier_y", relation=RelationType.SUPPLIES, target="firm_x", confidence=0.9))

    docs = [
        Document(
            id="doc1",
            source_type="news",
            text="Supplier Y reported a plant closure impacting output to Firm X.",
        )
    ]
    orchestrator = EARAGOrchestrator(
        kg=kg, documents=docs, llm=MockLLM(), config=OrchestratorConfig(tau=0.3, max_retries=1)
    )
    results = orchestrator.run("What is Firm X's exposure to Supplier Y?")
    assert len(results) == 1
    assert results[0].supported is True
    assert results[0].provenance is not None
    assert "SUPPLIES" in results[0].claim.text


def test_orchestrator_flags_unsupported_claim_without_fabricating():
    kg = FinancialKnowledgeGraph()
    docs = [Document(id="d1", source_type="news", text="Irrelevant text about the weather.")]
    orchestrator = EARAGOrchestrator(
        kg=kg, documents=docs, llm=MockLLM(), config=OrchestratorConfig(tau=0.9, max_retries=1)
    )
    results = orchestrator.run("What is Firm Q's exposure to Firm R?")
    assert results[0].supported is False
    assert results[0].provenance is None


# ---------------------------------------------------------------------
# Open-weight LLM backends (HuggingFaceLLM, OllamaLLM)
# ---------------------------------------------------------------------
# These tests do NOT download any real model weights or require a real
# Ollama install. HuggingFaceLLM is tested only for its error-handling
# path (this sandbox has no `torch`, which is realistic for a fresh
# environment); OllamaLLM is tested against a real local HTTP server that
# mimics Ollama's /api/chat response shape, so the request construction
# and response parsing are genuinely exercised end to end.

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def test_strip_thinking_removes_think_tags():
    assert _strip_thinking("<think>reasoning...</think>Final answer.") == "Final answer."
    assert _strip_thinking("No think tags here.") == "No think tags here."


def test_huggingface_llm_missing_dependency_gives_actionable_error():
    try:
        import torch  # noqa: F401

        pytest.skip("torch is installed in this environment; the missing-dependency path doesn't apply")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="pip install torch transformers accelerate"):
        HuggingFaceLLM()


class _MockOllamaHandler(BaseHTTPRequestHandler):
    """Stands in for a local Ollama server's /api/chat endpoint."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        assert body["model"] == "qwen3:14b"
        assert body["stream"] is False
        assert body["messages"][-1]["role"] == "user"
        assert "exposure" in body["messages"][-1]["content"]
        # enable_thinking defaults to False, so the "think" field must be
        # omitted entirely rather than sent as False -- this matters for
        # models like Gemma 3 that don't support the field at all.
        assert "think" not in body

        reply = {
            "message": {
                "role": "assistant",
                "content": "<think>tracing the graph path...</think>Firm X is exposed to Supplier Y.",
            }
        }
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):  # silence default request logging
        pass


@pytest.fixture
def mock_ollama_port():
    server = HTTPServer(("localhost", 0), _MockOllamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()
    thread.join()


def test_ollama_llm_default_model_is_gemma3_4b():
    """gemma3:4b is EA-RAG's default because it runs CPU-only on a laptop;
    explicitly locking this in so it isn't silently changed later."""
    assert OllamaLLM().model == "gemma3:4b"


def test_ollama_llm_end_to_end_against_local_mock_server(mock_ollama_port):
    llm = OllamaLLM(model="qwen3:14b", host=f"http://localhost:{mock_ollama_port}")
    result = llm.generate("What is Firm X's exposure?")
    # the <think>...</think> block must be stripped, leaving only the answer
    assert result == "Firm X is exposed to Supplier Y."


def test_ollama_llm_sends_think_field_only_when_enabled():
    """Gemma 3 has no thinking mode, so `think` must never be sent for it
    (verified above via `assert "think" not in body`); this test confirms
    the field *is* sent when a caller explicitly opts in for a model that
    does support it (e.g. Qwen3)."""

    class _ThinkingHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            assert body.get("think") is True
            reply = {"message": {"role": "assistant", "content": "ok"}}
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("localhost", 0), _ThinkingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        llm = OllamaLLM(model="qwen3:14b", host=f"http://localhost:{port}", enable_thinking=True)
        result = llm.generate("test prompt")
        assert result == "ok"
    finally:
        server.shutdown()
        thread.join()


def test_ollama_llm_can_drive_the_full_orchestrator(mock_ollama_port):
    """End-to-end sanity check: the orchestrator works identically whether
    driven by MockLLM or by a real (here, mocked-transport) OllamaLLM,
    since every agent only depends on BaseLLM.generate()."""
    kg = FinancialKnowledgeGraph()
    y = Entity(id="supplier_y", name="Supplier Y", type=EntityType.COMPANY)
    x = Entity(id="firm_x", name="Firm X", type=EntityType.COMPANY)
    for e in (y, x):
        kg.add_entity(e)
    kg.add_relation(Relation(source="supplier_y", relation=RelationType.SUPPLIES, target="firm_x", confidence=0.9))

    docs = [Document(id="doc1", source_type="news", text="Supplier Y plant closure affects Firm X exposure.")]
    llm = OllamaLLM(model="qwen3:14b", host=f"http://localhost:{mock_ollama_port}")
    orchestrator = EARAGOrchestrator(kg=kg, documents=docs, llm=llm, config=OrchestratorConfig(tau=0.1, max_retries=1))
    results = orchestrator.run("What is Firm X's exposure to Supplier Y?")
    assert results[0].claim.text == "Firm X is exposed to Supplier Y."
