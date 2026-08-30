"""
End-to-end demo of the EA-RAG pipeline, reproducing the illustrative case
study from Section VII of the paper:

    "If Supplier Y suffers a plant closure, what is our counterparty credit
    exposure through Firm X's guarantee of Subsidiary Z's debt?"

Run offline (no API key, no GPU) with the default mock backend:

    python demo.py

Run with a real open-weight model via Ollama (recommended -- see README.md
"Running with a real open-source LLM" for setup):

    ollama pull gemma3:4b
    python demo.py --backend ollama --model gemma3:4b

Run with a real open-weight model loaded directly via `transformers`
(requires: pip install torch transformers accelerate; add `--load-in-4bit`
on a smaller GPU):

    python demo.py --backend hf --model Qwen/Qwen3-14B

Run with Claude instead (requires: pip install anthropic, and
ANTHROPIC_API_KEY set):

    python demo.py --backend anthropic --model claude-sonnet-5
"""

from __future__ import annotations

import argparse

from ea_rag import (
    Document,
    Entity,
    EntityType,
    FinancialKnowledgeGraph,
    MockLLM,
    Relation,
    RelationType,
)
from ea_rag.llm import BaseLLM
from ea_rag.orchestrator import EARAGOrchestrator, OrchestratorConfig
from ea_rag.quant import MertonInputs, merton_distance_to_default


def build_llm(backend: str, model: str | None, load_in_4bit: bool) -> BaseLLM:
    if backend == "mock":
        return MockLLM()
    if backend == "ollama":
        from ea_rag.llm import OllamaLLM

        return OllamaLLM(model=model or "gemma3:4b")
    if backend == "hf":
        from ea_rag.llm import HuggingFaceLLM

        return HuggingFaceLLM(model_name=model or "Qwen/Qwen3-14B", load_in_4bit=load_in_4bit)
    if backend == "anthropic":
        from ea_rag.llm import AnthropicLLM

        return AnthropicLLM(model=model or "claude-sonnet-5")
    raise ValueError(f"Unknown backend '{backend}'")


def build_case_study_kg() -> FinancialKnowledgeGraph:
    kg = FinancialKnowledgeGraph(min_confidence_high_impact=0.75)

    supplier_y = Entity(id="supplier_y", name="Supplier Y", type=EntityType.COMPANY)
    firm_x = Entity(id="firm_x", name="Firm X", type=EntityType.COMPANY)
    subsidiary_z = Entity(id="subsidiary_z", name="Subsidiary Z", type=EntityType.SUBSIDIARY)

    for e in (supplier_y, firm_x, subsidiary_z):
        kg.add_entity(e)

    # Stage 1 (structured seeding): ownership/guarantee facts from Exhibit 21
    # and prior 8-K filings are treated as high-confidence structured facts.
    kg.add_relation(
        Relation(
            source="firm_x",
            relation=RelationType.GUARANTEES,
            target="subsidiary_z",
            confidence=0.97,
            source_doc_id="8K-2025-0143",
            extraction_method="structured_seed",
        )
    )

    # Stage 2 (LLM-assisted extraction): the supply relationship is extracted
    # from a risk-factor disclosure, with a slightly lower confidence.
    kg.add_relation(
        Relation(
            source="supplier_y",
            relation=RelationType.SUPPLIES,
            target="firm_x",
            confidence=0.88,
            source_doc_id="10K-2025-ITEM1A",
            extraction_method="llm_extraction",
        )
    )
    return kg


def build_documents() -> list[Document]:
    return [
        Document(
            id="NEWS-0231",
            source_type="news",
            issuer="Supplier Y",
            date="2026-06-02",
            text=open("data/appleQ2.pdf", "rb").read().decode("latin-1", errors="ignore")
        ),
        Document(
            id="10K-2025-ITEM1A",
            source_type="10-K",
            issuer="Firm X",
            date="2025-02-14",
            text=open("data/msftQ2.pdf", "rb").read().decode("latin-1", errors="ignore")
        ),
        Document(
            id="8K-2025-0143",
            source_type="8-K",
            issuer="Firm X",
            date="2025-05-30",
            text=open("data/pltrQ2.pdf", "rb").read().decode("latin-1", errors="ignore")
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="EA-RAG case-study demo")
    parser.add_argument(
        "--backend", choices=["mock", "ollama", "hf", "anthropic"], default="hf",
        help="LLM backend to use (default: hf - HuggingFace, no dependencies)",
    )
    parser.add_argument("--model", default=None, help="Model name/ID override for the chosen backend")
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantize (--backend hf only)")
    args = parser.parse_args()

    kg = build_case_study_kg()
    documents = build_documents()
    llm = build_llm(args.backend, args.model, args.load_in_4bit)

    print(f"=== Backend: {type(llm).__name__} (model={args.model or 'default'}) ===")
    print("=== Financial Knowledge Graph stats ===")
    print(kg.stats())
    print()

    orchestrator = EARAGOrchestrator(
        kg=kg,
        documents=documents,
        llm=llm,
        config=OrchestratorConfig(tau=0.5, max_retries=2, top_k_passages=2, max_hops=3),
    )

    query = ( "How is Palantir’s accelerating commercial revenue growth impacting or being impacted by the enterprise AI "
              "infrastructure spending and cloud consumption trends reported by Microsoft and Apple?")

    # Baseline and stressed Merton distance-to-default for Firm X, attached
    # as the quantitative sub-task. Asset value is stressed downward to
    # reflect the modeled impact of losing Supplier Y as a single-source
    # input; this is a simplified illustrative shock, not a calibrated model.
    baseline = merton_distance_to_default(
        MertonInputs(asset_value=500.0, debt_face_value=350.0, asset_volatility=0.25, risk_free_rate=0.04)
    )
    stressed_quant_spec = {
        "tool": "merton_distance_to_default",
        "params": dict(asset_value=460.0, debt_face_value=350.0, asset_volatility=0.30, risk_free_rate=0.04),
    }

    results = orchestrator.run(query, quant_spec=stressed_quant_spec)

    print("=== EA-RAG results ===")
    for vc in results:
        print(f"\n--- Sub-task: {vc.claim.subtask_id} ---")
        print(f"Claim       : {vc.claim.text}")
        print(f"Confidence  : {vc.confidence:.2f}  (supported={vc.supported}, retries={vc.retries_used})")
        if vc.provenance:
            prov = vc.provenance
            print("Provenance  :")
            for k, v in prov.to_dict().items():
                print(f"    {k}: {v}")

    # --- Counterfactual sensitivity explanation (Section IV.D, tier 3) ---
    # "What would have to be different for this assessment to change?"
    # Here: recompute the stressed scenario as if Firm X's guarantee of
    # Subsidiary Z's debt were released, i.e. Firm X's guaranteed debt face
    # value drops by the guaranteed term-loan amount (illustrative: 100).
    stressed = merton_distance_to_default(
        MertonInputs(asset_value=460.0, debt_face_value=350.0, asset_volatility=0.30, risk_free_rate=0.04)
    )
    released = merton_distance_to_default(
        MertonInputs(asset_value=460.0, debt_face_value=250.0, asset_volatility=0.30, risk_free_rate=0.04)
    )
    pd_stressed = stressed.outputs["probability_of_default"]
    pd_released = released.outputs["probability_of_default"]
    pct_change = (pd_released - pd_stressed) / pd_stressed * 100

    counterfactual_text = (
        f"Baseline P[default] = {baseline.outputs['probability_of_default']:.2%}; "
        f"stressed (post-closure) P[default] = {pd_stressed:.2%}. "
        f"Counterfactual: if Firm X's guarantee of Subsidiary Z's debt were released "
        f"(reducing guaranteed debt face value from 350 to 250), the stressed "
        f"P[default] would move to {pd_released:.2%}, a change of {pct_change:+.1f}% "
        f"relative to the stressed-with-guarantee case."
    )

    supported_claims = [vc for vc in results if vc.supported]
    if supported_claims:
        orchestrator.explain_with_counterfactual(supported_claims[-1], counterfactual_text)
        print("\n=== Counterfactual sensitivity (attached to final claim) ===")
        print(counterfactual_text)


if __name__ == "__main__":
    main()
