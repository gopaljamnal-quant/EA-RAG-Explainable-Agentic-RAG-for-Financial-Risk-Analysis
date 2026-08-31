from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

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
from pdf_loader import load_pdfs


def build_llm(backend: str, model: str | None, load_in_4bit: bool) -> BaseLLM:
    """Build LLM backend."""
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


def load_kg_from_config(kg_config_path: str) -> FinancialKnowledgeGraph:
    """
    Load FinancialKnowledgeGraph from a JSON config file.

    Config format:
    {
        "min_confidence_high_impact": 0.75,
        "entities": [
            {"id": "firm_x", "name": "Firm X", "type": "COMPANY"},
            ...
        ],
        "relations": [
            {
                "source": "firm_x",
                "relation": "GUARANTEES",
                "target": "subsidiary_z",
                "confidence": 0.97,
                "source_doc_id": "8K-2025-0143",
                "extraction_method": "structured_seed"
            },
            ...
        ]
    }
    """
    kg = FinancialKnowledgeGraph(min_confidence_high_impact=0.75)

    if not Path(kg_config_path).exists():
        print(f"⚠️  KG config not found: {kg_config_path}. Using empty graph.")
        return kg

    with open(kg_config_path) as f:
        config = json.load(f)

    # Add entities
    for ent_config in config.get("entities", []):
        entity = Entity(
            id=ent_config["id"],
            name=ent_config["name"],
            type=EntityType[ent_config.get("type", "COMPANY")],
        )
        kg.add_entity(entity)
        print(f"  ✓ Entity: {entity.name} ({entity.type.name})")

    # Add relations
    for rel_config in config.get("relations", []):
        relation = Relation(
            source=rel_config["source"],
            relation=RelationType[rel_config["relation"]],
            target=rel_config["target"],
            confidence=rel_config.get("confidence", 0.8),
            source_doc_id=rel_config.get("source_doc_id", "unknown"),
            extraction_method=rel_config.get("extraction_method", "manual"),
            reviewed_by=rel_config.get("reviewed_by"),
        )
        admitted = kg.add_relation(relation)
        status = "→ production" if admitted else "→ staging"
        print(f"  ✓ Relation: {relation.source} --{relation.relation.name}--> {relation.target} ({status})")

    return kg


def load_documents_from_directory(doc_dir: str, max_docs: Optional[int] = None) -> list[Document]:
    """
    Load documents from a directory of PDFs or text files.
    Uses pdf_loader.py if PDFs exist; falls back to text files.
    """
    doc_dir_path = Path(doc_dir)

    if not doc_dir_path.exists():
        print(f"⚠️  Document directory not found: {doc_dir}")
        return []

    # Try to load PDFs first
    pdf_files = list(doc_dir_path.glob("*.pdf"))
    if pdf_files:
        print(f"📄 Found {len(pdf_files)} PDFs. Loading with pdfplumber...")
        return load_pdfs(doc_dir, max_docs=max_docs)

    # Fallback: load plain text files
    text_files = list(doc_dir_path.glob("*.txt"))
    documents = []
    for i, txt_file in enumerate(text_files):
        if max_docs and i >= max_docs:
            break

        with open(txt_file) as f:
            text = f.read()

        doc = Document(
            id=txt_file.stem,
            text=text,
            source_type="text",
            issuer=txt_file.stem,
            date=None,
        )
        documents.append(doc)
        print(f"  ✓ Loaded: {txt_file.name} ({len(text)} chars)")

    if not documents:
        print(f"⚠️  No documents found in {doc_dir}")

    return documents


def load_query_from_config(query_config_path: str) -> str:
    """
    Load query from a JSON or plain text file.

    If JSON: expects {"query": "..."} or {"query": [...]}
    If text: just returns the raw text.
    """
    query_path = Path(query_config_path)

    if not query_path.exists():
        return "What are the financial risks?"  # default fallback

    try:
        with open(query_path) as f:
            content = json.load(f)
        if isinstance(content, dict):
            return content.get("query", "What are the financial risks?")
        if isinstance(content, list):
            return " ".join(content)
    except json.JSONDecodeError:
        # Not JSON, treat as plain text
        pass

    with open(query_path) as f:
        return f.read().strip()


def load_merton_params_from_config(merton_config_path: Optional[str]) -> tuple[MertonInputs, dict]:
    """
    Load Merton baseline and stressed parameters from JSON.

    Config format:
    {
        "baseline": {
            "asset_value": 500.0,
            "debt_face_value": 350.0,
            "asset_volatility": 0.25,
            "risk_free_rate": 0.04
        },
        "stressed": {
            "asset_value": 460.0,
            ...
        }
    }
    """
    baseline_params = MertonInputs(
        asset_value=500.0,
        debt_face_value=350.0,
        asset_volatility=0.25,
        risk_free_rate=0.04,
    )
    stressed_spec = {
        "tool": "merton_distance_to_default",
        "params": dict(
            asset_value=460.0,
            debt_face_value=350.0,
            asset_volatility=0.30,
            risk_free_rate=0.04,
        ),
    }

    if not merton_config_path or not Path(merton_config_path).exists():
        print("⚠️  Merton config not found. Using defaults.")
        return baseline_params, stressed_spec

    with open(merton_config_path) as f:
        config = json.load(f)

    baseline_params = MertonInputs(**config.get("baseline", baseline_params.__dict__))
    stressed_spec["params"] = config.get("stressed", stressed_spec["params"])

    return baseline_params, stressed_spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EA-RAG dynamic demo: loads KG, documents, and query from config files"
    )

    # File paths
    parser.add_argument(
        "--kg-config",
        default="config/knowledge_graph.json",
        help="Path to KG config JSON (entities + relations)",
    )
    parser.add_argument(
        "--doc-dir",
        default="financial_documents/",
        help="Directory containing PDFs or text documents",
    )
    parser.add_argument(
        "--query",
        default="config/query.json",
        help="Path to query file (JSON or plain text)",
    )
    parser.add_argument(
        "--merton-config",
        default="config/merton_params.json",
        help="Path to Merton model parameters JSON",
    )

    # LLM options
    parser.add_argument(
        "--backend",
        choices=["mock", "ollama", "hf", "anthropic"],
        default="mock",
        help="LLM backend to use (default: mock)",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantize (HF only)")

    # Orchestration options
    parser.add_argument("--tau", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retry attempts")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K passages to retrieve")
    parser.add_argument("--max-hops", type=int, default=3, help="Max graph hops")

    args = parser.parse_args()

    # ===== Load dynamic components =====
    print("\n=== Loading Configuration ===\n")

    print("📊 Knowledge Graph:")
    kg = load_kg_from_config(args.kg_config)

    print(f"\n📚 Documents from '{args.doc_dir}':")
    documents = load_documents_from_directory(args.doc_dir)

    print(f"\n❓ Query from '{args.query}':")
    query = load_query_from_config(args.query)
    print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")

    print(f"\n💰 Merton parameters:")
    baseline, stressed_spec = load_merton_params_from_config(args.merton_config)
    print(f"  Baseline: asset={baseline.asset_value}, debt={baseline.debt_face_value}")
    print(
        f"  Stressed: asset={stressed_spec['params']['asset_value']}, debt={stressed_spec['params']['debt_face_value']}")

    # ===== Build LLM and Orchestrator =====
    print(f"\n🤖 LLM Backend: {args.backend}")
    llm = build_llm(args.backend, args.model, args.load_in_4bit)

    config = OrchestratorConfig(
        tau=args.tau,
        max_retries=args.max_retries,
        top_k_passages=args.top_k,
        max_hops=args.max_hops,
    )

    orchestrator = EARAGOrchestrator(kg=kg, documents=documents, llm=llm, config=config)

    print(f"\n=== KG Stats ===")
    print(kg.stats())

    # ===== Run Pipeline =====
    print(f"\n=== Running EA-RAG Pipeline ===\n")
    results = orchestrator.run(query, quant_spec=stressed_spec)

    print("\n=== Results ===")
    for vc in results:
        print(f"\n--- Sub-task: {vc.claim.subtask_id} ---")
        print(f"Claim       : {vc.claim.text}")
        print(f"Confidence  : {vc.confidence:.2f}  (supported={vc.supported}, retries={vc.retries_used})")
        if vc.provenance:
            prov = vc.provenance
            print("Provenance  :")
            for k, v in prov.to_dict().items():
                if v:
                    print(f"    {k}: {v}")

    # ===== Counterfactual (optional) =====
    print("\n=== Counterfactual Sensitivity ===")
    baseline_result = merton_distance_to_default(baseline)
    stressed_result = merton_distance_to_default(
        MertonInputs(**stressed_spec["params"])
    )
    # (re-run with perturbed guarantee, if applicable)
    counterfactual_result = merton_distance_to_default(
        MertonInputs(
            asset_value=stressed_spec["params"]["asset_value"],
            debt_face_value=250.0,  # released guarantee: 350 -> 250
            asset_volatility=stressed_spec["params"]["asset_volatility"],
            risk_free_rate=stressed_spec["params"]["risk_free_rate"],
        )
    )

    pd_baseline = baseline_result.outputs["probability_of_default"]
    pd_stressed = stressed_result.outputs["probability_of_default"]
    pd_released = counterfactual_result.outputs["probability_of_default"]
    pct_change = (pd_released - pd_stressed) / pd_stressed * 100 if pd_stressed > 0 else 0

    counterfactual_text = (
        f"Baseline P[default] = {pd_baseline:.2%}; "
        f"stressed (post-closure) P[default] = {pd_stressed:.2%}. "
        f"Counterfactual: if guarantee were released (debt 350→250), "
        f"P[default] would be {pd_released:.2%}, a change of {pct_change:+.1f}%."
    )

    supported_claims = [vc for vc in results if vc.supported]
    if supported_claims:
        orchestrator.explain_with_counterfactual(supported_claims[-1], counterfactual_text)
        print(f"\n{counterfactual_text}")


if __name__ == "__main__":
    main()