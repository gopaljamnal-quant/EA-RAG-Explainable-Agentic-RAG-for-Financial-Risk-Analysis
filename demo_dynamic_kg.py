"""
Full end-to-end demo with IMPROVED KG visualization.

No static JSON required. Everything is extracted from real documents.
Graph is actually readable!

Run:
    mkdir -p financial_documents
    # Add your PDFs to financial_documents/
    python demo_dynamic_kg_improved.py --backend mock
    # Then open kg_graph_improved.html in browser for hierarchical visualization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ea_rag.llm import BaseLLM, MockLLM
from ea_rag.orchestrator import EARAGOrchestrator, OrchestratorConfig
from ea_rag.quant import MertonInputs, merton_distance_to_default
from ea_rag.dynamic_kg_extractor import DynamicKGBuilder

from ea_rag.improved_kg_visualizer import ImprovedKGVisualizer


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EA-RAG Dynamic Demo with IMPROVED Visualization"
    )

    # Document options
    parser.add_argument(
        "--doc-dir",
        default="financial_documents/",
        help="Directory containing PDF or text documents",
    )

    # KG extraction options
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence to admit relations to production graph",
    )

    # Visualization options
    parser.add_argument(
        "--graph-output",
        default="kg_graph_improved.html",
        help="Output path for improved hierarchical graph",
    )
    parser.add_argument(
        "--metrics-output",
        default="kg_metrics.html",
        help="Output path for metrics dashboard",
    )

    # Query options
    parser.add_argument(
        "--query",
        default="What are the financial risks from supply chain disruptions and guarantees?",
        help="Query to run against the extracted KG",
    )

    # LLM options
    parser.add_argument(
        "--backend",
        choices=["mock", "ollama", "hf", "anthropic"],
        default="mock",
        help="LLM backend",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantize (HF only)")

    # Orchestration options
    parser.add_argument("--tau", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retry attempts")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K passages")
    parser.add_argument("--max-hops", type=int, default=3, help="Max graph hops")

    args = parser.parse_args()

    # ===== Step 1: Dynamic KG Extraction =====
    print("\n" + "#" * 80)
    print("# STEP 1: DYNAMIC KNOWLEDGE GRAPH EXTRACTION")
    print("#" * 80)

    builder = DynamicKGBuilder(llm=build_llm(args.backend, args.model, args.load_in_4bit))
    kg, extracted_entities, extracted_relations = builder.build_from_directory(
        args.doc_dir, min_relation_confidence=args.min_confidence
    )

    # ===== Step 2: IMPROVED Interactive Visualization =====
    print("\n" + "#" * 80)
    print("# STEP 2: IMPROVED HIERARCHICAL VISUALIZATION")
    print("#" * 80)

    visualizer = ImprovedKGVisualizer(kg)

    # Print detailed console analysis
    visualizer.print_detailed_summary()

    # Generate improved visualizations
    print(f"\n🎨 Generating improved visualizations...")
    # After
    hierarchical_path = visualizer.to_plotly_hierarchical_html(
        output_path="kg_graph_improved.html",
        min_confidence=0.9,
        show_only_types=["COMPANY", "SUBSIDIARY", "RISK_FACTOR"]  # Add this
    )
    metrics_path = visualizer.to_metrics_dashboard(output_path=args.metrics_output)

    print(f"\n✅ Visualizations ready:")
    print(f"   🔗 Hierarchical graph: {hierarchical_path}")
    print(f"   📊 Metrics dashboard:  {metrics_path}")
    print(f"\n💡 TIP: Open {hierarchical_path} in a web browser to explore!")
    print(f"   - Node size = connectivity (bigger = more important)")
    print(f"   - Line thickness = confidence (thick = high confidence)")
    print(f"   - Node color = entity type (blue=company, orange=subsidiary, red=risk)")
    print(f"   - Line color = relation type (blue=SUPPLIES, orange=GUARANTEES, etc.)")

    # Export as JSON
    print(f"\n💾 Exporting KG data...")
    kg_dict = {
        "nodes": [
            {"id": e.id, "label": e.name, "type": e.type.name}
            for e in kg._entities.values()
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "relation": data["relation"].relation.value,
                "confidence": data["relation"].confidence,
                "source_doc": data["relation"].source_doc_id,
            }
            for u, v, _, data in kg._production.edges(keys=True, data=True)
        ],
        "stats": kg.stats(),
    }
    with open("kg_data.json", "w") as f:
        json.dump(kg_dict, f, indent=2)
    print(f"   ✓ JSON data: kg_data.json")

    # ===== Step 3: RAG Pipeline (Optional) =====
    print("\n" + "#" * 80)
    print("# STEP 3: EA-RAG PIPELINE (Optional)")
    print("#" * 80)

    # Load documents for retrieval
    from pdf_loader import load_pdfs
    documents = load_pdfs(args.doc_dir, max_docs=10) if Path(args.doc_dir).exists() else []

    if not documents:
        print("⚠️  No documents loaded. Skipping RAG pipeline.")
        print("\nTo run RAG pipeline:")
        print(f"  1. Add PDFs to {args.doc_dir}/")
        print(f"  2. Re-run this script")
    else:
        # Build orchestrator
        print(f"\n🤖 LLM Backend: {args.backend}")
        orchestrator = EARAGOrchestrator(
            kg=kg,
            documents=documents,
            llm=build_llm(args.backend, args.model, args.load_in_4bit),
            config=OrchestratorConfig(
                tau=args.tau,
                max_retries=args.max_retries,
                top_k_passages=args.top_k,
                max_hops=args.max_hops,
            ),
        )

        # Run query
        print(f"\n❓ Query: {args.query}")
        print("\n" + "-" * 80)
        results = orchestrator.run(args.query)

        print("\n=== EA-RAG Results ===")
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

        # ===== Optional: Quantitative Analysis =====
        print("\n" + "#" * 80)
        print("# STEP 4: COUNTERFACTUAL SENSITIVITY (MERTON MODEL)")
        print("#" * 80)

        try:
            print("\n📊 Running Merton distance-to-default analysis...")

            baseline = merton_distance_to_default(
                MertonInputs(asset_value=500.0, debt_face_value=350.0, asset_volatility=0.25, risk_free_rate=0.04)
            )

            stressed = merton_distance_to_default(
                MertonInputs(asset_value=460.0, debt_face_value=350.0, asset_volatility=0.30, risk_free_rate=0.04)
            )

            released = merton_distance_to_default(
                MertonInputs(asset_value=460.0, debt_face_value=250.0, asset_volatility=0.30, risk_free_rate=0.04)
            )

            pd_baseline = baseline.outputs["probability_of_default"]
            pd_stressed = stressed.outputs["probability_of_default"]
            pd_released = released.outputs["probability_of_default"]
            pct_change = (pd_released - pd_stressed) / pd_stressed * 100 if pd_stressed > 0 else 0

            print(f"\nBaseline P[default]          = {pd_baseline:.2%}")
            print(f"Stressed P[default]          = {pd_stressed:.2%}")
            print(f"Counterfactual P[default]    = {pd_released:.2%}")
            print(f"Impact of releasing guarantee: {pct_change:+.1f}%")

            if results and results[-1].supported:
                counterfactual_text = (
                    f"Baseline P[default] = {pd_baseline:.2%}; "
                    f"stressed P[default] = {pd_stressed:.2%}. "
                    f"If guarantee released (debt 350→250), P[default] = {pd_released:.2%} "
                    f"(change: {pct_change:+.1f}%)."
                )
                orchestrator.explain_with_counterfactual(results[-1], counterfactual_text)
                print(f"\n✓ Counterfactual attached to final claim.")
        except Exception as e:
            print(f"⚠️  Quantitative analysis skipped: {e}")

    print("\n" + "#" * 80)
    print("# ✅ ANALYSIS COMPLETE")
    print("#" * 80)
    print(f"\nNext steps:")
    print(f"  1. 🌐 Open {args.graph_output} in your browser")
    print(f"  2. 📈 View metrics at {args.metrics_output}")
    print(f"  3. 📋 Inspect kg_data.json for raw node/edge data")
    print(f"  4. ➕ Add more PDFs to {args.doc_dir}/ and re-run for larger KGs\n")


if __name__ == "__main__":
    main()
