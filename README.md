# EA-RAG: Explainable Agentic RAG for Financial Risk Analysis

Reference implementation accompanying the paper *"Explainable Agentic RAG for Financial Risk Analysis: A Knowledge-Graph-Grounded Multi-Agent Framework for Trustworthy LLM-Based Risk Intelligence."*

This is a runnable, dependency-light implementation of the architecture in the paper (Section IV), not a production system. It is meant to make the paper's design concrete — every class below maps directly to a component or algorithm described in the text — and to give you a base to extend with real data, a real LLM, and a real NLI/faithfulness model.

A static, single-page summary of the project (overview, run instructions, and a
sample of the risk-analysis output) is available in [`index.html`](index.html) —
open it directly in a browser, no server required.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo_dynamic_kg.py --backend mock              # offline, no API key, no GPU
python -m pytest tests/ -v                             # requires: pip install pytest
```

The mock run generates two local files that are not checked into the repo
(see `.gitignore`): `kg_graph_improved.html` (interactive graph) and
`kg_metrics.html` (metrics dashboard). Open either in a browser to explore.

## Simplified Structure

```
ea_rag/                  # Core library (see table below for what each file does)
demo_dynamic_kg.py        # Main entry point: extraction + orchestration + visualization
dynamic_kg_extractor.py   # Entity/relation extraction from document text
pdf_loader.py             # load_pdfs() utility
index.html                # Static single-page project summary
tests/test_ea_rag.py      # Unit tests
```

| File | Responsibility |
|------|---|
| `ea_rag/data_models.py` | `Entity`, `Relation`, `Document`, `Claim`, `Provenance` data types |
| `ea_rag/kg.py` | Confidence-gated production/staging knowledge graph |
| `ea_rag/retrieval.py` | `DenseRetriever` (TF-IDF), `GraphRetriever` |
| `ea_rag/quant.py` | Merton distance-to-default, parametric VaR |
| `ea_rag/agents.py` | `PlannerAgent`, `GraphReasonerAgent`, `CriticVerifierAgent`, `ExplainerAgent` |
| `ea_rag/orchestrator.py` | Verification-gated retry loop (Algorithm 1) |
| `ea_rag/llm.py` | Pluggable backends: `MockLLM`, `OllamaLLM`, `HuggingFaceLLM`, `AnthropicLLM` |
| `ea_rag/improved_kg_visualizer.py` | Hierarchical graph + metrics dashboard HTML generation |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo_dynamic_kg.py --backend mock              # offline, no API key, no GPU
python -m pytest tests/ -v                             # requires: pip install pytest
```

## Main Demo: `demo_dynamic_kg.py`

**Dynamic Knowledge Graph Extraction + Orchestration + Interactive Visualization**

This is the primary entry point. It demonstrates the full EA-RAG pipeline with automatic entity/relation extraction from financial documents:

```bash
# 1. Setup (one-time)
mkdir -p financial_documents
# Add your financial PDFs to financial_documents/ directory

# 2. Run with mock LLM (offline, instant)
python demo_dynamic_kg.py --backend mock

# 3. View outputs
# - kg_graph_improved.html          (interactive hierarchical knowledge graph)
# - kg_metrics.html                 (metrics dashboard: entity/relation distributions, confidence analysis)
# - Console output                  (verified claims, provenance, confidence scores)
```

### What it does:

1. **Document Loading** (`pdf_loader.py`): Loads PDFs from `./financial_documents/` and auto-detects document type (10-K, 8-K, earnings, etc.)

2. **Dynamic KG Extraction** (`dynamic_kg_extractor.py`):
   - Chunks documents semantically using TF-IDF
   - Extracts entities: companies, subsidiaries, risk factors
   - Extracts relations: SUPPLIES, GUARANTEES, OWNS, EXPOSED_TO using pattern matching
   - Confidence-gates high-impact relations (GUARANTEES, OWNS require >75% confidence for production graph)
   - Creates FinancialKnowledgeGraph with staging/production separation

3. **EA-RAG Orchestration** (Algorithm 1):
   - Plans multi-step queries decomposing complex risk questions
   - Retrieves relevant dense passages + graph evidence
   - Runs quantitative models (Merton distance-to-default, parametric VaR)
   - Verifies claims against evidence with confidence scoring
   - Retries on low confidence (verification-gated loop)
   - Generates explanations with full provenance

4. **Interactive Visualization** (`improved_kg_visualizer.py`):
   - Hierarchical layout (readable, not spring-force chaos)
   - Node size by connectivity, color by entity type
   - Edge width/dash style by relation confidence
   - Hover for details, filter by confidence/type
   - Metrics dashboard: entity distributions, relation statistics, confidence analysis

### Command-line Options:

```bash
# Mock backend (offline, fastest)
python demo_dynamic_kg.py --backend mock

# With Ollama (local LLM)
python demo_dynamic_kg.py --backend ollama --model gemma3:4b

# With Hugging Face (direct model loading)
python demo_dynamic_kg.py --backend hf --model Qwen/Qwen3-14B --load-in-4bit

# With Anthropic (Claude API)
python demo_dynamic_kg.py --backend anthropic --model claude-sonnet-5

# Customize output paths
python demo_dynamic_kg.py --backend mock \
  --graph-output my_kg.html \
  --metrics-output my_metrics.html

# Adjust KG confidence threshold
python demo_dynamic_kg.py --backend mock --min-confidence 0.8

# Orchestration tuning
python demo_dynamic_kg.py --backend mock --tau 0.7 --max-retries 3 --top-k 5
```

## Running with a Real LLM

Two backends are included for production use. Either one is a drop-in replacement for `MockLLM` — every agent talks only to `BaseLLM.generate()`.

### Option A — Ollama (recommended: easiest setup, no CUDA config)

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull an open-weight model (Gemma 3 4B is the default)
ollama pull gemma3:4b
# 3. Run
pip install requests
python demo_dynamic_kg.py --backend ollama --model gemma3:4b
```

**Notes:**
- `OllamaLLM` talks to Ollama's local `/api/chat` endpoint over HTTP
- For models with thinking mode (Qwen3, DeepSeek-R1), set `enable_thinking=True` in the code
- On memory-constrained systems, set `OLLAMA_KV_CACHE_TYPE=q8_0` before `ollama serve`

### Option B — Hugging Face `transformers` (more control, needs a GPU)

```bash
# Install dependencies
pip install torch transformers accelerate

# Run with a local model
python demo_dynamic_kg.py --backend hf --model Qwen/Qwen3-14B --load-in-4bit
```

**Notes:**
- `--load-in-4bit` requires `bitsandbytes` and a CUDA GPU
- Works on CPU for small models (<7B) but will be slow
- Applies the model's native chat template automatically

### Which model to use?

`gemma3:4b` is a reasonable laptop-only choice (no GPU). For alternatives and benchmarks, see the comments in `ea_rag/llm.py`. Update as your preferred open-source LLM evolves.

### Mixing models per agent

For production workflows, assign different models to different agents:

```python
from ea_rag.llm import OllamaLLM
from ea_rag.agents import CriticVerifierAgent, GraphReasonerAgent
from ea_rag.orchestrator import EARAGOrchestrator

fast_llm = OllamaLLM(model="gemma3:4b")
reasoning_llm = OllamaLLM(model="qwen3:14b", enable_thinking=True)

reasoner = GraphReasonerAgent(llm=reasoning_llm)  # slower but stronger reasoning
# critic uses heuristic (no LLM call) by default; upgrade in code if needed

orchestrator = EARAGOrchestrator(kg=kg, documents=docs, llm=fast_llm)
```

## Architecture Overview

### Package Layout → Paper Section

| File | Paper Section | What it Implements |
|------|------|---|
| `ea_rag/data_models.py` | III (Problem Formulation) | `Entity`, `Relation`, `Document`, `Claim`, `Provenance` = formal $\mathcal{G}$, $\mathcal{D}$, $y_i$, $p_i$ |
| `ea_rag/kg.py` | IV.B (FKG construction) | Confidence-gated production/staging graph, bounded traversal, minimal-path provenance |
| `ea_rag/retrieval.py` | IV.C | `DenseRetriever` (TF-IDF stand-in), `GraphRetriever` |
| `ea_rag/quant.py` | IV.C (Quantitative agent) | Merton (1974) distance-to-default, parametric VaR |
| `ea_rag/agents.py` | IV.C, IV.D | `PlannerAgent`, `GraphReasonerAgent`, `CriticVerifierAgent`, `ExplainerAgent` |
| `ea_rag/orchestrator.py` | Algorithm 1 | Verification-gated retry loop, line-for-line |
| `ea_rag/llm.py` | — | Pluggable backends: `MockLLM`, `OllamaLLM`, `HuggingFaceLLM`, `AnthropicLLM` |
| `dynamic_kg_extractor.py` | IV.B (Knowledge graph construction) | `SemanticDocumentIndex`, `EntityRelationExtractor`, `DynamicKGBuilder` |
| `improved_kg_visualizer.py` | — | Hierarchical layout visualization, metrics dashboard |
| `pdf_loader.py` | — | PDF document loading with auto-detection of document type |
| `demo_dynamic_kg.py` | — | End-to-end example: PDFs → extraction → orchestration → visualization |
| `tests/test_ea_rag.py` | — | Unit tests: KG gating, Merton model, orchestrator, LLM handling |

## What's Genuinely Implemented vs. Simplified

### Genuinely Implemented and Tested

- ✅ **Confidence-gated KG construction**: High-impact relations (GUARANTEES, OWNS) require >75% confidence or human sign-off for production graph
- ✅ **Bounded-hop graph traversal**: Minimal-path provenance extraction for explainability
- ✅ **Full verification-gated orchestration**: Algorithm 1 line-for-line, including retry mechanism and "flag unsupported rather than fabricate" behavior
- ✅ **Closed-form quantitative models**: Merton distance-to-default, parametric VaR (not LLM-generated numbers)
- ✅ **Dynamic entity/relation extraction**: From unstructured financial documents with confidence scoring
- ✅ **Interactive visualizations**: Hierarchical layout, confidence-weighted edges, metrics dashboard

### Simplified (Marked for Upgrade)

- ⚠️ **Entity/Relation Extraction**: Uses regex patterns + heuristics. Upgrade to:
  - Named Entity Recognition (spaCy, Hugging Face NER)
  - Specialized relation extraction models
  - Confidence calibration on labeled data

- ⚠️ **CriticVerifierAgent**: Uses lexical-overlap heuristic instead of trained NLI model. Swap `_textual_entailment()` for:
  - Cross-encoder NLI model (e.g., DPR, Sentence-BERT)
  - RAGAS faithfulness metric

- ⚠️ **PlannerAgent.decompose()**: Conjunction-splitting heuristic instead of LLM-based planning. See class docstring for where to add LLM prompting.

- ⚠️ **DenseRetriever**: Uses TF-IDF (lightweight). Upgrade to:
  - Sentence embeddings (sentence-transformers)
  - Dense vector index (FAISS, pgvector, Pinecone)

- ⚠️ **MockLLM**: Templates final phrasing instead of generating. For real experiments, pass `OllamaLLM`, `HuggingFaceLLM`, or `AnthropicLLM` to `EARAGOrchestrator` — no other code changes needed.

## Extending for Production Use

1. **Embeddings**: Replace `DenseRetriever`'s TF-IDF with a neural embedding model (sentence-transformers) and vector index (FAISS, pgvector). Interface stays the same (`retrieve()` returns `RetrievedPassage` objects).

2. **NLI/Faithfulness**: Replace `CriticVerifierAgent._textual_entailment()` with a real cross-encoder or RAGAS score.

3. **Entity/Relation Extraction**: Feed `FinancialKnowledgeGraph` from your actual pipeline:
   - NLP-extracted supply-chain relations
   - 8-K guarantee disclosures
   - Exhibit 21 ownership structures
   - Third-party data APIs

4. **LLM Backend**: Upgrade from `MockLLM` to real generation:
   ```python
   llm = OllamaLLM(model="gemma3:4b")           # Local
   llm = HuggingFaceLLM(model_name="Qwen/Qwen3-14B")  # Local with GPU
   llm = AnthropicLLM(model="claude-sonnet-5")  # API
   orchestrator = EARAGOrchestrator(..., llm=llm)
   ```

5. **Tuning**: Adjust `OrchestratorConfig.tau` (confidence threshold) and `max_retries` per risk tier, as discussed in the paper's Results & Discussion section.

## Project Structure

```
ea_rag/                           # Core library
├── __init__.py
├── data_models.py               # Entity, Relation, Document, Claim, Provenance
├── kg.py                        # FinancialKnowledgeGraph (production/staging)
├── retrieval.py                 # DenseRetriever, GraphRetriever
├── quant.py                     # Merton, VaR
├── agents.py                    # Planner, GraphReasoner, Critic, Explainer
├── orchestrator.py              # Algorithm 1 implementation
└── llm.py                       # BaseLLM, MockLLM, OllamaLLM, HuggingFaceLLM, AnthropicLLM

demo_dynamic_kg.py               # Main entry point: extraction + orchestration + visualization
dynamic_kg_extractor.py          # SemanticDocumentIndex, EntityRelationExtractor, DynamicKGBuilder
improved_kg_visualizer.py        # ImprovedKGVisualizer (hierarchical layout, metrics)
pdf_loader.py                    # load_pdfs() utility

tests/
└── test_ea_rag.py               # Unit tests

README.md                         # This file
requirements.txt                 # Dependencies
```

## Installation & Quick Run

```bash
# 1. Clone and setup
git clone https://github.com/gopaljamnal-quant/EA-RAG-Explainable-Agentic-RAG-for-Financial-Risk-Analysis.git
cd EA-RAG-Explainable-Agentic-RAG-for-Financial-Risk-Analysis

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate           # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run (no PDFs needed for first test)
python demo_dynamic_kg.py --backend mock

# 5. With your own PDFs
mkdir -p financial_documents
# Copy your financial PDFs here
python demo_dynamic_kg.py --backend mock

# 6. Open the visualizations
# - kg_graph_improved.html in your browser
# - kg_metrics.html for metrics dashboard
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover:
- KG confidence-gating logic
- Merton distance-to-default calculations
- End-to-end orchestrator (supported + unsupported paths)
- LLM backend request/response handling

## License & Status

Research/teaching reference code, provided as-is to accompany the paper draft. Not validated for use in a live risk-management workflow.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{jamnal2024earag,
  title={Explainable Agentic RAG for Financial Risk Analysis: A Knowledge-Graph-Grounded Multi-Agent Framework for Trustworthy LLM-Based Risk Intelligence},
  author={Jamnal, Gopal Singh},
  year={2024}
}
```

## Contributing

For bug reports, feature requests, or contributions, please open an issue or pull request.
