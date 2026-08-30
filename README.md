# EA-RAG: Explainable Agentic RAG for Financial Risk Analysis

Reference implementation accompanying the paper *"Explainable Agentic RAG for
Financial Risk Analysis: A Knowledge-Graph-Grounded Multi-Agent Framework for
Trustworthy LLM-Based Risk Intelligence."*

This is a runnable, dependency-light implementation of the architecture in
the paper (Section IV), not a production system. It is meant to make the
paper's design concrete — every class below maps directly to a component or
algorithm described in the text — and to give you a base to extend with
real data, a real LLM, and a real NLI/faithfulness model.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo.py                  # offline, no API key, no GPU (MockLLM)
python -m pytest tests/ -v      # requires: pip install pytest
```

`demo.py --backend mock` (the default) runs entirely offline using `MockLLM`,
and reproduces the paper's Section VII case study: a supplier plant-closure
query traced through a two-hop `SUPPLIES` → `GUARANTEES` graph path, backed
by a Merton distance-to-default calculation, with a genuine (recomputed, not
invented) counterfactual: "if the guarantee were released, how would the
stressed probability of default change?"

**Important: what `MockLLM` is, precisely.** It is not a language model. It
calls no model, local or remote. `agents.py` only ever asks the LLM to
rephrase evidence it has already assembled in Python (retrieved passages,
graph facts, quant tool output) into fluent prose while preserving citation
markers — `MockLLM.generate()` just returns that pre-assembled text
unchanged. This means the demo's *orchestration logic* (retrieval, graph
traversal, verification, retries, provenance) is 100% real and tested;
only the final sentence-level phrasing is templated rather than generated.
For a real experiment you want an actual model doing that phrasing (and,
if you enable LLM-based planning, actual planning) — see below.

## Running with a real open-source LLM

Two backends are included, covering the two common ways people actually run
open-weight models. Either one is a drop-in replacement for `MockLLM` —
every agent talks only to `BaseLLM.generate()`, so nothing else in the
pipeline changes.

### Option A — Ollama (recommended: easiest setup, no CUDA config)

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull an open-weight model, e.g. Gemma 3 4B (~3.3GB, runs fine CPU-only
#    on a laptop -- this is EA-RAG's default):
ollama pull gemma3:4b
# 3. Run the demo against it:
pip install requests
python demo.py --backend ollama --model gemma3:4b
```

`OllamaLLM` (in `ea_rag/llm.py`) talks to Ollama's local `/api/chat`
endpoint over HTTP. Note on the `system` message: Gemma's own chat
template has no native system role, but Ollama's packaged template for
`gemma3` merges a system message into the leading user turn automatically,
so `OllamaLLM` sends it as a normal `{"role": "system", ...}` message and
it works correctly without any special-casing in this codebase. For models
that *do* support a thinking mode (Qwen3, DeepSeek-R1 distills, etc.),
`OllamaLLM` also strips `<think>...</think>` blocks automatically and only
requests thinking mode when you pass `enable_thinking=True` — irrelevant
for Gemma 3, which has no thinking mode, so leave it at the default False.

If you're on a memory-constrained laptop and see out-of-memory errors even
with the 4B model, Ollama's KV-cache for Gemma 3's sliding-window attention
can be larger than expected on some versions; setting the environment
variable `OLLAMA_KV_CACHE_TYPE=q8_0` (or `q4_0`) before `ollama serve`
reduces that footprint.

### Option B — Hugging Face `transformers` (more control, needs a GPU for anything beyond small models)

```bash
pip install torch transformers accelerate
python demo.py --backend hf --model Qwen/Qwen3-14B --load-in-4bit
```

`HuggingFaceLLM` loads the model and tokenizer directly and applies the
model's chat template. `--load-in-4bit` needs `bitsandbytes` and a CUDA GPU.

### Which model to use

"Best open-source LLM" changes almost monthly and open-model leaderboards
are a heavily marketed space — be skeptical of any single "#1" claim,
including the one implied by this package's default. `gemma3:4b` (used as
the Ollama default above) is a reasonable, well-supported choice for
laptop-only use with no GPU — not a claim that it's unconditionally the
strongest option. See the "Why Gemma 3 4B is the default" note at the top
of `ea_rag/llm.py` for current alternatives (similarly-sized options like
`qwen3:4b`/`llama3.2:3b`; larger MoE models such as recent Qwen, DeepSeek,
Llama, GLM, or Kimi releases if you have more hardware or a hosted
endpoint) and check current benchmarks yourself before committing to one
for a real study.

### Mixing models per agent

Every agent takes its own `llm` instance if you construct them directly
instead of via `EARAGOrchestrator` — e.g. give the critic/verifier a
reasoning-tuned model for stronger entailment checking while the planner
and explainer use a faster, cheaper model:

```python
from ea_rag.llm import OllamaLLM
from ea_rag.agents import CriticVerifierAgent, GraphReasonerAgent

fast_llm = OllamaLLM(model="gemma3:4b")
reasoning_llm = OllamaLLM(model="qwen3:14b", enable_thinking=True)  # bigger model, only if you have the hardware

reasoner = GraphReasonerAgent(llm=fast_llm)
# CriticVerifierAgent in this reference implementation uses a lexical/
# structural heuristic rather than an LLM call at all (see below) --
# swap in an LLM- or NLI-based verifier here if you upgrade it.
```

## Package layout → paper section

| File | Paper section | What it implements |
|---|---|---|
| `ea_rag/data_models.py` | III (Problem Formulation) | `Entity`, `Relation`, `Document`, `Claim`, `Provenance` = the formal $\mathcal{G}$, $\mathcal{D}$, $y_i$, $p_i$ objects |
| `ea_rag/kg.py` | IV.B (FKG construction) | Confidence-gated production/staging graph, bounded traversal, minimal-path provenance extraction |
| `ea_rag/retrieval.py` | IV.C | `DenseRetriever` (TF-IDF stand-in for embeddings), `GraphRetriever` |
| `ea_rag/quant.py` | IV.C (Quantitative agent) | Merton (1974) distance-to-default, parametric VaR — deterministic tools, not LLM-generated numbers |
| `ea_rag/agents.py` | IV.C, IV.D | `PlannerAgent`, `GraphReasonerAgent`, `CriticVerifierAgent`, `ExplainerAgent` |
| `ea_rag/orchestrator.py` | Algorithm 1 | The verification-gated retry loop, line for line |
| `ea_rag/llm.py` | — | Pluggable LLM backend: offline `MockLLM` (default), open-weight `OllamaLLM` / `HuggingFaceLLM`, or closed-source `AnthropicLLM` |
| `demo.py` | VII (Case study) | Full worked example; `--backend {mock,ollama,hf,anthropic}` |
| `tests/test_ea_rag.py` | — | Unit tests for the KG gating logic, Merton model, the end-to-end orchestrator (supported + unsupported paths), and the `OllamaLLM` request/response handling (against a local mock HTTP server) |

## What's genuinely implemented vs. simplified

**Genuinely implemented and tested:**
- Confidence-gated knowledge graph construction (high-impact relation types
  require either a confidence threshold or human sign-off before entering
  the production graph used for final claims).
- Bounded-hop graph traversal and minimal-path provenance extraction.
- The full verification-gated orchestration loop from Algorithm 1, including
  the retry mechanism and the "flag as unsupported rather than fabricate"
  behavior when no evidence clears the threshold (see
  `test_orchestrator_flags_unsupported_claim_without_fabricating`).
- Closed-form Merton distance-to-default and parametric VaR.
- A genuine counterfactual: `demo.py` actually re-runs the Merton model with
  a perturbed input (guarantee released) and reports the real percentage
  change — it does not print a canned number.

**Simplified, and clearly marked in code for where to upgrade:**
- `CriticVerifierAgent` uses a transparent lexical-overlap + structural-
  coverage heuristic instead of a trained NLI/entailment model. Swap
  `_textual_entailment` for a cross-encoder NLI model or the RAGAS
  faithfulness metric for production use.
- `MockLLM` templates the final claim/explanation phrasing rather than
  generating it, so the whole pipeline runs with no API key and no GPU.
  Pass an `OllamaLLM`, `HuggingFaceLLM`, or `AnthropicLLM` instance (see
  `ea_rag/llm.py` and "Running with a real open-source LLM" above) to use
  a real model instead — no other code changes needed, since every agent
  only talks to the `BaseLLM.generate()` interface.
- `PlannerAgent.decompose` uses a simple conjunction-splitting + entity-name-
  matching heuristic rather than an LLM-generated plan; see the class
  docstring for where to add an LLM-based planning prompt.
- `DenseRetriever` uses TF-IDF rather than a neural embedding model, purely
  to avoid a heavyweight dependency in a reference implementation.

## Extending this for real use

1. Replace `DenseRetriever`'s TF-IDF vectorizer with a real embedding model
   and vector index (e.g. FAISS, pgvector) — it only needs to keep returning
   `RetrievedPassage` objects.
2. Replace `CriticVerifierAgent._textual_entailment` with a real faithfulness
   / NLI scorer.
3. Feed `FinancialKnowledgeGraph` from your actual filings pipeline (Exhibit
   21 ownership data, 8-K guarantee disclosures, NLP-extracted supply-chain
   relations) instead of the three hand-coded entities in `demo.py`.
4. Pass `OllamaLLM(model="gemma3:4b")`, `HuggingFaceLLM(model_name="Qwen/Qwen3-14B")`,
   `AnthropicLLM(model="claude-sonnet-5")`, or another backend you implement
   against `BaseLLM`, to `EARAGOrchestrator` for real generation.
5. Tune `OrchestratorConfig.tau` and `max_retries` per risk tier, as
   discussed in the paper's Results and Discussion section.

## License / status

Research/teaching reference code, provided as-is to accompany the paper
draft. Not validated for use in a live risk-management workflow.
