from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        """Return a text completion for `prompt`."""
        raise NotImplementedError


class MockLLM(BaseLLM):
    """A deterministic stand-in for offline demos and unit tests. Calls no
    model at all -- see the module docstring above.

    It does not "understand" the prompt; agents.py calls it only for the
    final natural-language phrasing step and passes it structured data
    (entities, relations, passages) as f-string context, so the templated
    output remains grounded in exactly the evidence the orchestrator
    retrieved -- the same evidence the critic/verifier independently checks.
    """

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        lines = [l.strip() for l in prompt.strip().splitlines() if l.strip()]
        return lines[-1] if lines else ""


def _strip_thinking(text: str) -> str:
    """Qwen3 / DeepSeek-R1-style models wrap chain-of-thought in
    <think>...</think> when thinking mode is on; agents.py only wants the
    final answer, so strip it if present."""
    if "<think>" in text and "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()


class HuggingFaceLLM(BaseLLM):
    """Real open-weight LLM backend using Hugging Face `transformers`.

    Requires (on a machine with internet access -- this class cannot be
    exercised inside a network-sandboxed environment that blocks
    huggingface.co):

        pip install torch transformers accelerate
        pip install bitsandbytes   # only if load_in_4bit=True

    Example
    -------
        from ea_rag.llm import HuggingFaceLLM
        llm = HuggingFaceLLM(model_name="Qwen/Qwen3-14B", load_in_4bit=True)
        orchestrator = EARAGOrchestrator(..., llm=llm)

    Notes
    -----
    - Qwen3's chat template supports a hybrid thinking / non-thinking mode
      via `enable_thinking`. This defaults to False here: the planner and
      graph-reasoner benefit more from low latency than from long chain-of-
      thought, since the actual multi-step reasoning is done by the
      orchestrator (Algorithm 1), not by any single LLM call. Set
      `enable_thinking=True` if you assign this backend to the
      CriticVerifierAgent specifically and want it to reason harder about
      entailment.
    - `device_map="auto"` requires `accelerate`; remove it and call
      `.to("cuda")` / `.to("cpu")` yourself if you prefer manual placement.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        enable_thinking: bool = False,
        dtype: str = "auto",
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised via error-path test
            raise ImportError(
                "HuggingFaceLLM requires: pip install torch transformers accelerate"
            ) from exc

        self._torch = torch
        self.enable_thinking = enable_thinking
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantization_config = None
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError("load_in_4bit=True requires: pip install bitsandbytes") from exc
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
        )

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _strip_thinking(decoded)


class OllamaLLM(BaseLLM):
    """Real open-weight LLM backend via a local Ollama server -- the
    lowest-friction way to run an open model for this pipeline: no CUDA,
    no quantization config, no manual device placement.

    Setup (on your own machine -- Ollama needs outbound internet the first
    time to pull weights, and a running local server; neither is available
    inside this sandboxed environment, which is why this class is
    unit-tested here only against a local mock HTTP server, not a real
    Ollama install):

        1. Install Ollama: https://ollama.com/download
        2. ollama pull gemma3:4b       # ~3.3GB, runs fine CPU-only on a laptop
        3. Ollama serves on http://localhost:11434 automatically after install

    Example
    -------
        from ea_rag.llm import OllamaLLM
        llm = OllamaLLM(model="gemma3:4b")
        orchestrator = EARAGOrchestrator(..., llm=llm)

    Note on `enable_thinking`: this only applies to models that support
    Ollama's "thinking" feature (Qwen3, DeepSeek-R1 distills, etc.). Gemma 3
    has no thinking mode, so leave this False (the default) -- the `think`
    field is only included in the request at all when True, so nothing
    Gemma-incompatible is ever sent for this model.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        host: str = "http://localhost:11434",
        enable_thinking: bool = False,
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.enable_thinking = enable_thinking
        self.timeout = timeout

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("OllamaLLM requires: pip install requests") from exc

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if self.enable_thinking:
            # Only sent for models that actually support it; omitted
            # entirely otherwise (e.g. for Gemma 3, which has no thinking
            # mode and no need to receive this field).
            payload["think"] = True

        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Could not reach an Ollama server at {self.host}. Is Ollama installed and "
                f"running (`ollama serve`), and has '{self.model}' been pulled "
                f"(`ollama pull {self.model}`)? See README.md 'Running with a real "
                f"open-source LLM' for setup."
            ) from exc
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        return _strip_thinking(content)


class AnthropicLLM(BaseLLM):
    """Real LLM backend using the Anthropic Messages API, for plugging in a
    closed-source model instead of an open-weight one.

    Example
    -------
        from ea_rag.llm import AnthropicLLM
        llm = AnthropicLLM(model="claude-sonnet-5")
        orchestrator = EARAGOrchestrator(..., llm=llm)
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AnthropicLLM requires the 'anthropic' package: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
