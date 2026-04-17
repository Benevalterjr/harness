from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


class GeminiLLM:
    """Wrapper para geração de texto com um modelo Gemini já instanciado."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class GeminiEmbedder:
    """Embedding provider para Gemini."""

    def __init__(self, model_name: str = "models/gemini-embedding-001", output_dimensionality: int = 768) -> None:
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality

    def __call__(self, text: str) -> np.ndarray:
        import google.generativeai as genai

        result = genai.embed_content(
            model=self.model_name,
            content=text,
            output_dimensionality=self.output_dimensionality,
        )
        return np.array(result["embedding"])


class OpenAILLM:
    """Provider de geração para OpenAI Responses API."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text


class AnthropicLLM:
    """Provider de geração para Anthropic Messages API."""

    def __init__(self, model: str = "claude-3-5-haiku-latest", api_key: str | None = None) -> None:
        from anthropic import Anthropic

        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        first = message.content[0]
        return first.text if hasattr(first, "text") else str(first)


@dataclass
class LocalHTTPLLM:
    """Provider simples para endpoints locais compatíveis com JSON HTTP (ex.: Ollama/LM Studio gateway)."""

    url: str = "http://localhost:11434/api/generate"
    model: str = "llama3"

    def generate(self, prompt: str) -> str:
        import requests

        resp = requests.post(self.url, json={"model": self.model, "prompt": prompt, "stream": False}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("output") or str(data)


def configure_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Defina GEMINI_API_KEY no ambiente antes de executar.")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return api_key


def load_model_with_fallback(primary_model_name: str, fallback_model_name: str = "models/gemini-2.5-flash"):
    import google.generativeai as genai

    try:
        return genai.GenerativeModel(primary_model_name)
    except Exception:
        return genai.GenerativeModel(fallback_model_name)
