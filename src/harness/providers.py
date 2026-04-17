from __future__ import annotations

import os
from typing import Callable

import numpy as np


class GeminiLLM:
    def __init__(self, model) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class GeminiEmbedder:
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
