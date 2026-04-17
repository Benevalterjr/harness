from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, time
from typing import Callable, Iterable, Protocol

import numpy as np


class EmbeddingFn(Protocol):
    def __call__(self, text: str) -> np.ndarray: ...


class Tokenizer:
    def __init__(self) -> None:
        try:
            import tiktoken

            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.enc = None

    def count(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())


class MiniFaiss:
    """Índice vetorial em memória com similaridade de cosseno."""
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.vectors: list[np.ndarray] = []
        self.texts: list[str] = []

    def _validate_dim(self, vector: np.ndarray) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"Dimensão inválida: esperado {self.dim}, recebido {len(vector)}")

    def add(self, vector: np.ndarray, text: str) -> None:
        self._validate_dim(vector)
        self.vectors.append(vector)
        self.texts.append(text)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[float, str]]:
        self._validate_dim(query_vector)
        sims: list[tuple[float, str]] = []
        for i, v in enumerate(self.vectors):
            sim = float(np.dot(query_vector, v) / (np.linalg.norm(query_vector) * np.linalg.norm(v) + 1e-9))
            sims.append((sim, self.texts[i]))

        sims.sort(reverse=True)
        return sims[:k]


class TurboQuant:
    """Quantização linear simples para reduzir memória de vetores."""
    def __init__(self, bits: int = 8) -> None:
        self.bits = bits
        self.scale: float | None = None

    def fit(self, vectors: np.ndarray) -> None:
        max_val = np.max(np.abs(vectors))
        self.scale = max_val / (2 ** (self.bits - 1) - 1)

    def quantize(self, vector: np.ndarray) -> np.ndarray:
        if self.scale is None:
            raise ValueError("TurboQuant não foi ajustado. Rode fit() antes.")
        return np.round(vector / self.scale).astype(np.int8)

    def dequantize(self, q_vector: np.ndarray) -> np.ndarray:
        if self.scale is None:
            raise ValueError("TurboQuant não foi ajustado. Rode fit() antes.")
        return q_vector.astype(np.float32) * self.scale


@dataclass
class TraceStep:
    step: str
    timestamp: float
    data: object


class Trace:
    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def log(self, step: str, data: object) -> None:
        self.steps.append(TraceStep(step=step, timestamp=time(), data=data))


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class Harness:
    """Pipeline de retrieval + prompting + geração + contagem de tokens."""
    def __init__(self, llm: LLM, retriever: MiniFaiss, tokenizer: Tokenizer, embed_fn: EmbeddingFn) -> None:
        self.llm = llm
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.embed_fn = embed_fn
        self.trace = Trace()

    def embed(self, text: str) -> np.ndarray:
        return self.embed_fn(text)

    def run(self, query: str, k: int = 3) -> dict[str, object]:
        query_vec = self.embed(query)
        docs = self.retriever.search(query_vec, k=k)
        self.trace.log("retrieval", docs)

        context = "\n".join([d[1] for d in docs])
        prompt = f"Contexto:\n{context}\n\nPergunta:\n{query}\n"
        response = self.llm.generate(prompt)
        self.trace.log("llm_response", response)

        prompt_tokens = self.tokenizer.count(prompt)
        response_tokens = self.tokenizer.count(response)

        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "tokens": prompt_tokens + response_tokens,
            "docs": docs,
        }


def quantization_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))


def compression_ratio(original_bits: int = 32, quantized_bits: int = 8) -> float:
    return original_bits / quantized_bits


def recall_at_k(retrieved: list[tuple[float, str]], expected_substring: str) -> float:
    return float(any(expected_substring in r[1] for r in retrieved))


def mrr_at_k(retrieved: list[tuple[float, str]], expected_substring: str) -> float:
    for rank, (_, text) in enumerate(retrieved, start=1):
        if expected_substring in text:
            return 1.0 / rank
    return 0.0


def evaluate_harness(
    harness: Harness,
    queries: Iterable[tuple[str, str]],
    *,
    top_k: int = 3,
    input_cost_per_1k_tokens: float = 0.0,
    output_cost_per_1k_tokens: float = 0.0,
) -> dict[str, float]:
    recalls: list[float] = []
    mrrs: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []

    for query, expected in queries:
        start = perf_counter()
        result = harness.run(query, k=top_k)
        elapsed = perf_counter() - start
        latencies.append(elapsed)

        docs = result["docs"]
        recalls.append(recall_at_k(docs, expected))
        mrrs.append(mrr_at_k(docs, expected))

        input_cost = (result["prompt_tokens"] / 1000.0) * input_cost_per_1k_tokens
        output_cost = (result["response_tokens"] / 1000.0) * output_cost_per_1k_tokens
        costs.append(input_cost + output_cost)

    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "mrr_at_k": float(np.mean(mrrs)) if mrrs else 0.0,
        "latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "cost_per_query": float(np.mean(costs)) if costs else 0.0,
    }
