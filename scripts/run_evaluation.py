from __future__ import annotations

import argparse
import json

import numpy as np

from harness_core import Harness, MiniFaiss, Tokenizer, TurboQuant, evaluate_harness


class EchoLLM:
    def generate(self, prompt: str) -> str:
        return "Resposta sintética para benchmark"


def deterministic_embed(text: str, dim: int = 768) -> np.ndarray:
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa benchmark automatizado do harness")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--input-cost", type=float, default=0.0003, help="USD por 1k tokens de entrada")
    parser.add_argument("--output-cost", type=float, default=0.0006, help="USD por 1k tokens de saída")
    args = parser.parse_args()

    docs = [
        "TurboQuant reduz custo de embeddings",
        "MiniFaiss permite busca vetorial leve",
        "Gemini 3 tem raciocínio avançado",
        "LLMs podem ser avaliados com harness",
    ]

    retriever = MiniFaiss(dim=768)
    vectors = np.array([deterministic_embed(d) for d in docs])

    tq = TurboQuant(bits=8)
    tq.fit(vectors)

    for doc, vec in zip(docs, vectors):
        retriever.add(tq.dequantize(tq.quantize(vec)), doc)

    harness = Harness(llm=EchoLLM(), retriever=retriever, tokenizer=Tokenizer(), embed_fn=deterministic_embed)

    queries = [
        ("Como reduzir custo de embeddings?", "TurboQuant"),
        ("Como fazer busca vetorial leve?", "MiniFaiss"),
    ]

    metrics = evaluate_harness(
        harness,
        queries,
        top_k=args.top_k,
        input_cost_per_1k_tokens=args.input_cost,
        output_cost_per_1k_tokens=args.output_cost,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
