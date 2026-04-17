import numpy as np

from harness_core import (
    Harness,
    MiniFaiss,
    Tokenizer,
    TurboQuant,
    evaluate_harness,
    mrr_at_k,
    quantization_error,
    recall_at_k,
)


class DummyLLM:
    def generate(self, prompt: str) -> str:
        return "ok"


def deterministic_embed(text: str, dim: int = 768) -> np.ndarray:
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float32)


def test_minifaiss_validates_dimensions() -> None:
    db = MiniFaiss(dim=768)
    with np.testing.assert_raises(ValueError):
        db.add(np.random.rand(128), "x")


def test_quantization_roundtrip_has_small_error() -> None:
    vectors = np.random.rand(10, 768).astype(np.float32)
    tq = TurboQuant(bits=8)
    tq.fit(vectors)
    q = tq.quantize(vectors[0])
    dq = tq.dequantize(q)

    assert quantization_error(vectors[0], dq) < 0.001


def test_metrics_pipeline_returns_expected_keys() -> None:
    docs = [
        "TurboQuant reduz custo de embeddings",
        "MiniFaiss permite busca vetorial leve",
        "LLMs podem ser avaliados com harness",
    ]

    db = MiniFaiss(dim=768)
    for doc in docs:
        db.add(deterministic_embed(doc), doc)

    harness = Harness(DummyLLM(), db, Tokenizer(), deterministic_embed)

    result = harness.run("Como reduzir custo de embeddings?")
    assert "docs" in result
    assert len(result["docs"]) > 0

    queries = [
        ("Como reduzir custo de embeddings?", "TurboQuant"),
        ("Como fazer busca vetorial leve?", "MiniFaiss"),
    ]
    metrics = evaluate_harness(harness, queries, top_k=3, input_cost_per_1k_tokens=0.001, output_cost_per_1k_tokens=0.002)

    assert set(metrics.keys()) == {"recall_at_k", "mrr_at_k", "latency_s", "cost_per_query"}
    assert 0 <= metrics["recall_at_k"] <= 1
    assert 0 <= metrics["mrr_at_k"] <= 1
    assert metrics["latency_s"] >= 0
    assert metrics["cost_per_query"] >= 0


def test_recall_and_mrr() -> None:
    retrieved = [(0.9, "a"), (0.8, "TurboQuant doc"), (0.7, "c")]
    assert recall_at_k(retrieved, "TurboQuant") == 1.0
    assert mrr_at_k(retrieved, "TurboQuant") == 0.5
