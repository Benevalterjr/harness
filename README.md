# harness

[![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/harness.svg)](https://pypi.org/project/harness/)
[![Python](https://img.shields.io/pypi/pyversions/harness.svg)](https://pypi.org/project/harness/)

Harness de avaliação de LLM com recuperação vetorial, quantização e benchmark automatizado.

## Licença

Distribuído sob licença **MIT**. Veja `LICENSE`.

## Instalação

### Desenvolvimento local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Quando publicar no PyPI

```bash
pip install harness
```

## Estrutura

- `src/harness/core.py`: núcleo do pipeline (`Harness`, `MiniFaiss`, `TurboQuant`, métricas).
- `src/harness/providers.py`: providers (Gemini, OpenAI, Anthropic e HTTP local para Ollama/LM Studio gateway).
- `data/eval_dataset.jsonl`: dataset de benchmark determinístico offline.
- `scripts/run_evaluation.py`: benchmark automatizado.
- `tests/test_harness_core.py`: testes unitários.
- `HARNESS.ipynb`: notebook de demonstração.

## Classe `Harness` (resumo)

`Harness` conecta quatro peças:
1. `embed_fn(text) -> vector`
2. `retriever.search(vector, k)`
3. `llm.generate(prompt)`
4. `tokenizer.count(text)`

Saída de `run(query)`:
- `response`
- `prompt_tokens`
- `response_tokens`
- `tokens`
- `docs` (top-k recuperado)

## Configuração Gemini

```bash
export GEMINI_API_KEY="sua_chave"
```

## Benchmark offline determinístico

```bash
python scripts/run_evaluation.py --dataset data/eval_dataset.jsonl
```

Métricas:
- `recall_at_k`
- `mrr_at_k`
- `latency_s`
- `cost_per_query`

## Testes

```bash
pytest -q
```

## CI

Pipeline em GitHub Actions (`.github/workflows/ci.yml`) executa:
- matrix Python 3.10 / 3.11 / 3.12
- instalação de dependências + `pip install -e .`
- sanity checks (py_compile + validação do notebook sem `!pip install` em células de código)
- `pytest -q`
- smoke test de benchmark

## Publicação no PyPI

Workflow pronto em `.github/workflows/publish.yml` (dispara em release publicada).
