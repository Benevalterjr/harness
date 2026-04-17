# harness

Harness de avaliação de LLM com recuperação vetorial, quantização e benchmark automatizado.

## Estrutura

- `src/harness/core.py`: classes principais (`MiniFaiss`, `TurboQuant`, `Harness`) e métricas (`Recall@K`, `MRR`, latência, custo por consulta).
- `src/harness/providers.py`: integração com Gemini (configuração de API, embedder e LLM).
- `scripts/run_evaluation.py`: execução automatizada de benchmark offline/reproduzível.
- `tests/test_harness_core.py`: testes unitários.
- `HARNESS.ipynb`: notebook de exploração usando o pacote `harness`.

## Ambiente reproduzível

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuração Gemini

```bash
export GEMINI_API_KEY="sua_chave"
```

## Testes

```bash
pytest -q
```

## Benchmark automatizado

```bash
python scripts/run_evaluation.py
```

Saída (JSON):
- `recall_at_k`
- `mrr_at_k`
- `latency_s`
- `cost_per_query`

## CI

Pipeline em GitHub Actions (`.github/workflows/ci.yml`) executa:
- matrix Python 3.10 / 3.11 / 3.12
- instalação de dependências + `pip install -e .`
- sanity checks (py_compile + validação do notebook sem `!pip install` em células de código)
- `pytest -q`
- smoke test de benchmark (`python scripts/run_evaluation.py`)
