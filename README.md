# harness

Harness de avaliação de LLM com recuperação vetorial, quantização e benchmark automatizado.

## Estrutura

- `harness_core.py`: classes principais (`MiniFaiss`, `TurboQuant`, `Harness`) e métricas (`Recall@K`, `MRR`, latência, custo por consulta).
- `providers.py`: integração com Gemini (configuração de API, embedder e LLM).
- `scripts/run_evaluation.py`: execução automatizada de benchmark offline/reproduzível.
- `tests/test_harness_core.py`: testes unitários.
- `HARNESS.ipynb`: notebook de exploração usando os módulos Python.

## Ambiente reproduzível

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

## Observações

- O notebook não instala mais dependências via `!pip install`; o setup fica centralizado em `requirements.txt`.
- Para CI, recomenda-se executar `pytest -q` e `python scripts/run_evaluation.py` em pipeline.

## CI

Pipeline em GitHub Actions (`.github/workflows/ci.yml`) executa:
- matrix Python 3.10 / 3.11 / 3.12
- instalação de dependências
- sanity checks (py_compile + validação do notebook sem `!pip install`)
- `pytest -q`
- smoke test de benchmark (`python scripts/run_evaluation.py`)
