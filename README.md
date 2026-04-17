# harness

Protótipo de harness para avaliação de LLM com:
- recuperação vetorial simplificada (`MiniFaiss`)
- quantização (`TurboQuant`)
- geração com Gemini

## Diagnóstico atual

### O que funciona
- Estrutura didática no notebook (`HARNESS.ipynb`) com pipeline completo de ingestão, recuperação e geração.
- Embedding real via `models/gemini-embedding-001` já integrado.
- Quantização e cálculo de erro/compressão implementados.
- Estratégia básica de retry para limite de taxa (429) durante ingestão.

### O que não funcionava (e foi ajustado)
- **Chave de API hardcoded** no notebook (risco de segurança).
- **Incompatibilidade de dimensão** entre consulta (128) e índice vetorial (768), quebrando `harness.run(...)`.
- Falta de validação de dimensão no índice vetorial, o que dificulta diagnóstico de erro.

## Melhorias aplicadas
- Remoção de chave fixa no código; agora usa `GEMINI_API_KEY` do ambiente.
- `MiniFaiss` padronizado para dimensão 768 com validação explícita.
- `Harness.embed()` atualizado para usar embedding real (`embed_text`) em vez de vetor aleatório.
- Fallback de modelo atualizado para nome atual (`models/gemini-2.5-flash`).

## Como executar

```bash
export GEMINI_API_KEY="sua_chave"
```

Depois rode as células do `HARNESS.ipynb` em ordem.

## Próximos passos recomendados
- Extrair classes do notebook para módulos `.py` e adicionar testes unitários.
- Versionar dependências (`requirements.txt` ou `pyproject.toml`).
- Substituir `!pip install` dentro do notebook por ambiente reproduzível.
- Adicionar métricas de avaliação (Recall@K, MRR, latência, custo por consulta) em execução automatizada.
