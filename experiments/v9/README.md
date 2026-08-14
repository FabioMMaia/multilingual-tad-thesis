# v9 — Ablation: MLP + MiniLM pré-treinado (sem fine-tuning)

## Objetivo

Isolar a contribuição do backbone (MiniLM > distiluse) da contribuição do contrastive fine-tuning (SetFit).

| Versão | Labels | Encoder | SetFit |
|---|---|---|---|
| v7 | v5 (LLM) | distiluse-v2 | ❌ |
| v8 | v5 (LLM) | MiniLM (fine-tuned) | ✅ |
| **v9** | v5 (LLM) | **MiniLM (pré-treinado)** | ❌ |

---

## Hipótese

Se v9 ≈ v8: o ganho de v8 sobre v7 vem do backbone melhor, **não** do fine-tuning contrastivo.  
Se v9 < v8: o fine-tuning contrastivo contribui de forma independente do backbone.

---

## Posição no 3×1

| Encoder | Fine-tune? | Versão |
|---|---|---|
| distiluse-v2 | ❌ | v7 |
| MiniLM | ❌ | **v9** |
| MiniLM | ✅ | v8 |

---

## Config

| Parâmetro | Valor |
|---|---|
| Datasets | tweets_hs, hatebr, 20_newsgroups, wikinews |
| Labels source | `data/llm_results/v5/{strategy}/N_{n}/{dataset}_llm_labels.csv` |
| SetFit | ❌ (`--no_setfit`) |
| Reencoder | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`--reencoder`) |
| Modelo AD | MLP (`--semisup_model mlp`) |
| Strategies | random, diversity |
| N | 50, 200 |
| Seeds | 0, 1, 42 |
| Total runs | 48 |
| Hardware | Colab T4 |
| Tempo estimado | ~1h (sem SetFit, encoding rápido) |

Results: `data/llm_results/v9/{strategy}/N_{n}/{dataset}.csv`

---

## Comando equivalente (por run)

```bash
python scripts/run_llm_active_loop.py \
    --project_path PROJECT_PATH \
    --data_dir DATA_DIR \
    --dataset {dataset} \
    --strategy {strategy} \
    --n_llm_calls {n} \
    --seed {seed} \
    --device cuda \
    --load_labels_from .../v5/{strategy}/N_{n}/{dataset}_llm_labels.csv \
    --load_labels_model qwen2.5-14b \
    --no_setfit \
    --reencoder sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --semisup_model mlp \
    --results_dir data/llm_results/v9
```

---

## Interpretação esperada dos resultados

| Resultado | Interpretação |
|---|---|
| v9 ≈ v8 > v7 | Backbone explica o ganho; fine-tuning não adiciona |
| v8 > v9 > v7 | Backbone + fine-tuning contribuem independentemente |
| v9 ≈ v7 | MiniLM sem fine-tune equivale ao distiluse; ganho de v8 é todo do contrastive |
| v9 < v7 | MiniLM sem fine-tune é pior que distiluse (improvável) |
