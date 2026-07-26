# Bootstrapping Text Anomaly Detection with LLM-Generated Weak Supervision

Official code, prompts, and experimental configuration for the STIL 2026 paper:

> **Bootstrapping Text Anomaly Detection with LLM-Generated Weak Supervision**  
> Fabio Masaracchia Maia, Anna Helena Reali Costa  
> Escola Politécnica, Universidade de São Paulo

---

## Overview

Semi-supervised text anomaly detection achieves strong performance — but only when labeled anomalies are available. This work removes that requirement by using a compact, locally deployed LLM as a **noisy annotator** over real unlabeled data.

**The pipeline (no human labels required):**

1. Encode all documents with a pre-trained multilingual sentence encoder (`distiluse-base-multilingual-cased-v2`).
2. Select a small candidate set *S* (N=200) from the training pool via **random** or **diversity** (k-means coverage) sampling.
3. Query a local LLM (Qwen2.5-Instruct 7B/14B, 4-bit GGUF) once per candidate — it returns a continuous anomaly score and a one-sentence explanation.
4. Binarize scores at τ=0.5 to produce weak labels *Â*.
5. If |*Â*| ≥ 8: fine-tune the encoder contrastively via **SetFit** using *Â*.
6. Re-encode the full dataset with the refined encoder.
7. Train a lightweight **MLP** binary classifier (or DeepSAD) using *Â* as labeled anomalies.
8. Evaluate on the held-out test set (ground-truth labels used **only here**).

**Key results** across four datasets (two languages, two tasks):

| Dataset | Best unsup. AUC | Ours (MLP + Qwen 14B) | Oracle (GT labels) | Gap recovered |
|---|---|---|---|---|
| 20 Newsgroups | 0.920 | **0.967** | 0.998 | 60% |
| WikiNews (PT) | 0.751 | **0.885** | 0.943 | 70% |
| HS Tweets (EN) | 0.575 | **0.867** | 0.946 | 79% |
| HateBR (PT) | 0.561 | **0.742** | 0.886 | 56% |

Mean AUC rises from **0.702 → 0.865** with zero human annotation, using only 7–16 LLM-flagged anomalies per run.

---

## Repository Structure

```
bootstrapping-text-anomaly-detection/
├── main.tex                        # Paper source (STIL 2026)
├── requirements.txt
├── scripts/
│   ├── run_data_preparation.py     # Step 0 — encode datasets, save parquets
│   └── run_llm_active_loop.py      # Step 1+ — full pipeline (LLM → SetFit → detector)
└── src/
    ├── pipeline/
    │   ├── llm_runner.py           # LLMAnnotator, TASK_CONTEXT prompts, run_llm_active_loop
    │   ├── anomaly_detection.py    # Data preprocessing utilities
    │   └── data_handler.py         # Dataset loading (HuggingFace, Kaggle, WikiNews)
    ├── models/
    │   └── MLP.py                  # Lightweight MLP binary classifier (PyTorch)
    └── encoders/
        └── text_encoders.py        # SentenceBERT / BERTimbau wrappers
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For local GGUF inference (no API key required):
```bash
# CPU-only
pip install llama-cpp-python

# GPU (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

### 2. Prepare data (encode datasets → parquet files)

```bash
python scripts/run_data_preparation.py --project_path .
```

This downloads the four datasets from HuggingFace / Kaggle, encodes them with `distiluse-base-multilingual-cased-v2`, and saves `texts_*.parquet`, `labels_*.parquet`, and `embeddings_*.parquet` into `data/`.

To prepare a single dataset:
```bash
python scripts/run_data_preparation.py --dataset_filter 20_newsgroups
```

### 3. Run the pipeline

#### Option A — Local GGUF model (recommended — no API key, no token limits)

This is the setup used in the paper. Download the Qwen2.5-7B model (~4.7 GB, sharded):

```bash
mkdir -p models
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -O models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf -O models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
```

```bash
python scripts/run_llm_active_loop.py \
    --dataset 20_newsgroups \
    --backend llamacpp \
    --llamacpp_model qwen2.5-7b \
    --strategy random \
    --n_llm_calls 200 \
    --device cpu
```

For GPU inference, add `--llamacpp_n_gpu_layers -1 --device cuda`.

#### Option B — OpenAI API

```bash
python scripts/run_llm_active_loop.py \
    --dataset 20_newsgroups \
    --backend openai \
    --api_key YOUR_OPENAI_KEY \
    --strategy random \
    --n_llm_calls 200
```

#### Option C — Google Colab (GPU recommended)

```python
!python scripts/run_llm_active_loop.py \
    --project_path "/content/drive/MyDrive/bootstrapping-text-anomaly-detection" \
    --dataset hatebr \
    --backend llamacpp \
    --llamacpp_model qwen2.5-7b \
    --strategy diversity \
    --n_llm_calls 200 \
    --llamacpp_n_gpu_layers -1 \
    --device cuda
```

### 4. Reproduce paper results (all datasets × seeds)

```bash
for dataset in 20_newsgroups wikinews tweets_hs hatebr; do
  for strategy in random diversity; do
    for seed in 42 123 456; do
      python scripts/run_llm_active_loop.py \
        --dataset $dataset \
        --backend llamacpp \
        --llamacpp_model qwen2.5-14b \
        --strategy $strategy \
        --n_llm_calls 200 \
        --seed $seed \
        --semisup_model mlp \
        --device cuda
    done
  done
done
```

Results are saved as CSV files in `data/llm_results/`.

---

## Supported Datasets

| Argument | Dataset | Language | Task |
|---|---|---|---|
| `20_newsgroups` | 20 Newsgroups (rec.sport.hockey vs. rest) | EN | Topic classification |
| `wikinews` | WikiNews PT (Politics vs. rest) | PT | Topic classification |
| `tweets_hs` | Tweets Hate Speech Detection | EN | Hate speech |
| `hatebr` | HateBR | PT | Hate speech |

All datasets are publicly available and downloaded automatically. WikiNews is extracted from a Wikimedia dump (~500 MB download, parsed automatically on first run).

---

## LLM Backends

| Backend | Key required | Cost | Notes |
|---|---|---|---|
| `llamacpp` | No | Free (local) | Qwen2.5-7B/14B, GPU recommended |
| `openai` | Yes (paid) | Paid | `openai` package |
| `gemini` | Yes (free tier) | Free (geo-restricted) | `google-genai` package |

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `told_br` | Dataset name (see table above) |
| `--backend` | `llamacpp` | LLM backend |
| `--strategy` | `score_guided` | Sampling: `random`, `diversity`, `score_guided` |
| `--n_llm_calls` | `100` | Annotation budget N |
| `--semisup_model` | `deepsad` | Downstream model: `mlp` or `deepsad` |
| `--seed` | `42` | Random seed |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--no_setfit` | off | Skip contrastive fine-tuning (ablation) |
| `--load_labels_from` | None | Reuse saved LLM labels CSV (ablation) |

See `python scripts/run_llm_active_loop.py --help` for the full list.

---

## Adding a New Dataset

1. Add a new entry in `TASK_CONTEXT` in `src/pipeline/llm_runner.py` with `description`, `normal_description`, and `anomaly_criterion`.
2. Add the dataset loading logic in `scripts/run_data_preparation.py` (HuggingFace, Kaggle, or a custom loader).
3. Add the dataset key to the `--dataset` choices in `scripts/run_llm_active_loop.py`.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{maia2026bootstrapping,
  title     = {Bootstrapping Text Anomaly Detection with LLM-Generated Weak Supervision},
  author    = {Maia, Fabio Masaracchia and Costa, Anna Helena Reali},
  booktitle = {Symposium in Information and Human Language Technology (STIL)},
  year      = {2026},
  address   = {Brazil}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

