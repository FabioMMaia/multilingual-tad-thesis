# Implementation Details

This document covers the technical implementation of the pipeline described in the paper. It is intended for researchers who want to reproduce the experiments, extend the method, or adapt it to new datasets.

---

## Pipeline Overview

The full pipeline is implemented in `src/pipeline/llm_runner.py` (`run_llm_active_loop`) and orchestrated by `scripts/run_llm_active_loop.py`.

```
                ┌───────────────────────────────────────────────────────┐
                │               Unlabeled corpus  X                     │
                └───────────────┬───────────────────────────────────────┘
                                │
                        [Encode with distiluse-v2]
                                │
                ┌───────────────▼───────────────────────────────────────┐
                │       Pre-computed embeddings  E  (N × 512)           │
                └───────┬───────────────────────────────────────────────┘
                        │ 80/20 split (stratified by ground-truth label)
              ┌─────────▼──────────┐         ┌──────────────────────────┐
              │   Train pool       │         │   Test set (locked)      │
              │   E_train, X_train │         │   E_test,  X_test        │
              └─────────┬──────────┘         └──────────────────────────┘
                        │
        [Step 2] Unsupervised DeepSVDD on E_train (only for score_guided)
                        │
        [Step 3] Select N candidates  S ⊂ X_train
                 via random / diversity / score_guided
                        │
        [Step 4] Query LLM: score_i ∈ [0,1] per candidate
                        │
        [Step 5] Binarize at τ = 0.5  →  weak labels Â
                        │
                 |Â| ≥ 8?
              ┌──┴──┐
             YES    NO
              │      └──► use original embeddings
        [Step 6] SetFit fine-tuning on balanced (Â_normal, Â_anomaly)
              │
        [Step 7] Re-encode X_train + X_test with SetFit encoder
              │
        [Step 8] Train MLP / DeepSAD on (E_train_sf, Â ∪ unlabeled)
              │
        [Step 9] Evaluate on (E_test_sf, y_test_gt) → ROC-AUC, PR-AUC
```

Ground-truth labels (`y_test_gt`) are **never** used during training — only for the final evaluation in Step 9.

---

## Data Preparation

**Script:** `scripts/run_data_preparation.py`

Produces three parquet files per dataset:

| File | Content |
|---|---|
| `data/texts_{dataset}.parquet` | Raw text strings (column: `text`) |
| `data/labels_{dataset}.parquet` | Original multi-class labels (column: `label`) |
| `data/embeddings_{dataset}_{encoder}.parquet` | Pre-computed embeddings (one row per sample, 512 columns) |

The encoder used in the paper is `sentence-transformers/distiluse-base-multilingual-cased-v2` (512-dim, multilingual).

Embeddings are computed once and reused across all experiments. This decouples the expensive encoding step from the fast iterative experimentation.

---

## Dataset Preprocessing

**Source:** `src/pipeline/anomaly_detection.py`

Two transformations are applied before any experiment:

1. **Binary labeling** (`label_normal_vs_anomaly`): the most frequent class becomes the normal class (label 0); all other classes are merged into the anomaly class (label 1).

2. **Contamination adjustment** (`adjust_contamination`): anomalies are downsampled to reach ρ ≈ 5% of the training set. This reflects a realistic semi-supervised setting where labeled anomalies are scarce. The contamination rate can be changed via `--contamination`.

---

## LLM Annotation

**Source:** `src/pipeline/llm_runner.py` — `LLMAnnotator`, `TASK_CONTEXT`

### Prompt structure

Each candidate document is formatted into a structured JSON-output prompt:

```
System: You are a text anomaly detector. [task description]
        NORMAL: [normal_description]
        ANOMALY: [anomaly_criterion]
        Respond ONLY with valid JSON: {"anomaly_score": <0.0-1.0>, "reason": "<one sentence>"}

User: Text: """<document>"""
```

The `anomaly_score` is a continuous value in [0, 1]. It is binarized at τ = 0.5 to produce a weak label (`llm_label`).

### Dataset-specific prompts (`TASK_CONTEXT`)

Each dataset has an entry in `TASK_CONTEXT` with three fields:

- `description`: one-line task framing (what the LLM is deciding)
- `normal_description`: what counts as normal (score → 0.0)
- `anomaly_criterion`: what counts as anomalous (score → 1.0), with examples

The prompts are deliberately asymmetric: they describe normality and anomaly in natural language without showing labeled examples. Careful prompt design is critical — the WikiNews (7B) failure mode in the paper is caused by prompt ambiguity that leads to label polarity inversion.

### Backend implementations

| Backend | Class / module | Notes |
|---|---|---|
| `gemini` | `google.genai` | Default model: `gemini-2.5-flash-lite` |
| `openai` | `openai.OpenAI` | Default model: `gpt-4o-mini` |
| `llamacpp` | `llama_cpp.Llama` | Local GGUF; grammar-constrained JSON output |

For `llamacpp`, JSON output is enforced via llama.cpp grammar (`LlamaGrammar.from_string`), making parsing more robust.

---

## Sampling Strategies

**Source:** `src/pipeline/llm_runner.py` — `select_samples`

| Strategy | Description | When to use |
|---|---|---|
| `random` | Uniform random sample of N candidates | Baseline; simple and reproducible |
| `diversity` | k-means++ partitions embedding space into N clusters; selects the centroid-nearest instance per cluster | Ensures peripheral (anomaly-prone) regions are covered |
| `score_guided` | Selects N candidates with highest unsupervised anomaly scores (DeepSVDD) | Biases toward likely anomalies; requires a pre-trained unsup. model |

The paper evaluates `random` and `diversity` at N=200 and finds no statistically significant difference in downstream AUC. Results are aggregated across both for variance reduction.

---

## Contrastive Fine-Tuning (SetFit)

**Source:** `src/pipeline/llm_runner.py` — inside `run_llm_active_loop`

SetFit fine-tuning is activated when `|Â| ≥ 8` (at least 8 LLM-labeled anomalies). It uses the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model by default.

**Balancing:** the majority class is downsampled to match the minority class before SetFit training, so contrastive pairs are equally drawn from both classes.

**Hyperparameters (fixed):**
- `num_iterations = 10` (contrastive pairs per sample)
- `num_epochs = 1`
- `batch_size = 4`

After fine-tuning, the full training and test sets are re-encoded using the updated SetFit model. The original `distiluse` embeddings are replaced for all downstream steps.

SetFit is skipped (`--no_setfit`) in ablation experiments to isolate the contribution of contrastive fine-tuning from the LLM label signal.

---

## Anomaly Detection Models

**Source:** `src/models/MLP.py`, `deepod` library

### MLP (proposed configuration)

A shallow PyTorch MLP binary classifier:
- Architecture: `512 → 128 → 64 → 1` with ReLU activations and sigmoid output
- Loss: binary cross-entropy (`nn.BCELoss`)
- Optimizer: Adam, lr=1e-3
- Training: 50 epochs, batch size 256

The BCE loss averages over all training samples (thousands of clean negatives), which dilutes the effect of mislabeled positives. This makes MLP substantially more robust to noisy labels than DeepSAD.

**Training signal:**
- Samples in *Â* (LLM-labeled): used with their weak labels
- Remaining training samples: used as normal (label 0), providing a dense normal signal

### DeepSAD

Uses the `deepod` implementation with:
- 3-layer MLP encoder: `512 → 100 → 50 → 128` with ReLU activations
- Hypersphere loss: pulls normal samples toward center, pushes known anomalies away
- 100 training epochs

DeepSAD is more sensitive to mislabeled positives than MLP because a single false positive directly distorts the hypersphere boundary.

---

## Ablation Flags

The script supports two ablation modes designed to isolate pipeline components:

### No-SetFit ablation (`--no_setfit`)

Skips contrastive fine-tuning entirely and uses the original `distiluse` embeddings for the downstream model. Combined with `--load_labels_from`, this isolates the contribution of encoder refinement.

```bash
python scripts/run_llm_active_loop.py \
    --dataset tweets_hs \
    --backend llamacpp \
    --llamacpp_model qwen2.5-14b \
    --no_setfit \
    --load_labels_from data/llm_results/v5/tweets_hs_labels.csv
```

### Reuse labels (`--load_labels_from`)

Loads pre-saved LLM labels from a CSV file instead of querying the LLM again. The CSV must contain columns: `seed`, `llm_score`, `llm_label`, `text`. Use `--load_labels_model` when the CSV contains runs from multiple models.

This is how we ran all ablation cells in the paper (same labels, different downstream configurations) without re-running LLM inference.

---

## Output Format

Each run appends one row to a CSV in `data/llm_results/`:

| Column | Description |
|---|---|
| `run_id` | UUID for the run |
| `dataset` | Dataset name |
| `strategy` | Sampling strategy |
| `n_llm_calls` | Annotation budget N |
| `seed` | Random seed |
| `roc_auc` | Test ROC-AUC |
| `pr_auc` | Test PR-AUC |
| `n_anomalies_found` | LLM-labeled anomalies |
| `n_normals_found` | LLM-labeled normals |
| `n_parse_errors` | Failed LLM responses |
| `llm_precision` | LLM precision vs. ground truth (diagnostic) |
| `llm_recall` | LLM recall vs. ground truth (diagnostic) |
| `llm_agreement` | LLM label agreement rate vs. ground truth |
| `semisup_model` | Model used (`mlp` or `deepsad`) |
| `setfit_skipped` | Whether SetFit was activated |
| `llm_model` | LLM model identifier |
| `elapsed_seconds` | Wall-clock time |

A companion `*_llm_labels.csv` is also saved with the per-sample LLM outputs (text, score, label, reason) for ablation and analysis.

---

## Reproducing Paper Numbers

The paper reports mean ± std over **6 runs** per configuration: 3 seeds × 2 sampling strategies (random + diversity), both at N=200. The `score_guided` strategy was evaluated separately (it requires a pre-trained DeepSVDD and does not use the diversity embedding coverage).

Seed values used: `42`, `123`, `456`.

Example command to reproduce the MLP + Qwen 14B row for HS Tweets:
```bash
for seed in 42 123 456; do
  for strategy in random diversity; do
    python scripts/run_llm_active_loop.py \
      --dataset tweets_hs \
      --backend llamacpp \
      --llamacpp_model qwen2.5-14b \
      --strategy $strategy \
      --n_llm_calls 200 \
      --seed $seed \
      --semisup_model mlp \
      --device cuda
  done
done
```

Aggregate the resulting CSV rows to compute mean ± std AUC.

---

## Failure Modes (from paper §4.3)

Three annotation regimes were identified:

1. **Implicit vocabulary gaps** (20 Newsgroups, Qwen 7B): near-perfect recall but very low precision — the LLM floods *Â* with normal hockey posts it misclassifies due to unfamiliar domain jargon (team abbreviations, NHL slang). The MLP absorbs this noise; DeepSAD does not.

2. **Criterion polarity error** (WikiNews, Qwen 7B): the LLM correctly identifies article semantics but assigns inverted label polarity for some normal political articles. SetFit amplifies this directional error (ΔAUCreduced by 0.069). The 14B model avoids this failure.

3. **Cross-lingual calibration mismatch** (HateBR): the 14B model applies a higher decision threshold and misses culturally specific Brazilian Portuguese offense markers that the 7B correctly flags. Counter-intuitively, the 7B outperforms the 14B on this dataset.

**Practical implication:** verify prompt alignment and inspect a few LLM explanations before scaling annotation. Quantitative noise (many false positives) is more tolerable than directional noise (label polarity inversion).
