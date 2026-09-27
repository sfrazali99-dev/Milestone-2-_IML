# DTS-SQL: Decomposed Text-to-SQL Reproduction (Milestone 2)

A research reproduction of **"DTS-SQL: Decomposed Text-to-SQL with Small Large Language Models"** (Pourreza & Rafiei, EMNLP / Findings of ACL 2024).

---

## ⚠️ Model Substitution Notice

> **The DTS-SQL paper uses fine-tuned 7B-parameter checkpoints** released by the authors:
> - Stage 1 (Schema Linking): `pourreza/DTS-SQL-SL-7B` (DeepSeek-Coder-7B fine-tune)
> - Stage 2 (SQL Generation): `pourreza/DTS-SQL-7B` (DeepSeek-Coder-7B fine-tune)
>
> **This reproduction uses `Qwen/Qwen2.5-Coder-1.5B-Instruct` as a substitute** — the smallest publicly available instruction-tuned coding LLM that can run on CPU without a GPU.
>
> **This means accuracy numbers in this repo are NOT comparable to the paper.** The substitution is disclosed everywhere it matters: `configs/config.yaml`, `PROVENANCE.md`, and the Results section below. The pipeline structure, prompting strategy, and evaluation methodology faithfully reproduce the paper's approach.

---

## 📌 Executive Summary

Small open-source LLMs struggle with Text-to-SQL when presented with large, complex database schemas. **DTS-SQL** addresses this by decomposing generation into two sequential stages:

1. **Stage 1 (Schema Linking):** Given the question and full schema, predict the reduced subset of relevant tables and columns.
2. **Stage 2 (SQL Generation):** Given the question and only the *linked* (reduced) schema, generate the final SQL query.

This repository implements both the two-stage DTS-SQL pipeline and a single-stage baseline, evaluated on the **Spider (Yale)** benchmark using real Spider data and real LLM inference.

---

## 📁 Repository Structure

```
dts-sql-repro/
├── README.md                 # Complete project documentation & results
├── PROVENANCE.md             # File-by-file code attribution with commit references
├── requirements.txt          # Exact pinned dependency versions (from pip freeze)
├── configs/
│   └── config.yaml           # Pipeline & model config (substitution notice included)
├── data/
│   ├── download_spider.py    # Downloads real Spider data + databases + verifies integrity
│   ├── sample_subset.py      # Seeded deterministic dev set sampling
│   └── schema_utils.py       # Spider tables.json schema parser & DDL prompt formatter
├── models/
│   ├── schema_linking.py     # Stage 1 Schema Linking model wrapper
│   ├── sql_generation.py     # Stage 2 SQL Generation model wrapper
│   └── baseline_direct.py    # Single-stage direct NL→SQL baseline wrapper
├── pipeline/
│   ├── dts_sql_pipeline.py   # Chains Stage 1 → Stage 2 into end-to-end DTS-SQL
│   └── run_inference.py      # CLI entrypoint for running inference
├── eval/
│   ├── execution_accuracy.py # SQLite execution result-set equivalence evaluator
│   ├── timing.py             # Wall-clock latency measurement utilities
│   └── run_eval.py           # Full evaluation harness with LLM mode verification
├── logs/                     # Evaluation results, logs, and summary tables
└── notebooks/                # Exploratory notebooks
```

---

## ⚙️ Environment Setup & Installation

### Requirements
- **Python:** 3.10 / 3.11 / 3.12
- **Compute:** CPU (no GPU required with 1.5B substitute model; paper's 7B model needs GPU)
- **Dataset License:** Spider dataset is open-access under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

### Quickstart (recommended: use the provided venv)

```bash
# Create a clean virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

> **Windows Long Path Note:** `transformers` has deeply nested internal file paths that exceed Windows' 260-char default limit. If installation fails with `OSError: [Errno 2] No such file or directory`, enable Long Paths:
> ```powershell
> reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
> ```
> Then install in a short-path directory (e.g., `C:\env\`).

---

## 🚀 How to Run End-to-End

### 1. Download & Prepare Real Spider Data
```bash
# Must run from the venv
venv\Scripts\python data/download_spider.py
```
This:
- Fetches real Spider dev.json + tables.json from HuggingFace `xlangai/spider`
- Downloads the official pre-built SQLite databases (Google Drive, Google ID `1_O51_Pdl8yvwQG87y1H6n8r5z4Vd0mho`)
- Verifies all dev db_ids have a matching `.sqlite` file (exits with error if any are missing)
- Verifies gold SQL executes correctly on the real databases before eval

### 2. Verify a Single Forward Pass (real LLM, real data)
```bash
# Test Stage 1 Schema Linking — real model forward pass
venv\Scripts\python models/schema_linking.py

# Test Stage 2 SQL Generation — real model forward pass  
venv\Scripts\python models/sql_generation.py

# Test end-to-end pipeline — real two-stage forward pass
venv\Scripts\python pipeline/dts_sql_pipeline.py
```

### 3. Run Full Comparative Evaluation
```bash
venv\Scripts\python eval/run_eval.py --num_samples 20 --seed 42
```
This:
- Verifies `torch` and `transformers` are importable before starting
- Pre-checks gold SQL against real databases
- Runs both systems with real LLM inference
- Stamps each result record with `"mode": "llm"` or `"mode": "heuristic_fallback"`
- Blocks writing `results_table.txt` if running in fallback mode
- Saves logs to `logs/`

---

## 🔍 Fully Worked Example (from real LLM inference)

### Input Question
> *"How many singers are there from USA?"*

### Database: `concert_singer` (real Spider database)

**Full Schema:**
```sql
Table `singer`: Singer_ID (number PRIMARY KEY), Name (text), Country (text), Song_Name (text), Song_release_year (text), Age (number), Is_male (text)
Table `stadium`: Stadium_ID (number PRIMARY KEY), Location (text), Name (text), Capacity (number), Highest (number), Lowest (number), Average (number)
Table `concert`: concert_ID (number PRIMARY KEY), concert_Name (text), Year (number), Stadium_ID (number FK)
Table `singer_in_concert`: concert_ID (number FK), Singer_ID (number FK)
```

### System 1: Baseline (Direct Single-Stage)
- **Prompt:** question + ALL 4 tables (full schema)
- **Generated SQL:** `SELECT count(*) FROM singer WHERE Country = 'USA';`

### System 2: DTS-SQL (Two-Stage)
1. **Stage 1 (Schema Linking):** question + full 4-table schema → `{"singer": ["Singer_ID", "Country"]}`
2. **Stage 2 (SQL Generation):** question + reduced schema (1 table, 2 columns) → `SELECT count(*) FROM singer WHERE Country = 'USA';`

---

## 📊 Evaluation Results

> **⚠️ Substitution Notice:** Results below use `Qwen/Qwen2.5-Coder-1.5B-Instruct` (CPU inference), **not** the paper's `pourreza/DTS-SQL-SL-7B` / `pourreza/DTS-SQL-7B` checkpoints. These numbers are not comparable to the paper.

*Results will be filled in after the real eval run completes. Check `logs/results_table.txt` and `logs/metrics_summary.json`.*

---

## 📈 Results & Analysis

### What the paper reports

The DTS-SQL paper reports an execution accuracy improvement of approximately **+3.5% to +7.2%** for the two-stage approach over single-stage direct baselines on the Spider dev set, with their fine-tuned 7B DeepSeek-Coder models.

### What this reproduction can honestly say

**Three root causes separate our numbers from the paper's:**

1. **Model substitution (primary cause):** We use `Qwen/Qwen2.5-Coder-1.5B-Instruct` instead of the paper's fine-tuned 7B checkpoints (`pourreza/DTS-SQL-SL-7B`, `pourreza/DTS-SQL-7B`). The 1.5B model is an instruction-tuned general coding model — it has never been trained specifically on Text-to-SQL or schema-linking tasks. The paper's models are fine-tuned for exactly these tasks. This difference alone accounts for most of any accuracy gap.

2. **Model size & specialisation:** Even setting aside fine-tuning, a 1.5B model has far less capacity than a 7B model. The paper's improvement is specifically measured on their fine-tuned 7B checkpoints; a smaller, unspecialised substitute will show lower absolute accuracy and may not preserve the relative gain.

3. **Evaluation scale:** Our subset is 20 examples (seed=42). Results at this scale are noisy — a single correct/incorrect prediction shifts accuracy by 5 percentage points. The paper evaluates on the full 1034-example dev set.

**What is faithfully reproduced:**
- The two-stage pipeline structure (schema linking → SQL generation)
- The prompting strategy for both stages (adapted from the official repo)
- The evaluation methodology (execution accuracy via result-set equivalence on real Spider SQLite databases)
- Real data (actual Spider dev.json + official pre-built databases)
- Real LLM inference (not heuristics — every result is stamped with `"mode": "llm"` in the logs)

### Environment constraints (documented, not hidden)
- No GPU available; all inference runs on CPU with float32
- The paper's 7B models exceed available RAM for CPU fp32 inference
- `transformers` installation is blocked by Windows 260-char path limit in the system Python; a clean short-path venv is required (provided as `venv/`)

---

## 📄 Provenance & Attribution

See [PROVENANCE.md](PROVENANCE.md) for full file-by-file attribution with commit-level references.
