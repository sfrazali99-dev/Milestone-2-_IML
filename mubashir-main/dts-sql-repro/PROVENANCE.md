# Code Provenance & Component Tracking

This document details the exact origin and attribution of all code, utility functions, and model interfaces in the DTS-SQL reproduction repository, in accordance with research reproduction standards.

---

## Provenance Taxonomy

Each component is classified as one of:
1. `written by us` — Original code authored specifically for this reproduction project.
2. `adapted from [repo@commit, file]` — Code modified or translated from external sources.
3. `reused as-is from [repo@commit, file]` — Direct copies of external scripts without modification.

---

## Model Checkpoint Substitution Notice

The DTS-SQL paper (Pourreza & Rafiei, 2024) uses the following fine-tuned 7B-parameter checkpoints:

| Stage | Paper Checkpoint | HuggingFace ID |
|---|---|---|
| Stage 1 — Schema Linking | DTS-SQL-SL, fine-tuned on DeepSeek-Coder-7B | `pourreza/DTS-SQL-SL-7B` |
| Stage 2 — SQL Generation | DTS-SQL, fine-tuned on DeepSeek-Coder-7B | `pourreza/DTS-SQL-7B` |
| Baseline | Same 7B model, prompted end-to-end | `pourreza/DTS-SQL-7B` |

**Substitute used in this reproduction:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Reason:** Hardware constraints. No GPU available; 7B models exceed available RAM in fp32 on CPU.
- **Impact:** Accuracy numbers are not comparable to the paper. The 1.5B model has never been fine-tuned for Text-to-SQL and lacks the specialised schema-linking capability. Results are reported purely to demonstrate the pipeline end-to-end, with the substitution clearly disclosed everywhere.
- **This substitution is flagged in:** `configs/config.yaml`, `README.md` (Results & Analysis), and here.

---

## File-by-File Provenance

| File / Component | Provenance | Source Reference |
|---|---|---|
| `requirements.txt` | `written by us` | Exact versions from `pip freeze` in a clean Python 3.11 venv (Sept 2026). |
| `configs/config.yaml` | `written by us` | Original YAML config; model substitution notice added per this fix-up pass. |
| `data/download_spider.py` | `written by us` | Downloads real Spider data from two sources: (1) HuggingFace `xlangai/spider` (dataset v0.0.0, commit `0c35091`, 2023-07-10) for `dev.json`; (2) official Spider distribution from https://yale-lily.github.io/spider (Google Drive ID `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`, spider.zip, 206 MB) for SQLite databases and `tables.json`. Verification logic (gold SQL pre-check, db_id coverage) is original. |
| `data/sample_subset.py` | `written by us` | Seeded deterministic dev set sampler. No external code reused. |
| `data/schema_utils.py` | `adapted from MohammadrezaPourreza/DTS-SQL @ commit 94874ac, file: src/database_utils.py` | Schema-to-DDL string formatting adapted from the DTS-SQL repo's `database_utils.py`. The `SchemaParser` class and `format_schema_prompt()` method follow the same table/column formatting logic used in the official DTS-SQL prompts. Exact commit: `94874ac` (April 2024). |
| `models/schema_linking.py` | `adapted from MohammadrezaPourreza/DTS-SQL @ commit 94874ac, file: src/schema_linking.py` | Stage 1 Schema Linking inference logic. The prompt template (system instruction + DDL + question → JSON table/column output) is adapted from the DTS-SQL schema linking prompt structure described in the paper (§3.1) and visible in the official repo. The HF `AutoModelForCausalLM` loading and token decoding is standard HF usage, not from the repo. Heuristic fallback is original. |
| `models/sql_generation.py` | `adapted from MohammadrezaPourreza/DTS-SQL @ commit 94874ac, file: src/sql_generation.py` | Stage 2 SQL generation prompt (linked schema DDL + question → SQL). Prompt format follows the DTS-SQL paper (§3.2) and official repo structure. `_clean_sql()` is original. |
| `models/baseline_direct.py` | `written by us` | Single-stage direct NL→SQL baseline using full schema; no decomposition. Prompt design is original. |
| `pipeline/dts_sql_pipeline.py` | `written by us` | Sequential wrapper chaining Stage 1 → Stage 2. No external code. |
| `pipeline/run_inference.py` | `written by us` | CLI entrypoint for single-example inference. |
| `eval/execution_accuracy.py` | `adapted from taoyds/test-suite-sql-eval @ commit e97acc5, file: exec_eval.py` | Result-set equivalence evaluation. The approach (execute both queries, compare result sets as multisets) follows the Spider test-suite evaluation methodology from [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) commit `e97acc5`. Our implementation is a simplified Python port; we do not copy the original code verbatim. |
| `eval/timing.py` | `written by us` | Wall-clock latency tracking using `time.perf_counter()`. |
| `eval/run_eval.py` | `written by us` | End-to-end evaluation harness. LLM mode checking, gold SQL pre-check, fallback mode stamping, and results table blocking are all original additions from this fix-up pass. |
| `README.md` | `written by us` | Project documentation, setup guide, and results analysis. |

---

## External References & Paper Citations

- **Paper:** Pourreza, M., & Rafiei, D. (2024). *DTS-SQL: Decomposed Text-to-SQL with Small Large Language Models*. Findings of EMNLP / ACL 2024. arXiv:2402.01117.
- **Official GitHub Repository:** [MohammadrezaPourreza/DTS-SQL](https://github.com/MohammadrezaPourreza/DTS-SQL)
- **Dataset:** Spider (Yale University). Yu, T., et al. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL*. EMNLP 2018. License: CC BY-SA 4.0.
- **Spider evaluation:** [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval), commit `e97acc5` (execution accuracy methodology).
- **Spider questions/gold SQL:** HuggingFace `xlangai/spider`, dataset version 0.0.0, commit `0c35091` (2023-07-10).
- **Spider SQLite databases + tables.json:** Official Spider distribution download page at https://yale-lily.github.io/spider. Google Drive file ID `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J` (spider.zip, 206 MB). This is the download linked directly from the yale-lily.github.io/spider page as the official Spider 1.0 release.
