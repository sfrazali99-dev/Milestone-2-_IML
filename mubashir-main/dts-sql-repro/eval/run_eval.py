"""
Full Evaluation Harness for DTS-SQL Reproduction.
Runs Baseline (Direct) vs DTS-SQL (Two-Stage) on fixed evaluation subset.
Computes execution accuracy, wall-clock timing, prints formatted results table,
and saves logs to logs/ directory.

NOTE ON INFERENCE MODE:
  This harness requires real LLM inference (transformers + torch installed and importable).
  If the models fail to load, the per-example mode is stamped "heuristic_fallback" and
  the final results table is BLOCKED from being written until you acknowledge the mode.
  A heuristic-fallback run is NOT a valid reproduction result.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
from tabulate import tabulate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.download_spider import main as download_spider_if_needed
from data.sample_subset import get_spider_dir, sample_dev_subset
from data.schema_utils import SchemaParser, get_spider_tables_path
from models.baseline_direct import DirectBaselineModel
from pipeline.dts_sql_pipeline import DTSSQLPipeline
from eval.execution_accuracy import evaluate_execution_accuracy
from eval.timing import LatencyTracker


def _verify_llm_available():
    """
    Confirm transformers + torch are importable.  Called before loading any model.
    Prints a hard warning and returns False if either is missing — the caller must
    decide whether to abort or continue in heuristic mode.
    """
    try:
        import torch
        import transformers
        print(f"[OK] torch {torch.__version__} and transformers {transformers.__version__} are available.")
        return True
    except ImportError as e:
        print("\n" + "!" * 70)
        print("!! HARD WARNING: Real LLM inference is NOT available.")
        print(f"!! Import error: {e}")
        print("!! All model calls will fall back to keyword/regex heuristics.")
        print("!! Results produced in this mode are NOT a valid reproduction.")
        print("!! Activate the venv (venv\\Scripts\\activate) and retry.")
        print("!" * 70 + "\n")
        return False


def get_db_path(spider_dir: Path, db_id: str) -> Path:
    """
    Find SQLite database file path for given db_id.
    Raises FileNotFoundError loudly if the database is missing — does NOT silently
    fall back to a different DB, which would contaminate evaluation results.
    """
    db_file = spider_dir / "database" / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return db_file

    # One alt path to check (direct child of spider dir)
    alt_file = spider_dir.parent / "database" / db_id / f"{db_id}.sqlite"
    if alt_file.exists():
        return alt_file

    raise FileNotFoundError(
        f"[ERROR] SQLite database not found for db_id='{db_id}'. "
        f"Expected at: {db_file}\n"
        f"Run data/download_spider.py to fetch the real Spider databases."
    )


def evaluate_system_on_subset(system_name: str,
                              system_runner,
                              eval_dataset: List[Dict[str, Any]],
                              schema_parser: SchemaParser,
                              spider_dir: Path,
                              llm_available: bool) -> Dict[str, Any]:
    """Run system over evaluation subset and calculate execution accuracy & timing."""
    print(f"\n=======================================================")
    print(f" Evaluating System: {system_name}")
    print(f" Total Examples: {len(eval_dataset)}")
    print(f" Inference mode: {'llm' if llm_available else 'heuristic_fallback'}")
    print(f"=======================================================")

    if not llm_available:
        print("\n" + "!" * 60)
        print("!! WARNING: running in HEURISTIC FALLBACK mode.")
        print("!! These results are NOT from a real model.")
        print("!" * 60 + "\n")

    correct_count = 0
    syntax_error_count = 0
    tracker = LatencyTracker()
    per_query_records = []
    inference_mode = "llm" if llm_available else "heuristic_fallback"

    for idx, item in enumerate(eval_dataset):
        question = item["question"]
        db_id = item["db_id"]
        gold_sql = item["query"]

        full_schema = schema_parser.get_full_schema(db_id)

        try:
            db_path = get_db_path(spider_dir, db_id)
        except FileNotFoundError as exc:
            print(f"\n[FATAL] {exc}")
            sys.exit(1)

        t0 = time.perf_counter()
        if system_name == "DTS-SQL (two-stage)":
            res = system_runner.run_dts_sql(question, full_schema)
            pred_sql = res["predicted_sql"]
            linked_schema = res["linked_schema"]
        else:
            pred_sql = system_runner.generate_sql_direct(question, full_schema)
            linked_schema = None
        latency = time.perf_counter() - t0
        tracker.record(latency)

        eval_res = evaluate_execution_accuracy(db_path, pred_sql, gold_sql)

        if eval_res["correct"]:
            correct_count += 1
        if not eval_res["pred_success"]:
            syntax_error_count += 1

        record = {
            "index": idx,
            "system": system_name,
            "mode": inference_mode,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "predicted_sql": pred_sql,
            "linked_schema": linked_schema,
            "correct": eval_res["correct"],
            "pred_success": eval_res["pred_success"],
            "error": eval_res["error"],
            "latency_seconds": round(latency, 4)
        }
        per_query_records.append(record)

        status_str = "[CORRECT]" if eval_res["correct"] else "[INCORRECT]"
        sys.stdout.write(f"\r[{idx+1}/{len(eval_dataset)}] {status_str} (latency: {latency:.3f}s)")
        sys.stdout.flush()

    print("\nEvaluation complete.")

    acc_pct = (correct_count / len(eval_dataset)) * 100.0 if eval_dataset else 0.0
    timing_stats = tracker.summary()

    return {
        "system_name": system_name,
        "mode": inference_mode,
        "total_examples": len(eval_dataset),
        "correct": correct_count,
        "syntax_errors": syntax_error_count,
        "execution_accuracy_pct": round(acc_pct, 2),
        "avg_time_per_query": timing_stats["mean"],
        "timing_stats": timing_stats,
        "per_query_records": per_query_records
    }


def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation benchmark for DTS-SQL vs Direct Baseline.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of dev set samples to evaluate (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset sampling")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="Model checkpoint. NOTE: This is a substitute model, not the paper's official checkpoint.")
    parser.add_argument("--allow_fallback", action="store_true",
                        help="Allow heuristic-fallback mode. Results will be clearly marked as invalid for reproduction.")
    args = parser.parse_args()

    # Step 1: Verify LLM imports before doing anything else
    llm_available = _verify_llm_available()
    if not llm_available and not args.allow_fallback:
        print("[ABORT] Real LLM inference is required. Use --allow_fallback to run anyway (results will be marked invalid).")
        sys.exit(1)

    download_spider_if_needed()
    spider_dir = get_spider_dir(Path("data/spider"))
    tables_path = get_spider_tables_path(spider_dir)
    schema_parser = SchemaParser(tables_path)

    # Step 2: Verify gold SQL executes on real databases for a sample before running eval
    print("\n[Pre-check] Verifying gold SQL executes on real databases for 3 random examples...")
    eval_dataset_full = sample_dev_subset(spider_dir, num_samples=args.num_samples, seed=args.seed)
    import random
    rng = random.Random(args.seed)
    check_sample = rng.sample(eval_dataset_full, min(3, len(eval_dataset_full)))
    for item in check_sample:
        try:
            db_path = get_db_path(spider_dir, item["db_id"])
        except FileNotFoundError as exc:
            print(f"\n[FATAL] {exc}")
            sys.exit(1)
        gold_res = evaluate_execution_accuracy(db_path, item["query"], item["query"])
        if not gold_res["pred_success"]:
            print(f"\n[FATAL] Gold SQL failed to execute on real database '{item['db_id']}':")
            print(f"  SQL: {item['query']}")
            print(f"  Error: {gold_res['error']}")
            print("  This is a data pipeline bug. Fix before running eval.")
            sys.exit(1)
    print("[Pre-check OK] Gold SQL executes correctly on real databases.\n")

    print(f"\n[Eval Harness] Model substitute in use: {args.model_name}")
    print("[Eval Harness] NOTE: This is NOT the paper's official DTS-SQL checkpoint.")

    # Load model ONCE and share across all components to avoid OOM on CPU
    # (each fp32 copy of Qwen2.5-Coder-1.5B is ~3GB; loading 3 copies would require ~9GB)
    shared_model = None
    shared_tokenizer = None
    if llm_available:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[Eval Harness] Loading shared model: {args.model_name}...")
        shared_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        shared_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to("cpu")
        print(f"[Eval Harness] Model loaded ({sum(p.numel() for p in shared_model.parameters())/1e6:.1f}M params, fp32, CPU).")

    print("[Eval Harness] Initializing Direct Baseline model...")
    baseline_runner = DirectBaselineModel(
        model_name=args.model_name,
        shared_model=shared_model, shared_tokenizer=shared_tokenizer
    )

    print("\n[Eval Harness] Initializing DTS-SQL Two-Stage Pipeline...")
    dts_sql_runner = DTSSQLPipeline(
        schema_linking_model_name=args.model_name,
        sql_generation_model_name=args.model_name,
        shared_model=shared_model, shared_tokenizer=shared_tokenizer
    )

    # Determine actual mode by checking if model loaded
    actual_llm_available = llm_available and (baseline_runner.model is not None)

    baseline_results = evaluate_system_on_subset(
        "Baseline (direct)", baseline_runner, eval_dataset_full, schema_parser, spider_dir, actual_llm_available
    )

    dts_sql_results = evaluate_system_on_subset(
        "DTS-SQL (two-stage)", dts_sql_runner, eval_dataset_full, schema_parser, spider_dir, actual_llm_available
    )

    # Determine overall run mode
    run_mode = "llm" if actual_llm_available else "heuristic_fallback"

    table_data = [
        [
            baseline_results["system_name"],
            f"{baseline_results['execution_accuracy_pct']:.2f}% ({baseline_results['correct']}/{baseline_results['total_examples']})",
            f"{baseline_results['avg_time_per_query']:.4f}s",
            baseline_results["total_examples"],
            run_mode
        ],
        [
            dts_sql_results["system_name"],
            f"{dts_sql_results['execution_accuracy_pct']:.2f}% ({dts_sql_results['correct']}/{dts_sql_results['total_examples']})",
            f"{dts_sql_results['avg_time_per_query']:.4f}s",
            dts_sql_results["total_examples"],
            run_mode
        ]
    ]

    headers = ["System", "Execution Accuracy", "Avg. time/query", "# examples", "Mode"]
    results_table = tabulate(table_data, headers=headers, tablefmt="github")

    print("\n" + "=" * 70)
    print("        FINAL MILESTONE 2 REPRODUCTION RESULTS TABLE")
    if run_mode == "heuristic_fallback":
        print("        *** WARNING: HEURISTIC FALLBACK MODE — NOT A VALID REPRODUCTION ***")
    print("=" * 70)
    print(results_table)
    print("=" * 70)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_file = logs_dir / "metrics_summary.json"
    details_file = logs_dir / "eval_details.jsonl"
    table_file = logs_dir / "results_table.txt"

    summary_data = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "model_name": args.model_name,
        "model_is_paper_checkpoint": False,
        "model_substitution_note": (
            "Substitute: Qwen/Qwen2.5-Coder-1.5B-Instruct. "
            "Paper uses pourreza/DTS-SQL-SL-7B and pourreza/DTS-SQL-7B (fine-tuned 7B models). "
            "Substitute used due to hardware constraints (no GPU, insufficient RAM for 7B fp32)."
        ),
        "run_mode": run_mode,
        "baseline": {
            "accuracy_pct": baseline_results["execution_accuracy_pct"],
            "avg_latency": baseline_results["avg_time_per_query"],
            "correct": baseline_results["correct"],
            "total": baseline_results["total_examples"],
            "mode": run_mode
        },
        "dts_sql": {
            "accuracy_pct": dts_sql_results["execution_accuracy_pct"],
            "avg_latency": dts_sql_results["avg_time_per_query"],
            "correct": dts_sql_results["correct"],
            "total": dts_sql_results["total_examples"],
            "mode": run_mode
        },
        "accuracy_gain_pct": round(dts_sql_results["execution_accuracy_pct"] - baseline_results["execution_accuracy_pct"], 2)
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Only write results_table.txt if in LLM mode; otherwise write a clearly marked stub
    if run_mode == "heuristic_fallback":
        stub = (
            "*** HEURISTIC FALLBACK RUN — NOT A VALID REPRODUCTION RESULT ***\n"
            "These numbers were produced by keyword/regex heuristics, not a real LLM.\n"
            "Re-run with transformers + torch installed (activate venv first).\n\n"
        ) + results_table
        with open(table_file, "w", encoding="utf-8") as f:
            f.write(stub)
    else:
        with open(table_file, "w", encoding="utf-8") as f:
            f.write(f"Model substitute: {args.model_name} (NOT the paper's official checkpoint)\n")
            f.write(results_table)

    with open(details_file, "w", encoding="utf-8") as f:
        for rec in baseline_results["per_query_records"] + dts_sql_results["per_query_records"]:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSaved evaluation summary to {summary_file}")
    print(f"Saved formatted table to {table_file}")
    print(f"Saved per-query details log to {details_file}")

    if run_mode == "heuristic_fallback":
        print("\n*** REMINDER: This was a HEURISTIC FALLBACK run. Results are NOT valid for reproduction. ***")


if __name__ == "__main__":
    main()
