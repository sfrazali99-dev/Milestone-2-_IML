"""
CLI Inference Runner for DTS-SQL and Direct Baseline.
Runs specified system over dataset split and produces output predictions.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.schema_utils import SchemaParser, get_spider_tables_path
from data.sample_subset import get_spider_dir, sample_dev_subset
from pipeline.dts_sql_pipeline import DTSSQLPipeline
from models.baseline_direct import DirectBaselineModel


def run_system_inference(system_type: str,
                         eval_dataset: List[Dict[str, Any]],
                         schema_parser: SchemaParser,
                         model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> List[Dict[str, Any]]:
    """Run inference over evaluation dataset for specified system."""
    results = []

    if system_type == "dts_sql":
        pipeline = DTSSQLPipeline(
            schema_linking_model_name=model_name,
            sql_generation_model_name=model_name
        )
    elif system_type == "baseline":
        baseline = DirectBaselineModel(model_name=model_name)
    else:
        raise ValueError(f"Unknown system type '{system_type}'. Choose 'dts_sql' or 'baseline'.")

    print(f"\nRunning inference for system '{system_type}' on {len(eval_dataset)} examples...")

    for idx, item in enumerate(eval_dataset):
        question = item["question"]
        db_id = item["db_id"]
        gold_sql = item["query"]

        full_schema = schema_parser.get_full_schema(db_id)

        if system_type == "dts_sql":
            res = pipeline.run_dts_sql(question, full_schema)
            predicted_sql = res["predicted_sql"]
            linked_schema = res["linked_schema"]
            latency = res["total_time"]
        else:
            t0 = time.perf_counter()
            predicted_sql = baseline.generate_sql_direct(question, full_schema)
            latency = time.perf_counter() - t0
            linked_schema = None

        record = {
            "example_id": idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "linked_schema": linked_schema,
            "latency_seconds": latency,
            "system": system_type
        }
        results.append(record)
        
        sys.stdout.write(f"\rProcessed example {idx + 1}/{len(eval_dataset)} ({latency:.3f}s)")
        sys.stdout.flush()

    print("\nInference complete.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference for DTS-SQL or Baseline direct system.")
    parser.add_argument("--system", type=str, choices=["dts_sql", "baseline"], required=True, help="System to evaluate")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of dev samples to evaluate")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="LLM checkpoint name")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON results filepath")
    args = parser.parse_args()

    spider_dir = get_spider_dir(Path("data/spider"))
    tables_path = get_spider_tables_path(spider_dir)
    schema_parser = SchemaParser(tables_path)

    eval_data = sample_dev_subset(spider_dir, num_samples=args.num_samples, seed=42)
    results = run_system_inference(args.system, eval_data, schema_parser, model_name=args.model_name)

    output_path = Path(args.output_file) if args.output_file else Path("logs") / f"predictions_{args.system}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved inference predictions to {output_path}")


if __name__ == "__main__":
    main()
