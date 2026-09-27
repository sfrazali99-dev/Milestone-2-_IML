"""
Exploratory Demo Notebook / Script for DTS-SQL Reproduction
Demonstrates step-by-step schema parsing, schema linking, and SQL generation.
"""

import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.download_spider import main as download_spider
from data.sample_subset import get_spider_dir, sample_dev_subset
from data.schema_utils import SchemaParser, get_spider_tables_path
from pipeline.dts_sql_pipeline import DTSSQLPipeline
from eval.execution_accuracy import evaluate_execution_accuracy


def run_demo():
    print("=== DTS-SQL Interactive Demo ===")
    
    # 1. Download Spider data
    download_spider()
    spider_dir = get_spider_dir(Path("data/spider"))
    tables_path = get_spider_tables_path(spider_dir)
    schema_parser = SchemaParser(tables_path)

    # 2. Pick sample query from dev set
    dev_samples = sample_dev_subset(spider_dir, num_samples=1, seed=123)
    sample = dev_samples[0]

    question = sample["question"]
    db_id = sample["db_id"]
    gold_sql = sample["query"]

    print(f"\n[Question]: {question}")
    print(f"[Database ID]: {db_id}")
    print(f"[Gold SQL]: {gold_sql}")

    # 3. Parse Full Schema
    full_schema = schema_parser.get_full_schema(db_id)
    print(f"\n[Full Schema Tables]: {list(full_schema['tables'].keys())}")

    # 4. Run DTS-SQL Pipeline
    pipeline = DTSSQLPipeline()
    result = pipeline.run_dts_sql(question, full_schema)

    print(f"\n[Stage 1 Output - Linked Schema]:")
    for tbl, cols in result["linked_schema"].items():
        print(f"  Table `{tbl}`: {cols}")

    print(f"\n[Stage 2 Output - Predicted SQL]: {result['predicted_sql']}")
    print(f"[Latencies]: Schema Linking = {result['schema_linking_time']:.4f}s, SQL Gen = {result['sql_generation_time']:.4f}s, Total = {result['total_time']:.4f}s")

    # 5. Evaluate against SQLite
    db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
    eval_res = evaluate_execution_accuracy(db_path, result["predicted_sql"], gold_sql)

    print(f"\n[SQLite Execution Evaluation]: Correct = {eval_res['correct']}, Status = {eval_res['error'] or 'Success'}")


if __name__ == "__main__":
    run_demo()
