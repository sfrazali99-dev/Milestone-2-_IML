"""
DTS-SQL Two-Stage Pipeline
Chains Stage 1 (Schema Linking) and Stage 2 (SQL Generation).
Logs intermediate linked schema alongside final SQL query.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.schema_linking import SchemaLinkingModel
from models.sql_generation import SQLGeneratorModel


class DTSSQLPipeline:
    def __init__(self, 
                 schema_linking_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                 sql_generation_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                 device: str = "auto",
                 shared_model=None, shared_tokenizer=None):
        print("[DTS-SQL Pipeline] Initializing Stage 1 and Stage 2 models...")
        self.stage1_model = SchemaLinkingModel(
            model_name=schema_linking_model_name, device=device,
            shared_model=shared_model, shared_tokenizer=shared_tokenizer
        )
        self.stage2_model = SQLGeneratorModel(
            model_name=sql_generation_model_name, device=device,
            shared_model=shared_model, shared_tokenizer=shared_tokenizer
        )
        print("[DTS-SQL Pipeline] Pipeline ready.")

    def run_dts_sql(self, question: str, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full two-stage DTS-SQL inference.
        """
        t0 = time.perf_counter()
        
        # Stage 1: Schema Linking
        linked_schema = self.stage1_model.predict_relevant_schema(question, full_schema)
        t1 = time.perf_counter()
        schema_linking_latency = t1 - t0

        # Stage 2: SQL Generation using LINKED schema
        predicted_sql = self.stage2_model.generate_sql(question, linked_schema, full_schema)
        t2 = time.perf_counter()
        sql_gen_latency = t2 - t1
        total_latency = t2 - t0

        return {
            "predicted_sql": predicted_sql,
            "linked_schema": linked_schema,
            "schema_linking_time": schema_linking_latency,
            "sql_generation_time": sql_gen_latency,
            "total_time": total_latency
        }


def run_dts_sql(question: str, full_schema: Dict[str, Any], model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> Dict[str, Any]:
    """Convenience functional wrapper for DTS-SQL pipeline execution."""
    pipeline = DTSSQLPipeline(
        schema_linking_model_name=model_name,
        sql_generation_model_name=model_name
    )
    return pipeline.run_dts_sql(question, full_schema)


if __name__ == "__main__":
    print("--- DTS-SQL Two-Stage Pipeline Sanity Check ---")
    mock_full_schema = {
        "db_id": "concert_singer",
        "tables": {
            "singer": {
                "name": "singer",
                "columns": [
                    {"name": "Singer_ID", "type": "number", "is_pk": True, "is_fk": False},
                    {"name": "Name", "type": "text", "is_pk": False, "is_fk": False},
                    {"name": "Country", "type": "text", "is_pk": False, "is_fk": False},
                    {"name": "Age", "type": "number", "is_pk": False, "is_fk": False}
                ]
            },
            "stadium": {
                "name": "stadium",
                "columns": [
                    {"name": "Stadium_ID", "type": "number", "is_pk": True, "is_fk": False},
                    {"name": "Location", "type": "text", "is_pk": False, "is_fk": False},
                    {"name": "Capacity", "type": "number", "is_pk": False, "is_fk": False}
                ]
            }
        }
    }
    question = "How many singers are there from USA?"

    pipeline = DTSSQLPipeline(schema_linking_model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    result = pipeline.run_dts_sql(question, mock_full_schema)

    print(f"Question: {question}")
    print(f"Intermediate Linked Schema: {result['linked_schema']}")
    print(f"Final Predicted SQL: {result['predicted_sql']}")
    print(f"Latency: Stage 1 = {result['schema_linking_time']:.4f}s, Stage 2 = {result['sql_generation_time']:.4f}s, Total = {result['total_time']:.4f}s")
    
    assert isinstance(result["predicted_sql"], str) and len(result["predicted_sql"]) > 0
    assert isinstance(result["linked_schema"], dict) and len(result["linked_schema"]) > 0
    print("[OK] Two-stage DTS-SQL pipeline sanity check passed.")
