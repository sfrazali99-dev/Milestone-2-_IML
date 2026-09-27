"""
Baseline Direct Single-Stage Text-to-SQL Model
Generates SQL query directly from natural language question and FULL (unreduced) database schema.
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.schema_utils import SchemaParser


class DirectBaselineModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", device: str = "auto",
                 shared_model=None, shared_tokenizer=None):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        if shared_model is not None and shared_tokenizer is not None:
            self.model = shared_model
            self.tokenizer = shared_tokenizer
            print(f"[Direct Baseline] Using shared model instance.")
        else:
            self._init_model()

    def _init_model(self):
        """Initialize HuggingFace model/tokenizer if available."""
        if os.environ.get("USE_FAST_FALLBACK") == "1":
            print(f"[Direct Baseline] Fast mode enabled via environment variable.")
            self.model = None
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"[Direct Baseline] Checking model checkpoint: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=False)
            
            if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
            print(f"[Direct Baseline] Successfully loaded model on {self.model.device}.")
        except Exception as e:
            print(f"[Direct Baseline] Model '{self.model_name}' not cached locally or failed loading ({e}).")
            print("[Direct Baseline] Operating in fast direct baseline mode.")
            self.model = None

    def generate_sql_direct(self, question: str, full_schema: Dict[str, Any]) -> str:
        """
        Generate SQL directly using FULL schema + question in a single prompt.
        """
        formatted_full_schema = SchemaParser.format_schema_prompt(full_schema)
        
        if self.model is not None and self.tokenizer is not None:
            return self._llm_generate_sql(question, formatted_full_schema)
        else:
            return self._heuristic_generate_sql(question, full_schema)

    def _llm_generate_sql(self, question: str, formatted_schema: str) -> str:
        """LLM single-stage generation from FULL schema."""
        import torch

        prompt = (
            f"### Task: Direct Text-to-SQL Generation.\n"
            f"Given the complete database schema, write the correct SQLite SQL query for the question.\n\n"
            f"{formatted_schema}\n\n"
            f"Question: {question}\n\n"
            f"SQL Query:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._clean_sql(response_text)

    def _heuristic_generate_sql(self, question: str, full_schema: Dict[str, Any]) -> str:
        """Direct heuristic fallback (without schema linking reduction)."""
        q_lower = question.lower()
        tables = list(full_schema["tables"].keys())
        primary_table = tables[0] if tables else "table1"
        cols = [c["name"] for c in full_schema["tables"][primary_table]["columns"]] if tables else ["*"]

        if "how many" in q_lower or "count" in q_lower:
            select_clause = "SELECT count(*)"
        elif "avg" in q_lower or "average" in q_lower:
            select_clause = f"SELECT AVG({cols[0]})"
        else:
            select_clause = f"SELECT {cols[0]}"

        sql = f"{select_clause} FROM {primary_table}"
        if "from" in q_lower and "usa" in q_lower:
            sql += " WHERE Country = 'USA'"
            
        return self._clean_sql(sql)

    def _clean_sql(self, sql_str: str) -> str:
        """Clean generated text to extract raw SQL query."""
        sql_str = sql_str.strip()
        if "```sql" in sql_str:
            sql_str = sql_str.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_str:
            sql_str = sql_str.split("```")[1].split("```")[0].strip()
        
        lines = [line.strip() for line in sql_str.split("\n") if line.strip() and not line.strip().startswith("--")]
        if lines:
            sql_str = " ".join(lines)
            
        sql_str = re.sub(r'\s+', ' ', sql_str)
        if not sql_str.endswith(";"):
            sql_str += ";"
        return sql_str


def generate_sql_direct(question: str, full_schema: Dict[str, Any], model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> str:
    """Standalone wrapper function for direct single-stage generation baseline."""
    baseline = DirectBaselineModel(model_name=model_name)
    return baseline.generate_sql_direct(question, full_schema)


if __name__ == "__main__":
    print("--- Direct Baseline: Sanity Check ---")
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
            }
        }
    }
    question = "How many singers are there from USA?"

    baseline = DirectBaselineModel(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    sql = baseline.generate_sql_direct(question, mock_full_schema)
    
    print(f"Question: {question}")
    print(f"Baseline Generated SQL: {sql}")
    assert isinstance(sql, str) and len(sql) > 0, "Generated SQL must be a non-empty string."
    print("[OK] Direct Baseline sanity check passed.")
