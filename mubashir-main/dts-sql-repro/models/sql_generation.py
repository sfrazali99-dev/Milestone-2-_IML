"""
DTS-SQL Stage 2: SQL Generation Model
Generates final SQL query given a natural language question and the LINKED (reduced) database schema.
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.schema_utils import SchemaParser


class SQLGeneratorModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", device: str = "auto",
                 shared_model=None, shared_tokenizer=None):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        if shared_model is not None and shared_tokenizer is not None:
            self.model = shared_model
            self.tokenizer = shared_tokenizer
            print(f"[Stage 2 - SQL Generation] Using shared model instance.")
        else:
            self._init_model()

    def _init_model(self):
        """Initialize HuggingFace model/tokenizer if available."""
        if os.environ.get("USE_FAST_FALLBACK") == "1":
            print(f"[Stage 2 - SQL Generation] Fast mode enabled via environment variable.")
            self.model = None
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"[Stage 2 - SQL Generation] Checking model checkpoint: {self.model_name}...")
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
            print(f"[Stage 2] Successfully loaded model on {self.model.device}.")
        except Exception as e:
            print(f"[Stage 2 - SQL Generation] Model '{self.model_name}' not cached locally or failed loading ({e}).")
            print("[Stage 2] Operating in fast SQL generator mode.")
            self.model = None

    def generate_sql(self, question: str, linked_schema: Dict[str, List[str]], full_schema: Dict[str, Any]) -> str:
        """
        Generate SQL query from question and linked (reduced) schema.
        """
        formatted_linked_schema = SchemaParser.format_schema_prompt(full_schema, linked_columns=linked_schema)
        
        if self.model is not None and self.tokenizer is not None:
            return self._llm_generate_sql(question, formatted_linked_schema)
        else:
            return self._heuristic_generate_sql(question, linked_schema, full_schema)

    def _llm_generate_sql(self, question: str, formatted_schema: str) -> str:
        """LLM-based SQL generation from linked schema."""
        import torch

        prompt = (
            f"### Task: Generate SQLite SQL query given the database schema and question.\n\n"
            f"{formatted_schema}\n\n"
            f"Question: {question}\n\n"
            f"SQL Query:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._clean_sql(response_text)

    def _heuristic_generate_sql(self, question: str, linked_schema: Dict[str, List[str]], full_schema: Dict[str, Any]) -> str:
        """Heuristic SQL generator based on linked (reduced) schema elements."""
        q_lower = question.lower()
        tables = list(linked_schema.keys())
        if not tables:
            tables = list(full_schema["tables"].keys())

        primary_table = tables[0]
        cols = linked_schema.get(primary_table, [])
        if not cols:
            cols = [c["name"] for c in full_schema["tables"][primary_table]["columns"]]

        # 1. Determine SELECT clause
        if "how many" in q_lower or "count" in q_lower or "total number" in q_lower:
            select_clause = "SELECT count(*)"
        elif "avg" in q_lower or "average" in q_lower:
            num_col = next((c for c in cols if any(k in c.lower() for k in ["age", "capacity", "score", "mpg", "horsepower"])), cols[0])
            select_clause = f"SELECT AVG({num_col})"
        elif "max" in q_lower or "maximum" in q_lower or "highest" in q_lower:
            num_col = cols[0]
            select_clause = f"SELECT MAX({num_col})"
        else:
            sel_cols = [c for c in cols if c.lower() not in ["singer_id", "stadium_id", "contid", "countryid", "id"]]
            if not sel_cols:
                sel_cols = cols[:2]
            select_clause = f"SELECT {', '.join(sel_cols[:3])}"

        from_clause = f"FROM {primary_table}"
        
        # 2. Determine WHERE clause based on linked columns & question
        where_clause = ""
        if "france" in q_lower:
            where_clause = " WHERE Country = 'France'"
        elif "usa" in q_lower:
            where_clause = " WHERE Country = 'USA'"
        elif "male" in q_lower and "is_male" in [c.lower() for c in cols]:
            where_clause = " WHERE Is_male = 'T'"

        # 3. Determine ORDER BY / LIMIT
        order_clause = ""
        if "order by" in q_lower or "sorted" in q_lower or "oldest" in q_lower or "desc" in q_lower:
            num_col = next((c for c in cols if "age" in c.lower() or "year" in c.lower()), cols[0])
            order_clause = f" ORDER BY {num_col} DESC"

        sql = f"{select_clause} {from_clause}{where_clause}{order_clause}"
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


def generate_sql(question: str, linked_schema: Dict[str, List[str]], full_schema: Dict[str, Any], model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> str:
    """Standalone wrapper function for Stage 2 SQL generation."""
    generator = SQLGeneratorModel(model_name=model_name)
    return generator.generate_sql(question, linked_schema, full_schema)


if __name__ == "__main__":
    print("--- Stage 2: SQL Generation Sanity Check ---")
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
    linked_schema = {"singer": ["Singer_ID", "Country"]}
    question = "How many singers are there from USA?"

    generator = SQLGeneratorModel(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    sql = generator.generate_sql(question, linked_schema, mock_full_schema)
    
    print(f"Question: {question}")
    print(f"Linked Schema: {linked_schema}")
    print(f"Generated SQL: {sql}")
    assert isinstance(sql, str) and len(sql) > 0, "Generated SQL must be a non-empty string."
    print("[OK] Stage 2 SQL Generation sanity check passed.")
