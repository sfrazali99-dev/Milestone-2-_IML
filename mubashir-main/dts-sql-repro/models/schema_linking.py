"""
DTS-SQL Stage 1: Schema Linking Model
Predicts the subset of relevant tables and columns from a natural language question and database schema.
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.schema_utils import SchemaParser


class SchemaLinkingModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", device: str = "auto",
                 shared_model=None, shared_tokenizer=None):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        if shared_model is not None and shared_tokenizer is not None:
            self.model = shared_model
            self.tokenizer = shared_tokenizer
            print(f"[Stage 1 - Schema Linking] Using shared model instance.")
        else:
            self._init_model()

    def _init_model(self):
        """Initialize HuggingFace model/tokenizer if available and requested, or set up fallback mode."""
        if os.environ.get("USE_FAST_FALLBACK") == "1":
            print(f"[Stage 1 - Schema Linking] Fast mode enabled via environment variable.")
            self.model = None
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"[Stage 1 - Schema Linking] Loading model checkpoint: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
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
                    trust_remote_code=True
                ).to("cpu")
            print(f"[Stage 1] Successfully loaded model on device: {getattr(self.model, 'device', 'cpu')}.")
        except Exception as e:
            print(f"[Stage 1 - Schema Linking] Model '{self.model_name}' not loaded ({e}).")
            print("[Stage 1] Operating in fast schema linking mode.")
            self.model = None

    def predict_relevant_schema(self, question: str, full_schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Given natural language question and full database schema,
        predict relevant tables and columns.
        """
        formatted_schema = SchemaParser.format_schema_prompt(full_schema)
        
        if self.model is not None and self.tokenizer is not None:
            return self._llm_predict_schema(question, full_schema, formatted_schema)
        else:
            return self._heuristic_predict_schema(question, full_schema)

    def _llm_predict_schema(self, question: str, full_schema: Dict[str, Any], formatted_schema: str) -> Dict[str, List[str]]:
        """LLM-based schema linking prediction."""
        import torch

        prompt = (
            f"You are a schema linking assistant for Text-to-SQL.\n"
            f"Given the database schema and question, identify only the tables and columns required to answer the question.\n\n"
            f"{formatted_schema}\n\n"
            f"Question: {question}\n\n"
            f"Output JSON format:\n"
            f"{{\n  \"table_name\": [\"column1\", \"column2\"]\n}}\n\n"
            f"JSON Output:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(getattr(self.model, "device", "cpu"))
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_json_schema_output(response_text, full_schema)

    def _heuristic_predict_schema(self, question: str, full_schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Deterministic schema linking matching question tokens to table & column names.
        """
        q_clean = re.sub(r'[^\w\s]', ' ', question.lower())
        q_tokens = set(q_clean.split())

        linked: Dict[str, List[str]] = {}

        for tbl_key, tbl_info in full_schema["tables"].items():
            tbl_name = tbl_info["name"]
            tbl_clean = tbl_name.lower().replace("_", " ")
            tbl_match = any(t in q_tokens for t in tbl_clean.split())
            
            matched_cols = []
            for col in tbl_info["columns"]:
                col_name = col["name"]
                col_clean = col_name.lower().replace("_", " ")
                if any(t in q_tokens for t in col_clean.split()) or col_name.lower() in q_clean:
                    matched_cols.append(col_name)

            if tbl_match or matched_cols:
                if not matched_cols:
                    matched_cols = [c["name"] for c in tbl_info["columns"] if c["is_pk"] or c["is_fk"]]
                if not matched_cols:
                    matched_cols = [tbl_info["columns"][0]["name"]]
                linked[tbl_name] = list(set(matched_cols))

        if not linked:
            for tbl_key, tbl_info in full_schema["tables"].items():
                tbl_name = tbl_info["name"]
                linked[tbl_name] = [c["name"] for c in tbl_info["columns"]]

        return linked

    def _parse_json_schema_output(self, response_text: str, full_schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Parse LLM output JSON safely into schema dict."""
        try:
            # Strip markdown block if present
            clean_text = response_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and len(parsed) > 0:
                    return parsed
        except Exception:
            pass
        return self._heuristic_predict_schema(response_text, full_schema)


def predict_relevant_schema(question: str, full_schema: Dict[str, Any], model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> Dict[str, List[str]]:
    """Standalone wrapper function for Stage 1 schema linking."""
    linker = SchemaLinkingModel(model_name=model_name)
    return linker.predict_relevant_schema(question, full_schema)


if __name__ == "__main__":
    print("--- Stage 1: Schema Linking Sanity Check ---")
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
    
    linker = SchemaLinkingModel(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    linked_schema = linker.predict_relevant_schema(question, mock_full_schema)
    
    print(f"Question: {question}")
    print(f"Linked Schema Output: {json.dumps(linked_schema, indent=2)}")
    assert isinstance(linked_schema, dict), "Linked schema output must be a dictionary."
    assert len(linked_schema) > 0, "Linked schema output must not be empty."
    print("[OK] Stage 1 Schema Linking sanity check passed.")
