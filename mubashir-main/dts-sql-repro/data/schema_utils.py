"""
Schema parsing and linking utility functions for Spider database schemas.
Parses tables.json format and formats schemas for prompt feeding and filtering.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "spider"


class SchemaParser:
    def __init__(self, tables_json_path: Path):
        with open(tables_json_path, "r", encoding="utf-8") as f:
            raw_schemas = json.load(f)
        
        # Index schemas by db_id
        self.schemas: Dict[str, Dict[str, Any]] = {}
        for db in raw_schemas:
            db_id = db["db_id"]
            self.schemas[db_id] = self._parse_single_db(db)

    def _parse_single_db(self, db_raw: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw Spider table schema object into structured dict."""
        table_names = db_raw["table_names_original"]
        column_names_raw = db_raw["column_names_original"]
        column_types = db_raw["column_types"]
        primary_keys = set(db_raw.get("primary_keys", []))
        foreign_keys = db_raw.get("foreign_keys", [])

        tables: Dict[str, Dict[str, Any]] = {tbl.lower(): {"name": tbl, "columns": []} for tbl in table_names}

        fk_map = {}
        for col1_idx, col2_idx in foreign_keys:
            fk_map[col1_idx] = col2_idx
            fk_map[col2_idx] = col1_idx

        for idx, (tbl_idx, col_name) in enumerate(column_names_raw):
            if tbl_idx == -1:
                continue
            tbl_name = table_names[tbl_idx].lower()
            col_type = column_types[idx]
            is_pk = idx in primary_keys
            is_fk = idx in fk_map

            tables[tbl_name]["columns"].append({
                "name": col_name,
                "type": col_type,
                "is_pk": is_pk,
                "is_fk": is_fk,
                "index": idx
            })

        return {
            "db_id": db_raw["db_id"],
            "tables": tables
        }

    def get_full_schema(self, db_id: str) -> Dict[str, Any]:
        """Retrieve structured full schema for a database ID."""
        if db_id not in self.schemas:
            # Fallback to first available schema if specific db_id not present
            first_key = list(self.schemas.keys())[0]
            return self.schemas[first_key]
        return self.schemas[db_id]

    @staticmethod
    def format_schema_prompt(schema: Dict[str, Any], linked_columns: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Format a full or linked schema into DDL-like text for LLM prompt context.
        """
        lines = [f"### Database Schema: {schema['db_id']} ###"]
        
        for tbl_key, tbl_info in schema["tables"].items():
            tbl_name = tbl_info["name"]
            
            if linked_columns is not None:
                if tbl_name.lower() not in [t.lower() for t in linked_columns.keys()]:
                    continue
                allowed_cols = [c.lower() for c in linked_columns.get(tbl_name, linked_columns.get(tbl_name.lower(), []))]
            else:
                allowed_cols = None

            cols_desc = []
            for col in tbl_info["columns"]:
                if allowed_cols is not None and col["name"].lower() not in allowed_cols:
                    continue
                pk_str = " PRIMARY KEY" if col["is_pk"] else ""
                fk_str = " FOREIGN KEY" if col["is_fk"] else ""
                cols_desc.append(f"{col['name']} ({col['type']}{pk_str}{fk_str})")
            
            if cols_desc or allowed_cols is None:
                lines.append(f"Table `{tbl_name}`:")
                for cd in cols_desc:
                    lines.append(f"  - {cd}")
                    
        return "\n".join(lines)


def get_spider_tables_path(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Find tables.json path."""
    if (data_dir / "tables.json").exists():
        return data_dir / "tables.json"
    elif (data_dir.parent / "spider" / "tables.json").exists():
        return data_dir.parent / "spider" / "tables.json"
    elif (data_dir / "spider" / "tables.json").exists():
        return data_dir / "spider" / "tables.json"
    raise FileNotFoundError(f"tables.json not found in {data_dir}")


def main():
    """Unit test for SchemaParser."""
    tables_path = get_spider_tables_path()
    parser = SchemaParser(tables_path)
    db_ids = list(parser.schemas.keys())
    print(f"Loaded schemas for {len(db_ids)} databases.")
    
    first_db = db_ids[0]
    print(f"\n--- Parsed Schema for DB '{first_db}' ---")
    full_schema = parser.get_full_schema(first_db)
    formatted_full = parser.format_schema_prompt(full_schema)
    print(formatted_full)
    
    sample_table = list(full_schema["tables"].keys())[0]
    sample_cols = [c["name"] for c in full_schema["tables"][sample_table]["columns"][:2]]
    linked_sample = {sample_table: sample_cols}
    print(f"\n--- Test Filtered Linked Schema for '{first_db}' ---")
    formatted_linked = parser.format_schema_prompt(full_schema, linked_columns=linked_sample)
    print(formatted_linked)
    assert len(formatted_linked) < len(formatted_full), "Linked schema should be smaller than full schema."
    print("\n[OK] SchemaParser unit test passed.")


if __name__ == "__main__":
    main()
