"""
SQLite Execution Accuracy Evaluator for Text-to-SQL.
Executes predicted SQL and gold SQL against corresponding Spider SQLite database file
and evaluates result set equivalence.
"""

import sys
import sqlite3
from pathlib import Path
from typing import Tuple, Any, List, Set, Optional, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def execute_sql_query(db_path: Path, sql: str) -> Tuple[bool, Any, Optional[str]]:
    """
    Execute SQL query against SQLite database file.
    """
    if not db_path.exists():
        return False, None, f"Database file does not exist: {db_path}"

    clean_sql = sql.strip().rstrip(";")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(clean_sql)
        results = cursor.fetchall()
        conn.close()
        return True, results, None
    except Exception as e:
        return False, None, str(e)


def compare_result_sets(pred_res: List[Tuple], gold_res: List[Tuple]) -> bool:
    """
    Compare predicted result set with gold result set.
    """
    if pred_res is None or gold_res is None:
        return False

    try:
        pred_set = set(tuple(str(x) for x in row) for row in pred_res)
        gold_set = set(tuple(str(x) for x in row) for row in gold_res)
        return pred_set == gold_set
    except Exception:
        return False


def evaluate_execution_accuracy(db_path: Path, predicted_sql: str, gold_sql: str) -> Dict[str, Any]:
    """
    Evaluate execution accuracy for a single query pair against SQLite DB.
    """
    gold_ok, gold_results, gold_err = execute_sql_query(db_path, gold_sql)
    pred_ok, pred_results, pred_err = execute_sql_query(db_path, predicted_sql)

    if not gold_ok:
        return {
            "correct": False,
            "pred_success": pred_ok,
            "gold_success": False,
            "error": f"Gold query error: {gold_err}"
        }

    if not pred_ok:
        return {
            "correct": False,
            "pred_success": False,
            "gold_success": True,
            "error": f"Predicted query error: {pred_err}"
        }

    is_correct = compare_result_sets(pred_results, gold_results)
    return {
        "correct": is_correct,
        "pred_success": True,
        "gold_success": True,
        "error": None if is_correct else "Result set mismatch"
    }


if __name__ == "__main__":
    print("--- Execution Accuracy Unit Test ---")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        tmp_db = Path(tmp.name)

    conn = sqlite3.connect(str(tmp_db))
    conn.execute("CREATE TABLE singer (id INT, name TEXT, country TEXT)")
    conn.execute("INSERT INTO singer VALUES (1, 'Alice', 'USA'), (2, 'Bob', 'UK')")
    conn.commit()
    conn.close()

    gold_sql = "SELECT name FROM singer WHERE country = 'USA'"
    pred_sql_correct = "SELECT name FROM singer WHERE country = 'USA';"
    pred_sql_incorrect = "SELECT name FROM singer WHERE country = 'UK';"
    pred_sql_syntax_error = "SELECT name FROM non_existing_table;"

    r1 = evaluate_execution_accuracy(tmp_db, pred_sql_correct, gold_sql)
    r2 = evaluate_execution_accuracy(tmp_db, pred_sql_incorrect, gold_sql)
    r3 = evaluate_execution_accuracy(tmp_db, pred_sql_syntax_error, gold_sql)

    print(f"Correct SQL match test: {r1}")
    print(f"Incorrect result test: {r2}")
    print(f"Syntax error test: {r3}")

    assert r1["correct"] is True
    assert r2["correct"] is False
    assert r3["correct"] is False and r3["pred_success"] is False

    tmp_db.unlink()
    print("[OK] execution_accuracy unit test passed.")
