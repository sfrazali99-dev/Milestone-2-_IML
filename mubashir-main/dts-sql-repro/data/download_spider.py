"""
Spider Dataset Loader & Database Setup
Fetches real Spider data from:
  1. HuggingFace xlangai/spider  — for dev.json (questions + gold SQL)
  2. GitHub taoyds/spider clone  — for the 166 pre-built SQLite databases and tables.json

DATA SOURCES & PROVENANCE:
  - Questions/gold SQL: HuggingFace `xlangai/spider`, dataset v0.0.0 (commit 0c35091, 2023-07-10)
    https://huggingface.co/datasets/xlangai/spider
  - SQLite databases + tables.json: Official Spider GitHub repo (taoyds/spider)
    https://github.com/taoyds/spider
    Cloned via `git clone https://github.com/taoyds/spider.git data/spider_official`
    Database files are tracked via Git LFS in the official repo.

DATASET CITATION (required by the Spider authors):
  Yu, T. et al. (2018). Spider: A Large-Scale Human-Labeled Dataset for
  Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. EMNLP 2018.
  License: CC BY-SA 4.0.

IMPORTANT:
  The HuggingFace xlangai/spider dataset does NOT ship .sqlite files.
  This script uses the official GitHub clone (data/spider_official/) to get real databases.
  Run `git clone https://github.com/taoyds/spider.git data/spider_official` first if missing.

WRITTEN BY: us (this reproduction project)
"""

import sys
import json
import sqlite3
import random
import shutil
from pathlib import Path
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "spider"

# Official Spider distribution — Google Drive file ID from yale-lily.github.io/spider
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
SPIDER_ZIP_PATH = ROOT_DIR / "data" / "spider.zip"


def _download_hf_dev_set(spider_dir: Path) -> List[Dict[str, Any]]:
    """
    Download dev.json from HuggingFace xlangai/spider.
    Source: https://huggingface.co/datasets/xlangai/spider (v0.0.0, commit 0c35091)
    """
    from datasets import load_dataset
    print("Loading Spider dev set from HuggingFace (xlangai/spider)...")
    ds = load_dataset("xlangai/spider")
    val_set = ds["validation"]

    dev_data = []
    for item in val_set:
        dev_data.append({
            "db_id": item["db_id"],
            "question": item["question"],
            "query": item["query"]
        })

    dev_json_path = spider_dir / "dev.json"
    with open(dev_json_path, "w", encoding="utf-8") as f:
        json.dump(dev_data, f, indent=2)
    print(f"[OK] Saved {len(dev_data)} Spider dev examples to {dev_json_path}")
    return dev_data


def _download_and_extract_spider_zip(spider_dir: Path):
    """
    Download and extract the official Spider distribution zip (includes all SQLite databases
    and tables.json) from the official download page at yale-lily.github.io/spider.

    File: spider.zip (206 MB)
    Source: https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing
    Found at: https://yale-lily.github.io/spider
    """
    import zipfile

    db_dest = spider_dir / "database"
    if db_dest.exists():
        existing = [p for p in db_dest.iterdir() if p.is_dir()]
        if len(existing) > 10:
            print(f"[OK] database/ dir already has {len(existing)} entries, skipping download.")
            return

    zip_path = SPIDER_ZIP_PATH

    if not zip_path.exists() or zip_path.stat().st_size < 10_000_000:
        try:
            import gdown
        except ImportError:
            print("[ERROR] gdown not installed. Run: pip install gdown")
            sys.exit(1)

        print(f"\nDownloading official Spider distribution (206 MB) via gdown...")
        print(f"  Source: https://drive.google.com/file/d/{SPIDER_GDRIVE_ID}/view?usp=sharing")
        print(f"  (Found at: https://yale-lily.github.io/spider)")
        try:
            gdown.download(id=SPIDER_GDRIVE_ID, output=str(zip_path), quiet=False)
        except Exception as e:
            print(f"\n[ERROR] gdown download failed: {e}")
            print("\nManual download instructions:")
            print("  1. Go to: https://yale-lily.github.io/spider")
            print("  2. Click the 'Spider 1.0' download button to get spider.zip")
            print(f"  3. Save it as: {zip_path}")
            print("  4. Re-run this script.")
            sys.exit(1)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(spider_dir.parent))

    # The zip extracts to a 'spider/' subdirectory; move database/ and tables.json into place
    extracted_spider = spider_dir.parent / "spider"
    if extracted_spider.exists() and extracted_spider != spider_dir:
        # Move database/ folder
        src_db = extracted_spider / "database"
        if src_db.exists() and not db_dest.exists():
            shutil.move(str(src_db), str(db_dest))
        # Move tables.json
        src_tables = extracted_spider / "tables.json"
        dest_tables = spider_dir / "tables.json"
        if src_tables.exists() and not dest_tables.exists():
            shutil.copy(str(src_tables), str(dest_tables))
        # Move train/dev json if present
        for fname in ["train_spider.json", "train_others.json", "dev.json"]:
            src_f = extracted_spider / fname
            dest_f = spider_dir / fname
            if src_f.exists() and not dest_f.exists():
                shutil.copy(str(src_f), str(dest_f))

    sqlite_count = len(list(db_dest.glob("*/*.sqlite"))) if db_dest.exists() else 0
    print(f"[OK] Extracted {sqlite_count} SQLite databases to {db_dest}")


def _verify_tables_json(spider_dir: Path):
    """Verify tables.json was extracted from the official Spider distribution."""
    dest = spider_dir / "tables.json"
    if not dest.exists():
        print(f"[ERROR] tables.json missing at {dest}")
        print("tables.json should have been extracted from spider.zip.")
        sys.exit(1)
    with open(dest, encoding="utf-8") as f:
        tables = json.load(f)
    print(f"[OK] tables.json present with {len(tables)} real database schemas.")


def _verify_db_coverage(dev_data: List[Dict[str, Any]], spider_dir: Path):
    """
    Verify every db_id referenced in dev.json has a matching .sqlite file.
    Exits loudly if any are missing — does NOT silently fall back to a different DB.
    """
    db_dir = spider_dir / "database"
    dev_db_ids = set(item["db_id"] for item in dev_data)
    missing = [
        db_id for db_id in dev_db_ids
        if not (db_dir / db_id / f"{db_id}.sqlite").exists()
    ]
    if missing:
        print(f"\n[FATAL] {len(missing)} db_ids in dev.json have no matching .sqlite file:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print("\nThis is a data pipeline error. Run data/download_spider.py to fix it.")
        sys.exit(1)

    print(f"[OK] All {len(dev_db_ids)} dev db_ids have matching .sqlite files.")


def _verify_gold_sql_executes(dev_data: List[Dict[str, Any]], spider_dir: Path, sample_size: int = 10):
    """
    Randomly sample examples and confirm their gold SQL executes against the real database.
    Exits loudly if any fail — this is a data pipeline bug, not a model error.
    """
    db_dir = spider_dir / "database"
    rng = random.Random(42)
    sample = rng.sample(dev_data, min(sample_size, len(dev_data)))

    print(f"\n[Pre-check] Verifying gold SQL executes on real databases ({len(sample)} samples)...")
    failures = []
    for item in sample:
        db_id = item["db_id"]
        gold_sql = item["query"]
        sqlite_file = db_dir / db_id / f"{db_id}.sqlite"
        try:
            conn = sqlite3.connect(str(sqlite_file))
            conn.execute(gold_sql)
            conn.close()
        except Exception as e:
            failures.append({"db_id": db_id, "sql": gold_sql, "error": str(e)})

    if failures:
        print(f"\n[ERROR] {len(failures)} gold SQL queries failed on real databases:")
        for f in failures:
            print(f"  db_id={f['db_id']}: {f['error']}")
            print(f"    SQL: {f['sql'][:100]}")
        print("\nThis is a data pipeline bug — fix before evaluating.")
        sys.exit(1)

    print(f"[OK] All {len(sample)} sampled gold SQL queries execute successfully.")


def main(data_dir: Path = DEFAULT_DATA_DIR):
    """
    Full pipeline:
      1. Download dev.json from HuggingFace xlangai/spider
      2. Download and extract official spider.zip (databases + tables.json)
         from yale-lily.github.io/spider via gdown
      3. Verify all dev db_ids have a matching .sqlite file (fail loudly if not)
      4. Verify gold SQL executes on real databases
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Real dev.json from HuggingFace (questions + gold SQL)
    dev_data = _download_hf_dev_set(data_dir)

    # Step 2: Real SQLite databases + tables.json from official Spider zip
    _download_and_extract_spider_zip(data_dir)

    # Step 3: Verify tables.json present
    _verify_tables_json(data_dir)

    # Step 4: Verify database coverage
    _verify_db_coverage(dev_data, data_dir)

    # Step 5: Gold SQL sanity check
    _verify_gold_sql_executes(dev_data, data_dir)

    db_count = sum(1 for _ in (data_dir / "database").glob("*/*.sqlite"))
    print(f"\n[OK] Spider dataset fully prepared and verified.")
    print(f"  dev.json:    {len(dev_data)} real examples (from HuggingFace xlangai/spider)")
    print(f"  databases:   {db_count} real SQLite databases (from yale-lily.github.io/spider)")
    print(f"  tables.json: real schema definitions (from yale-lily.github.io/spider)")


if __name__ == "__main__":
    main()
