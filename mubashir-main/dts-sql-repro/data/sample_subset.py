"""
Deterministic subset sampler for Spider dataset.
Allows fast iteration and evaluation scaling.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "spider"


def get_spider_dir(data_dir: Path) -> Path:
    """Return actual Spider directory containing json files."""
    if (data_dir / "dev.json").exists():
        return data_dir
    elif (data_dir.parent / "spider" / "dev.json").exists():
        return data_dir.parent / "spider"
    elif (data_dir / "spider" / "dev.json").exists():
        return data_dir / "spider"
    raise FileNotFoundError(f"Could not locate dev.json in {data_dir}")


def sample_dev_subset(data_dir: Path = DEFAULT_DATA_DIR, num_samples: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """Deterministically sample N examples from Spider dev set."""
    spider_path = get_spider_dir(data_dir)
    dev_path = spider_path / "dev.json"

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    # Filter to dev examples whose db_id is present in tables.json
    tables_path = spider_path / "tables.json"
    if tables_path.exists():
        with open(tables_path, "r", encoding="utf-8") as tf:
            raw_tables = json.load(tf)
        available_db_ids = set(tbl["db_id"] for tbl in raw_tables)
        filtered_data = [item for item in dev_data if item["db_id"] in available_db_ids]
        if filtered_data:
            dev_data = filtered_data

    print(f"Total dev set examples available: {len(dev_data)}")

    random.seed(seed)
    if num_samples >= len(dev_data) or num_samples <= 0:
        sampled_data = dev_data
        print(f"Using full dev set ({len(sampled_data)} examples).")
    else:
        sampled_data = random.sample(dev_data, num_samples)
        print(f"Sampled {len(sampled_data)} examples using seed={seed}.")

    db_ids = set(item["db_id"] for item in sampled_data)
    print(f"Subset spans {len(db_ids)} unique databases.")
    return sampled_data


def save_subset(sampled_data: List[Dict[str, Any]], output_path: Path):
    """Save sampled subset to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=2)
    print(f"Saved subset ({len(sampled_data)} items) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample a subset of Spider dev set.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to draw (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    spider_dir = get_spider_dir(DEFAULT_DATA_DIR)
    output_path = Path(args.output) if args.output else spider_dir / f"dev_subset_{args.num_samples}.json"
    
    subset = sample_dev_subset(spider_dir, num_samples=args.num_samples, seed=args.seed)
    save_subset(subset, output_path)


if __name__ == "__main__":
    main()
