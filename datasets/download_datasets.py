#!/usr/bin/env python3
"""
Download datasets from HuggingFace and convert to multiple formats.

Usage:
    python download_datasets.py                    # Download all
    python download_datasets.py --dataset tickets  # Download specific
    python download_datasets.py --size small       # Download small version (1K rows)
"""

import argparse
from pathlib import Path

# Dataset configurations
DATASETS = {
    "tickets": {
        "hf_path": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "split": None,
        "description": "Customer support tickets for classification and search",
    },
    "banking": {
        "hf_path": "PolyAI/banking77",
        "split": "train",
        "description": "Banking intent classification (77 intents)",
    },
    "financial_news": {
        "hf_path": "ashraq/financial-news",
        "split": "train[:50000]",
        "description": "Financial news for sentiment analysis",
    },
    "emails": {
        "hf_path": "aeslc",
        "split": "train",
        "description": "Email dataset for summarization",
    },
}

SIZE_LIMITS = {
    "tiny": 100,
    "small": 1000,
    "medium": 10000,
    "full": None,
}


def download_dataset(name: str, config: dict, size: str = "full", output_dir: Path = None):
    """Download a single dataset and save in multiple formats."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return

    output_dir = output_dir or Path(".")
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {name}...")
    print(f"  Source: {config['hf_path']}")
    print(f"  Description: {config['description']}")

    # Load dataset
    try:
        if config["split"]:
            ds = load_dataset(config["hf_path"], split=config["split"])
        else:
            ds = load_dataset(config["hf_path"])
            if hasattr(ds, "train"):
                ds = ds["train"]
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return

    # Apply size limit
    limit = SIZE_LIMITS.get(size)
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
        print(f"  Limited to {limit} rows")

    print(f"  Loaded {len(ds)} rows")

    # Save in multiple formats
    try:
        # Parquet
        parquet_path = dataset_dir / "data.parquet"
        ds.to_parquet(str(parquet_path))
        print(f"  Saved: {parquet_path}")

        # JSONL
        jsonl_path = dataset_dir / "data.jsonl"
        ds.to_json(str(jsonl_path), orient="records", lines=True)
        print(f"  Saved: {jsonl_path}")

        # CSV
        csv_path = dataset_dir / "data.csv"
        ds.to_csv(str(csv_path))
        print(f"  Saved: {csv_path}")

    except Exception as e:
        print(f"  Error saving: {e}")

    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser(description="Download datasets from HuggingFace")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--size",
        choices=list(SIZE_LIMITS.keys()),
        default="small",
        help="Size of dataset to download",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Dataset Downloader")
    print("=" * 50)
    print(f"Size: {args.size} ({SIZE_LIMITS[args.size] or 'unlimited'} rows)")
    print(f"Output: {args.output}")

    if args.dataset == "all":
        for name, config in DATASETS.items():
            download_dataset(name, config, args.size, args.output)
    else:
        config = DATASETS[args.dataset]
        download_dataset(args.dataset, config, args.size, args.output)

    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
