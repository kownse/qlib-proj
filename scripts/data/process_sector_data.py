"""
Process sector data into features for the data handler.

Loads sector_info.json + AI affinity labels and generates:
  - 11 sector one-hot features (sector_technology, sector_healthcare, etc.)
  - 1 AI affinity score (ai_affinity, normalized to [-1, 1])

Output: my_data/sector_data/sector_features.parquet
  Index: stock symbol
  Columns: 12 features (11 sector + 1 AI affinity)

Usage:
    python scripts/data/process_sector_data.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SECTOR_DATA_DIR = PROJECT_ROOT / "my_data" / "sector_data"
SECTOR_INFO_PATH = SECTOR_DATA_DIR / "sector_info.json"
OUTPUT_PATH = SECTOR_DATA_DIR / "sector_features.parquet"

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Standard GICS sector mapping (Yahoo Finance sector names → canonical names)
SECTOR_MAP = {
    "Technology": "technology",
    "Information Technology": "technology",
    "Communication Services": "communication_services",
    "Consumer Cyclical": "consumer_discretionary",
    "Consumer Discretionary": "consumer_discretionary",
    "Consumer Defensive": "consumer_staples",
    "Consumer Staples": "consumer_staples",
    "Healthcare": "healthcare",
    "Health Care": "healthcare",
    "Financial Services": "financials",
    "Financials": "financials",
    "Industrials": "industrials",
    "Energy": "energy",
    "Utilities": "utilities",
    "Real Estate": "real_estate",
    "Basic Materials": "materials",
    "Materials": "materials",
}

# All canonical sector names (for one-hot columns)
CANONICAL_SECTORS = [
    "technology",
    "healthcare",
    "financials",
    "consumer_discretionary",
    "consumer_staples",
    "communication_services",
    "industrials",
    "energy",
    "utilities",
    "real_estate",
    "materials",
]


def process_sector_features(sector_info_path: Path = SECTOR_INFO_PATH,
                            output_path: Path = OUTPUT_PATH):
    """
    Process sector info JSON + AI affinity into a features parquet.

    Returns:
        DataFrame with sector one-hot + AI affinity features, indexed by symbol
    """
    from data.ai_affinity_labels import AI_AFFINITY_SCORES

    # Load sector info
    if not sector_info_path.exists():
        raise FileNotFoundError(
            f"Sector info not found: {sector_info_path}\n"
            "Run: python scripts/data/download_sector_data.py"
        )

    with open(sector_info_path) as f:
        sector_info = json.load(f)

    print(f"Loaded sector info for {len(sector_info)} stocks")

    # Build features DataFrame
    rows = []
    for symbol, info in sorted(sector_info.items()):
        raw_sector = info.get("sector", "Unknown")
        canonical = SECTOR_MAP.get(raw_sector, None)

        # One-hot encode sectors
        row = {"symbol": symbol}
        for s in CANONICAL_SECTORS:
            row[f"sector_{s}"] = 1.0 if canonical == s else 0.0

        # AI affinity score (normalize from [-2, 2] to [-1, 1])
        raw_score = AI_AFFINITY_SCORES.get(symbol, 0)
        row["ai_affinity"] = raw_score / 2.0

        rows.append(row)

    df = pd.DataFrame(rows).set_index("symbol")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    print(f"Saved {df.shape[0]} stocks × {df.shape[1]} features to {output_path}")
    print(f"\nFeature columns: {df.columns.tolist()}")
    print(f"\nSector distribution:")
    for col in [c for c in df.columns if c.startswith("sector_")]:
        count = int(df[col].sum())
        if count > 0:
            print(f"  {col}: {count}")

    print(f"\nAI affinity distribution:")
    affinity_counts = df["ai_affinity"].value_counts().sort_index()
    for val, count in affinity_counts.items():
        label = {-1.0: "Strong disruption", -0.5: "Moderate disruption",
                 0.0: "Neutral", 0.5: "Moderate beneficiary", 1.0: "Strong beneficiary"}
        print(f"  {val:+.1f} ({label.get(val, '?')}): {count}")

    return df


def main():
    df = process_sector_features()
    print(f"\nSample data:")
    print(df.head(10))


if __name__ == "__main__":
    main()
