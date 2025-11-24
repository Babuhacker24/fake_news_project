"""
Step 1 — Clean merged news dataset (no vectorization, no models).

Behavior:
- Always use <repo>/data/processed/merged_news.csv.
- Save to <repo>/artifacts/preprocessed_news.csv.
- Add a 'clean_text' column: lowercase, remove URLs, punctuation, numbers, extra spaces.
- Normalize 'label' column: convert true/false or real/fake to 0/1.

Usage:
    python src/clean.py
    or specify columns manually:
    python src/clean.py --text-col text
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- cleaning ----------
def basic_clean(s: str) -> str:
    """Lowercase, remove URLs, non-letters, collapse spaces."""
    if not isinstance(s, str):
        s = "" if s is np.nan else str(s)
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)     # remove URLs
    s = re.sub(r"[^a-z\s]", " ", s)             # keep only letters/spaces
    s = re.sub(r"\s+", " ", s).strip()          # collapse spaces
    return s


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    input_path = repo_root / "data" / "processed" / "merged_news.csv"
    output_path = repo_root / "artifacts" / "preprocessed_news.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Expected dataset not found at: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", type=str, default="text",
                    help="Name of the text column (default: text)")
    args = ap.parse_args()

    # Load merged dataset
    try:
        df = pd.read_csv(input_path)
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="latin-1")

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in dataset columns: {list(df.columns)}")

    # --- normalize labels to 0/1 if present ---
    if "label" in df.columns:
        df["label"] = (
            df["label"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"true": 1, "false": 0, "real": 1, "fake": 0, "1": 1, "0": 0})
            .astype(int)
        )
        print("[Clean] Normalized 'label' column to 0/1 values.")

    # Clean text column
    df["clean_text"] = df[args.text_col].astype(str).map(basic_clean)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("=== Cleaning complete ===")
    print(f"Input CSV : {input_path}")
    print(f"Text col  : {args.text_col}")
    print(f"Saved to  : {output_path}")
    print("Preview:")
    print(df[[args.text_col, "clean_text"]].head(5))


if __name__ == "__main__":
    main()


def get_clean_df() -> pd.DataFrame:
    """Return the cleaned dataset so other scripts (like embeddings) can import it."""
    path = Path("artifacts/preprocessed_news.csv")
    if not path.exists():
        raise FileNotFoundError("Run src/clean.py first to generate artifacts/preprocessed_news.csv")

    df = pd.read_csv(path)

    # Keep only the cleaned text column and rename it to 'text'
    if "clean_text" in df.columns:
        if "text" in df.columns:
            df = df.drop(columns=["text"])
        df = df.rename(columns={"clean_text": "text"})

    return df
