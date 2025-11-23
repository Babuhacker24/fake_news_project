import pandas as pd
import zipfile
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def load_true_news():
    true_path = RAW_DIR / "True_5_record.csv"
    df_true = pd.read_csv(true_path)
    df_true["label"] = "true"
    return df_true[["title", "text", "label"]]

def load_fake_news():
    fake_zip_path = RAW_DIR / "Fake.csv.zip"

    # Unzip fake news file
    with zipfile.ZipFile(fake_zip_path, "r") as z:
        z.extractall(RAW_DIR)

    fake_csv_path = RAW_DIR / "Fake.csv"
    df_fake = pd.read_csv(fake_csv_path)
    df_fake["label"] = "fake"
    return df_fake[["title", "text", "label"]]

def merge_datasets():
    df_true = load_true_news()
    df_fake = load_fake_news()

    df = pd.concat([df_true, df_fake], ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "merged_news.csv"
    df.to_csv(output_path, index=False)

    print(f"Merged dataset created at: {output_path}")

if __name__ == "__main__":
    merge_datasets()

