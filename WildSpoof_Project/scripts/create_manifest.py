import os
import sys
from pathlib import Path
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def create_manifests():
    print("Creating CSV manifests...")
    splits = {
        "training": config.MANIFESTS_DIR / "train.csv",
        "validation": config.MANIFESTS_DIR / "validation.csv",
        "testing": config.MANIFESTS_DIR / "test.csv"
    }

    classes = ["real", "fake"]

    for split_name, manifest_path in splits.items():
        split_dir = config.DATASET_DIR / split_name
        if not split_dir.exists():
            print(f"Skipping {split_name}, directory not found: {split_dir}")
            continue

        records = []
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue

            audio_files = list(cls_dir.glob("*.wav"))
            label = config.LABEL_MAP[cls]
            label_name = config.LABEL_NAMES[label]

            for filepath in tqdm(audio_files, desc=f"Processing {split_name}/{cls}"):
                try:
                    info = sf.info(str(filepath))
                    records.append({
                        "filepath": str(filepath),
                        "label": label,
                        "label_name": label_name,
                        "duration": info.duration,
                        "sample_rate": info.samplerate
                    })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        df = pd.DataFrame(records)
        df.to_csv(manifest_path, index=False)
        print(f"Saved {len(df)} records to {manifest_path}")

    print("\nAll manifests generated successfully!\n")

if __name__ == "__main__":
    create_manifests()
