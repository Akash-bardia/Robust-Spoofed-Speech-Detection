import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def inspect_dataset():
    print(f"Inspecting dataset at: {config.DATASET_DIR}")
    if not config.DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {config.DATASET_DIR}")

    splits = ["training", "validation", "testing"]
    classes = ["real", "fake"]

    records = []

    for split in splits:
        split_dir = config.DATASET_DIR / split
        if not split_dir.exists():
            print(f"Warning: Split dir {split_dir} does not exist!")
            continue

        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"Warning: Class dir {cls_dir} does not exist!")
                continue

            audio_files = list(cls_dir.glob("*.wav"))
            print(f"Found {len(audio_files)} files in {split}/{cls}")

            # Sample a subset for detailed audio metadata if large, or process all
            # Since files are 2sec wavs, we can quickly inspect using soundfile header
            for filepath in tqdm(audio_files, desc=f"Analyzing {split}/{cls}"):
                try:
                    info = sf.info(str(filepath))
                    records.append({
                        "split": split,
                        "class": cls,
                        "label": config.LABEL_MAP[cls],
                        "filepath": str(filepath),
                        "duration": info.duration,
                        "sample_rate": info.samplerate,
                        "channels": info.channels
                    })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    df = pd.DataFrame(records)
    print(f"\nTotal files processed: {len(df)}")

    # Compute Statistics Table
    stats = []
    
    total_files = len(df)
    train_df = df[df["split"] == "training"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "testing"]

    stats_dict = {
        "Metric": [
            "Number of training files",
            "Number of validation files",
            "Number of testing files",
            "Real training samples",
            "Fake training samples",
            "Real validation samples",
            "Fake validation samples",
            "Real testing samples",
            "Fake testing samples",
            "Mean duration (sec)",
            "Minimum duration (sec)",
            "Maximum duration (sec)",
            "Sample-rate distribution"
        ],
        "Value": [
            len(train_df),
            len(val_df),
            len(test_df),
            len(train_df[train_df["class"] == "real"]),
            len(train_df[train_df["class"] == "fake"]),
            len(val_df[val_df["class"] == "real"]),
            len(val_df[val_df["class"] == "fake"]),
            len(test_df[test_df["class"] == "real"]),
            len(test_df[test_df["class"] == "fake"]),
            f"{df['duration'].mean():.4f}",
            f"{df['duration'].min():.4f}",
            f"{df['duration'].max():.4f}",
            str(dict(df['sample_rate'].value_counts()))
        ]
    }

    stats_df = pd.DataFrame(stats_dict)
    stats_csv_path = config.METRICS_DIR / "dataset_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\nDataset Statistics saved to {stats_csv_path}:\n")
    print(stats_df.to_string(index=False))

    # Generate Figures
    sns.set_theme(style="whitegrid")
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="split", hue="class", palette="viridis")
    plt.title("FoR-2seconds Dataset Class Distribution across Splits", fontsize=14, fontweight="bold")
    plt.xlabel("Dataset Split", fontsize=12)
    plt.ylabel("Number of Utterances", fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "dataset_class_distribution.png", dpi=300)
    plt.close()

    # 2. Duration Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="duration", hue="class", kde=False, bins=30, palette="magma")
    plt.title("Audio Duration Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Duration (seconds)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "dataset_duration_distribution.png", dpi=300)
    plt.close()

    # 3. Sample Rate Distribution Plot
    plt.figure(figsize=(6, 4))
    sr_counts = df["sample_rate"].value_counts().reset_index()
    sr_counts.columns = ["Sample Rate (Hz)", "Count"]
    sns.barplot(data=sr_counts, x="Sample Rate (Hz)", y="Count", palette="crest")
    plt.title("Sample Rate Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "dataset_sample_rate_distribution.png", dpi=300)
    plt.close()

    print("\nDataset analysis complete! Figures generated in results/figures/\n")
    return df

if __name__ == "__main__":
    inspect_dataset()
