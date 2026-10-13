import json
import os
from pathlib import Path

notebooks_dir = Path(r"d:\Speech\PROJECT\WildSpoof_Project\notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

def create_notebook(nb_path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {nb_path}")

# 1. Notebook 01: Dataset Analysis
nb1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01 — Dataset Inspection & Analysis\n",
            "This notebook inspects the `for-2seconds` dataset structure, computes counts across train/validation/test splits, and plots duration, class, and sample rate distributions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "sys.path.insert(0, str(Path.cwd().parent))\n",
            "from scripts import analyze_dataset\n",
            "\n",
            "df_stats = analyze_dataset.inspect_dataset()\n",
            "df_stats.head(15)"
        ]
    }
]

# 2. Notebook 02: Preprocessing
nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02 — Audio Preprocessing & Exploratory Speech Analysis\n",
            "Demonstrates 16 kHz mono resampling, peak normalization, duration padding/cropping, and plots real vs. synthetic speech waveforms, STFT spectrograms, Mel spectrograms, MFCCs, and LFCCs."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "sys.path.insert(0, str(Path.cwd().parent))\n",
            "from scripts import extract_features\n",
            "\n",
            "extract_features.plot_exploratory_analysis()"
        ]
    }
]

# 3. Notebook 03: MFCC & LFCC Baselines
nb3_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03 — Classical ML Baselines (SVM + MFCC, SVM + LFCC, GMM + LFCC)\n",
            "Trains and evaluates classical speech processing features using Support Vector Machines and Gaussian Mixture Models."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import pandas as pd\n",
            "from pathlib import Path\n",
            "sys.path.insert(0, str(Path.cwd().parent))\n",
            "from src import config\n",
            "\n",
            "baseline_df = pd.read_csv(config.METRICS_DIR / 'baseline_comparison.csv')\n",
            "baseline_df"
        ]
    }
]

# 4. Notebook 04: CNN Deep Learning Model
nb4_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 04 — PyTorch Mel-Spectrogram CNN Classifier\n",
            "Trains a 2D Convolutional Neural Network on Log-Mel Spectrograms with BCE loss, Adam optimizer, early stopping, and checkpointing."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import pandas as pd\n",
            "from pathlib import Path\n",
            "sys.path.insert(0, str(Path.cwd().parent))\n",
            "from src import config\n",
            "\n",
            "results_df = pd.read_csv(config.METRICS_DIR / 'baseline_comparison.csv')\n",
            "results_df[results_df['Model'] == 'CNN']"
        ]
    }
]

# 5. Notebook 05: Robustness Testing
nb5_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 05 — Robustness & In-the-Wild Acoustic Degradation Simulation\n",
            "Evaluates the trained detector under additive Gaussian noise (20dB, 10dB, 5dB), room reverberation, and combined acoustic corruption."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import pandas as pd\n",
            "from pathlib import Path\n",
            "sys.path.insert(0, str(Path.cwd().parent))\n",
            "from src import config\n",
            "\n",
            "robustness_df = pd.read_csv(config.METRICS_DIR / 'robustness_results.csv')\n",
            "robustness_df"
        ]
    }
]

create_notebook(notebooks_dir / "01_dataset_analysis.ipynb", nb1_cells)
create_notebook(notebooks_dir / "02_audio_preprocessing.ipynb", nb2_cells)
create_notebook(notebooks_dir / "03_mfcc_lfcc_baseline.ipynb", nb3_cells)
create_notebook(notebooks_dir / "04_cnn_model.ipynb", nb4_cells)
create_notebook(notebooks_dir / "05_robustness_testing.ipynb", nb5_cells)
print("All 5 notebooks successfully generated!")
