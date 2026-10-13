import os
from pathlib import Path

# Paths Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get("FOR_DATASET_DIR", r"d:\Speech\PROJECT\for-2seconds"))

DATA_DIR = BASE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"

# Ensure output directories exist
for d in [DATA_DIR, MANIFESTS_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Audio Preprocessing Configuration
SAMPLE_RATE = 16000
AUDIO_DURATION = 2.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 32,000 samples
MONO = True
NORMALIZE_AMPLITUDE = True

# Feature Extraction Configuration
N_MFCC = 20
N_LFCC = 20
N_FFT = 512
HOP_LENGTH = 160  # 10ms hop
WIN_LENGTH = 400  # 25ms window
N_MELS = 80
FMIN = 0
FMAX = SAMPLE_RATE // 2  # Nyquist: 8000 Hz

# Experiment Configuration / Sampling Limits for Rapid Experimentation
# Set to None to process full dataset
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None

# Training & Reproducibility Configuration
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 15

# Label Mapping
LABEL_MAP = {
    "real": 0,
    "bonafide": 0,
    "fake": 1,
    "spoof": 1
}

LABEL_NAMES = {
    0: "bonafide",
    1: "spoof"
}
