import os
from pathlib import Path

# Paths Configuration — dynamically resolve repository root
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent if BASE_DIR.name == "WildSpoof_Project" else BASE_DIR

# Primary Dataset: ASVspoof 2019 Logical Access (LA)
LA_DIR = Path(os.environ.get("ASVSPOOF_LA_DIR", str(REPO_ROOT / "LA")))

DATA_DIR = BASE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"

# Ensure output directories exist
for d in [DATA_DIR, MANIFESTS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Audio Preprocessing Configuration
SAMPLE_RATE = 16000
AUDIO_DURATION = 2.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # exactly 32,000 samples
MONO = True
NORMALIZE_AMPLITUDE = True

# Feature Extraction Configuration
# Canonical LFCC: 60 linear filterbanks spanning 0 to 8000 Hz, followed by DCT to 60 coefficients
N_LFCC_FILTERS = 60
N_LFCC = 60
N_FFT = 512
HOP_LENGTH = 160  # 10ms hop
WIN_LENGTH = 400  # 25ms window
N_MELS = 80
FMIN = 0.0
FMAX = float(SAMPLE_RATE // 2)  # Nyquist: 8000 Hz

# Acoustic Parameter Settings
N_MFCC = 20
EMBEDDING_DIM = 64  # 64-D per branch -> 128-D concatenated representation
NUM_CLASSES = 2     # 0: bona fide, 1: spoof
DROPOUT = 0.4

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
EPOCHS = 20
RANDOM_SEED = 42
GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 5
