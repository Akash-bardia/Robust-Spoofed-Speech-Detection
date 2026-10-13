import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LA_DIR = Path(r"d:\Speech\PROJECT\LA")
PROTO_DIR = LA_DIR / "ASVspoof2019_LA_cm_protocols"
OUT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
OUT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# ASVspoof 2019 System ID descriptions
ATTACK_INFO = {
    "A01": {"type": "TTS", "method": "neural_waveform", "known": True},
    "A02": {"type": "TTS", "method": "vocoder", "known": True},
    "A03": {"type": "TTS", "method": "vocoder", "known": True},
    "A04": {"type": "TTS", "method": "waveform_concatenation", "known": True},
    "A05": {"type": "VC",  "method": "vocoder", "known": True},
    "A06": {"type": "VC",  "method": "spectral_filtering", "known": True},
    "A07": {"type": "TTS", "method": "vocoder_gan", "known": False},
    "A08": {"type": "TTS", "method": "neural_waveform", "known": False},
    "A09": {"type": "TTS", "method": "vocoder", "known": False},
    "A10": {"type": "TTS", "method": "neural_waveform", "known": False},
    "A11": {"type": "TTS", "method": "griffin_lim", "known": False},
    "A12": {"type": "TTS", "method": "neural_waveform", "known": False},
    "A13": {"type": "TTS_VC", "method": "waveform_concat_filtering", "known": False},
    "A14": {"type": "TTS_VC", "method": "vocoder", "known": False},
    "A15": {"type": "TTS_VC", "method": "neural_waveform", "known": False},
    "A16": {"type": "TTS", "method": "waveform_concatenation", "known": False},
    "A17": {"type": "VC",  "method": "waveform_filtering", "known": False},
    "A18": {"type": "VC",  "method": "vocoder", "known": False},
    "A19": {"type": "VC",  "method": "spectral_filtering", "known": False},
}

def parse_protocol_file(protocol_path, audio_dir, split_name):
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    records = []
    print(f"Parsing {split_name} protocol: {protocol_path}...")
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id = parts[0]
            audio_id = parts[1]
            system_id = parts[3]
            key = parts[4].lower()

            audio_file = audio_dir / f"{audio_id}.flac"
            is_bonafide = (key == "bonafide")
            label = 0 if is_bonafide else 1

            if is_bonafide:
                attack_type = "bonafide"
                attack_method = "bonafide"
                is_known = True
            else:
                attack_meta = ATTACK_INFO.get(system_id, {"type": "unknown", "method": "unknown", "known": False})
                attack_type = attack_meta["type"]
                attack_method = attack_meta["method"]
                is_known = attack_meta["known"]

            records.append({
                "speaker_id": speaker_id,
                "audio_id": audio_id,
                "filepath": str(audio_file),
                "system_id": system_id if not is_bonafide else "bonafide",
                "attack_type": attack_type,
                "attack_method": attack_method,
                "key": key,
                "label": label,
                "is_known_attack": is_known,
                "split": split_name,
                "file_exists": audio_file.exists()
            })

    df = pd.DataFrame(records)
    print(f"  --> Total {split_name} records: {len(df)}")
    missing = len(df[~df["file_exists"]])
    if missing > 0:
        print(f"  [WARNING] {missing} audio files not found on disk!")
    else:
        print("  --> All audio files verified to exist on disk.")

    counts = df["key"].value_counts().to_dict()
    print(f"  --> Class distribution: {counts}")
    return df

def generate_all_manifests():
    splits = [
        ("train", PROTO_DIR / "ASVspoof2019.LA.cm.train.trn.txt", LA_DIR / "ASVspoof2019_LA_train" / "flac"),
        ("dev",   PROTO_DIR / "ASVspoof2019.LA.cm.dev.trl.txt",   LA_DIR / "ASVspoof2019_LA_dev" / "flac"),
        ("eval",  PROTO_DIR / "ASVspoof2019.LA.cm.eval.trl.txt",  LA_DIR / "ASVspoof2019_LA_eval" / "flac"),
    ]

    manifest_summary = []
    for split_name, proto_file, audio_path in splits:
        df = parse_protocol_file(proto_file, audio_path, split_name)
        out_path = OUT_MANIFEST_DIR / f"{split_name}_la.csv"
        df.to_csv(out_path, index=False)
        print(f"  --> Saved {out_path} ({len(df)} rows)")

        manifest_summary.append({
            "Split": split_name,
            "Total Utterances": len(df),
            "Bona Fide": (df["label"] == 0).sum(),
            "Spoof": (df["label"] == 1).sum(),
            "Unique Speakers": df["speaker_id"].nunique(),
            "Attacks": sorted(df[df["system_id"] != "bonafide"]["system_id"].unique().tolist())
        })

    summary_df = pd.DataFrame(manifest_summary)
    summary_path = PROJECT_ROOT / "results" / "metrics" / "la_dataset_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print("\n=== ASVspoof 2019 LA Dataset Summary ===")
    print(summary_df.to_string(index=False))
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    generate_all_manifests()
