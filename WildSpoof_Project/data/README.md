# Manifests and Processed Cache Directory

This directory stores intermediate manifests and feature caches generated from the canonical **ASVspoof 2019 Logical Access (LA)** dataset.

To generate manifests from raw protocols:
```bash
python WildSpoof_Project/scripts/prepare_la_manifests.py
```
Generated files:
- `manifests/train_la.csv` (25,380 entries, attacks A01–A06)
- `manifests/dev_la.csv` (24,844 entries, attacks A01–A06)
- `manifests/eval_la.csv` (71,237 entries, attacks A07–A19)
