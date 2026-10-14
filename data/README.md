# ASVspoof 2019 Logical Access (LA) Dataset Setup

Place the official **ASVspoof 2019 Logical Access (LA)** dataset in this directory or in the repository root under `LA/`.

### Expected Directory Layout:
```
LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/
│   └── flac/
├── ASVspoof2019_LA_dev/
│   └── flac/
└── ASVspoof2019_LA_eval/
    └── flac/
```

### Protocol Ingestion & Manifest Generation:
Once the dataset is in place, parse the official protocols into clean CSV manifests:
```bash
python WildSpoof_Project/scripts/prepare_la_manifests.py
```
This generates `train_la.csv`, `dev_la.csv`, and `eval_la.csv` in `WildSpoof_Project/data/manifests/`.

### Download Information:
The ASVspoof 2019 Logical Access database is available from the official ASVspoof consortium via Academic Data Sharing / Edinburgh DataShare:
https://datashare.ed.ac.uk/handle/10283/3336
