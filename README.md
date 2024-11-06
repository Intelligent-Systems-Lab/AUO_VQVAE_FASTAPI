### 1. git clone this repo
### 2. 有train和inference的api
### 3. cd到你要執行的動作(train/inference)
### 4. 照著資料夾內的README操作即可

## File Structure
.
├── inference_vqvae_api
│   ├── code
│   │   ├── configs
│   │   │   └── config.yaml
│   │   ├── data
│   │   ├── dataset
│   │   ├── functions.py
│   │   ├── inference.py
│   │   ├── inference_vqvae_api.py
│   │   ├── modules.py
│   │   ├── run.sh
│   │   ├── status.json
│   │   ├── watchdog_2.py
│   │   └── weights
│   │       └── info.json
│   ├── Dockerfile
│   ├── environment.yml
│   └── README.md
├── README.md
└── train_vqvae_api
    ├── code
    │   ├── configs
    │   │   └── config.yaml
    │   ├── data
    │   ├── dataset.py
    │   ├── functions.py
    │   ├── jobs
    │   │   └── job.json
    │   ├── models
    │   ├── modules.py
    │   ├── run.sh
    │   ├── status.json
    │   ├── train_vqvae_api.py
    │   ├── vqvae.py
    │   └── watchdog_2.py
    ├── Dockerfile
    ├── environment.yml
    └── README.md