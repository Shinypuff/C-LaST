# C-LaST — Contrastive LaST for Time-Series Forecasting

Latent Seasonal-Trend Representations with a contrastive twist.
This repo implements C-LaST, a simplified variant of LaST that replaces heavy MI/ELBO terms with a contrastive similarity objective and adopts probabilistic forecasting via Gaussian NLL. The goal is to keep LaST’s seasonal/trend factorization while making training simpler and more stable

### What`s new

- Simpler training: MI bounds + complex ELBO pieces were replaced with a contrastive loss over temporal neighborhoods. 
- Probabilistic forecasts: from point prediction to Gaussian likelihoods (means + stds) to model uncertainty. 
- Moved code to PyTorch Lightning, keep a single entry point.

### Repo sctructure:

```
├─ config/                # YAML configs for data, model
├─ src/                   # All library code (models, losses, trainers)
├─ Dockerfile             # Reproducible environment
├─ build_image            # Helper to build the Docker image
├─ launch_container       # Helper to run the container
├─ main.py                # Single entry point (train/eval)
├─ requirements.txt       # Pinned runtime dependencies (alt to pyproject)
```

### Datasets used:

```
ETTh1, ETTh2, Illiness
```

### Installation:

#### Option 1
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

#### Option 2
```bash
bash build_image
bash launch_container
```

### Running:

1) Choose the appropriate model, dataset, bacbone in `main.yaml`, options: 
```
CLaST, GLaST, LaST
```
choices for backbone:
```
feednet, transformer
```
2) run `main.py`
```bash
python main.py
```

### References 

- Alexander Marusov, Aleksandr Yugay, A. Z. A theoretical framework for self-supervised contrastive learning for continuous dependent data. Technical report, Skoltech, 2025.
- Woo, G., Liu, Y., Lim, B., Lee, N. C., and Hooi, B. Cost: Contrastive learning of disentangled seasonal-trend representations for time series forecasting. arXiv preprint arXiv:2202.01575, 2022.
- Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y.,and Xu, B. Ts2vec: Towards universal representation of time series. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.
- Zhiyuan Wang, Xovee Xu, Weifeng Zhang, Goce Trajcevski, Ting Zhong, Fan Zhou. Learning latent seasonal-trend representations for time series forecasting. NeurIPS 2022, 2022
