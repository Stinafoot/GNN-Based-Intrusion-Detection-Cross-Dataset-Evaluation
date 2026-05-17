# GNN-Based Network Intrusion Detection: Cross-Dataset Evaluation

> Evaluating six Graph Neural Network architectures for network intrusion detection across NSL-KDD, UNSW-NB15, and CICIDS2017 — with emphasis on cross-dataset generalization.

---

## Overview

This project investigates how well Graph Neural Network (GNN) models generalize across different network intrusion detection benchmark datasets. Six architectures are trained on NSL-KDD and evaluated on UNSW-NB15 and CICIDS2017 to assess cross-dataset robustness — a critical but often overlooked aspect of real-world intrusion detection system (IDS) deployment.

A key finding of this study: **in-distribution accuracy does not reliably predict cross-dataset performance.** All six models achieve 0.75–0.90 on NSL-KDD, but generalization varies significantly, with more expressive architectures (RGCN, GIN, ChebNet) outperforming spectral baselines (GCN) on unseen traffic distributions.

**Research Question:** *Which GNN architecture performs best for intrusion detection across varying data distributions and network environments?*

---

## Models Evaluated

| Architecture | Type | Key Characteristic |
|---|---|---|
| **GCN** | Spectral | Normalized adjacency convolution — fast baseline |
| **GAT** | Attention | Learned attention weights over neighbors (4 heads) |
| **GraphSAGE** | Inductive | Neighborhood sampling — designed for unseen nodes |
| **R-GCN** | Relational | Relation-typed edges; homogeneous mode (1 relation) |
| **GIN** | Expressive | Injective MLP aggregator — maximally expressive under 1-WL test |
| **ChebNet** | Spectral | K=3 Chebyshev polynomial filters — captures 3-hop context per layer |

All models share the same training template:
- Input (8 features) → Hidden layer (64 units) → Output (2 classes: Normal / Attack)
- Binary classification: Normal (0) vs. Attack (1)
- Optimizer: Adam (lr=0.01) | Loss: Weighted Cross-Entropy | Epochs: 300
- LR scheduler: ReduceLROnPlateau (halves LR on plateau, min=1e-4)
- Graph: **bidirectional** sequential edges (i→i+1 and i+1→i) for richer neighborhood context

---

## Datasets

| Dataset | Role | Description |
|---|---|---|
| **NSL-KDD** | Training | Improved KDD Cup 1999 benchmark; removes redundant records. ~125k flows, ~53% attack. |
| **UNSW-NB15** | Evaluation | Modern synthetic traffic with 9 attack categories. ~175k flows, ~46% attack. |
| **CICIDS2017** | Evaluation | Realistic lab-captured attack scenarios (PortScan file). ~286k flows, ~96% attack. |

Each dataset is independently normalized with its own `StandardScaler`. This tests whether the **learned graph structure and weights transfer** across datasets — not whether raw feature scales do — which is the correct methodology for cross-dataset GNN evaluation.

---

## Feature Set

Eight semantically aligned features are used across all three datasets:

| NSL-KDD Feature | Meaning | UNSW-NB15 Mapping | CICIDS2017 Mapping |
|---|---|---|---|
| `duration` | Connection duration | `dur` | `Flow Duration` / 1e6 |
| `src_bytes` | Bytes source→dest | `sbytes` | `Total Length of Fwd Packets` |
| `dst_bytes` | Bytes dest→source | `dbytes` | `Total Length of Bwd Packets` |
| `count` | Same-host connections | `ct_srv_src` | `Total Fwd Packets` |
| `srv_count` | Same-service connections | `ct_srv_dst` | `Total Backward Packets` |
| `serror_rate` | SYN error rate (0–1) | `srate` (normalized) | `SYN Flag Count / total pkts` |
| `rerror_rate` | REJ error rate (0–1) | `drate` (normalized) | `RST Flag Count / total pkts` |
| `same_srv_rate` | Same-service rate (0–1) | `ct_srv_src` (normalized) | `Avg Fwd Segment Size` (normalized) |

---

## Project Structure

```
gnn-intrusion-detection/
│
├── 01_NSLKDDCoreTraining.ipynb     # EDA + trains all 6 GNN models on NSL-KDD
├── 02_CICIDS_Evaluation.ipynb      # Cross-dataset eval on CICIDS2017
├── 03_UNSW_Evaluation.ipynb        # Cross-dataset eval on UNSW-NB15
│
├── gnn_model_gcn.pt                # Saved GCN weights    (generated after training)
├── gnn_model_gat.pt                # Saved GAT weights
├── gnn_model_sage.pt               # Saved GraphSAGE weights
├── gnn_model_rgcn.pt               # Saved R-GCN weights
├── gnn_model_gin.pt                # Saved GIN weights
├── gnn_model_cheb.pt               # Saved ChebNet weights
│
├── data/                           # Place dataset CSVs here (not tracked — see below)
│   ├── KDDTrain+.txt
│   ├── UNSW_NB15_testing-set.csv
│   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│
└── README.md
```

---

## Setup & Installation

### Requirements

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib joblib
```

> PyTorch Geometric may require extra steps depending on your CUDA version.
> See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Datasets

Download and place in the project root (or update the file paths in each notebook):

| Dataset | Source |
|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |

---

## How to Run

Run notebooks **in order** — Notebook 01 must complete before 02 or 03:

```
1. 01_NSLKDDCoreTraining.ipynb
   → Loads NSL-KDD, trains all 6 GNNs, saves gnn_model_*.pt weights

2. 02_CICIDS_Evaluation.ipynb
   → Loads saved weights, evaluates cross-dataset on CICIDS2017

3. 03_UNSW_Evaluation.ipynb
   → Loads saved weights, evaluates cross-dataset on UNSW-NB15
```

Each eval notebook fits its own `StandardScaler` independently and prints a feature alignment check before evaluation runs.

---

## Evaluation Metrics

Each model is evaluated on:

- **Accuracy** — overall correct predictions
- **Precision** — of predicted attacks, how many were real attacks
- **Recall** — of real attacks, how many were detected
- **F1-Score** — harmonic mean of precision and recall
- **ROC-AUC** — threshold-independent ranking quality (most reliable metric for cross-dataset comparison)

Results are displayed as bar charts comparing all 6 architectures side by side per dataset.

---

## Key Findings

| Result | Detail |
|---|---|
| Best in-distribution (NSL-KDD) | SAGE / RGCN / ChebNet — ~0.90 accuracy, precision >0.98 |
| Best recall balance | GIN — 0.865 accuracy with lowest false-negative rate |
| Best cross-dataset (CICIDS2017) | RGCN — 0.915 accuracy / 0.929 F1 |
| GCN distribution-shift failure | ROC-AUC 0.83 but predicts nearly all flows as normal on CICIDS (96% attack file vs 53% NSL-KDD) |
| Bidirectional edges impact | GCN in-distribution accuracy: 0.59 → 0.75 after adding reverse edges |
| UNSW label ambiguity | Labels derived from `attack_cat` column to avoid encoding inconsistencies across file versions |

**Summary:** More expressive architectures (RGCN, GIN, ChebNet) generalize better across datasets. GCN's spectral averaging is brittle to class-distribution shift. ROC-AUC is the most reliable metric when attack rates differ significantly between train and test sets.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-blue?logo=scikit-learn)

---

## Author

**Christina Barefoot**  
B.S. Cybersecurity and Operations | Mississippi State University  
[LinkedIn](https://www.linkedin.com/in/christina-barefoot) · [GitHub](https://github.com/Stinafoot)

---

## License

This project is for academic purposes. Datasets are subject to their respective terms of use.
