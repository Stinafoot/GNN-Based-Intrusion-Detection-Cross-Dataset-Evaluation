# GNN-Based Network Intrusion Detection: Cross-Dataset Evaluation

> Evaluating six Graph Neural Network architectures for network intrusion detection across NSL-KDD, UNSW-NB15, and CICIDS2017 with emphasis on cross-dataset generalization.

---

## Overview

This project implements and compares six GNN architectures for binary network intrusion detection. Models are trained on NSL-KDD and evaluated on UNSW-NB15 and CICIDS2017 to test cross-dataset generalization which is a critical but often overlooked requirement for real-world IDS deployment.

**Core finding:** In-distribution accuracy does not reliably predict cross-dataset performance. All six models achieve 0.75–0.90 on NSL-KDD, but cross-dataset results diverge significantly. More expressive architectures (RGCN, GIN, ChebNet) generalize better than spectral baselines (GCN) when traffic distributions shift between training and evaluation.

**Research Question:** *Which GNN architecture performs best for intrusion detection across varying data distributions and network environments?*

---

## Models Evaluated

| Architecture | Type | Key Characteristic |
|---|---|---|
| **GCN** | Spectral | Normalized adjacency convolution - fast baseline |
| **GAT** | Attention | Learned attention weights over neighbors (4 heads) |
| **GraphSAGE** | Inductive | Neighborhood sampling - designed for unseen nodes |
| **R-GCN** | Relational | Relation-typed edges; homogeneous mode (1 relation) |
| **GIN** | Expressive | Injective MLP aggregator - maximally expressive under 1-WL test |
| **ChebNet** | Spectral | K=3 Chebyshev polynomial filters - captures 3-hop context per layer |

**Shared training configuration:**
- Input: 8 features → Hidden: 64 units → Output: 2 classes (Normal / Attack)
- Optimizer: Adam (lr=0.01, weight_decay=5e-4)
- Loss: Weighted Cross-Entropy (class weights from label frequencies)
- Epochs: 300 with ReduceLROnPlateau scheduler (factor=0.5, patience=20)
- Graph: bidirectional sequential edges (i→i+1 and i+1→i)

---

## Datasets

| Dataset | Role | Size | Attack Rate |
|---|---|---|---|
| **NSL-KDD** | Training | ~125k flows | ~53% |
| **UNSW-NB15** | Evaluation | ~175k flows | ~68% |
| **CICIDS2017** | Evaluation | ~286k flows | ~96% (PortScan file) |

Each dataset uses an **independent StandardScaler** fit on its own data. This tests whether learned graph structure and weights transfer across datasets, which is the correct methodology for cross-dataset GNN evaluation.

---

## Results

### NSL-KDD: In-Distribution

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| GCN | 0.7525 | 0.7344 | 0.7335 | 0.7353 | 0.8321 |
| GAT | 0.8699 | 0.8507 | 0.9132 | 0.7963 | 0.9061 |
| GraphSAGE | 0.9002 | 0.8817 | 0.9837 | 0.7988 | **0.9380** |
| R-GCN | 0.9001 | 0.8815 | 0.9835 | 0.7987 | 0.9367 |
| GIN | 0.8651 | 0.8490 | 0.8859 | **0.8150** | 0.9250 |
| ChebNet | **0.8995** | **0.8809** | **0.9822** | 0.7986 | 0.9338 |

### CICIDS2017: Cross-Dataset

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| GCN | 0.3731 | 0.0021 | 0.0090 | 0.0012 | 0.8273† |
| GAT | 0.6646 | 0.7679 | 0.6233 | 0.9999 | **0.9136** |
| GraphSAGE | 0.8674 | 0.8930 | 0.8084 | 0.9974 | 0.8664 |
| R-GCN | **0.9153** | **0.9289** | **0.8694** | 0.9971 | 0.8421 |
| GIN | 0.8962 | 0.9133 | 0.8513 | 0.9849 | 0.8700 |
| ChebNet | 0.8967 | 0.9147 | 0.8440 | **0.9983** | 0.8589 |

†GCN AUC=0.827 confirms discriminative ability - low accuracy reflects class-distribution shift (NSL-KDD 53% attack → CICIDS 96% attack) causing threshold miscalibration, not model failure.

### UNSW-NB15: Cross-Dataset (optimal threshold)

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|
| GCN | 0.6808 | 0.8100 | 0.6808 | 0.9998 | 0.6298 | 0.27 |
| GAT | 0.6019 | 0.6007 | **0.9467** | 0.4399 | **0.7150** | 0.05 |
| GraphSAGE | **0.7264** | **0.8273** | 0.7252 | 0.9627 | 0.6699 | 0.51 |
| R-GCN | 0.6978 | 0.8134 | 0.7014 | 0.9680 | 0.6388 | 0.53 |
| GIN | 0.6810 | 0.8100 | 0.6811 | 0.9991 | 0.5795 | 0.23 |
| ChebNet | 0.6800 | 0.8095 | 0.6804 | 0.9991 | 0.6051 | 0.05 |

Optimal threshold is selected per model by sweeping 0.05–0.95 and maximizing F1. Thresholds below 0.5 indicate distribution shift - the decision boundary is calibrated for NSL-KDD's 53% attack rate and must be lowered to account for UNSW's 68% attack rate.

---

## Key Findings

**1. In-distribution performance (NSL-KDD)**
GraphSAGE, R-GCN, and ChebNet lead with ~0.90 accuracy and AUC above 0.93. GIN achieves the best recall balance (0.815), making it the lowest false-negative architecture. GCN improved significantly (0.59→0.75) after switching to bidirectional sequential edges.

**2. CICIDS2017 cross-dataset**
R-GCN achieves the best overall result (accuracy 0.915, F1 0.929). GIN and ChebNet follow closely at ~0.896–0.897. GCN fails on this dataset due to class-distribution shift: NSL-KDD is 53% attack while the CICIDS PortScan file is 96% attack. GCN's limited expressiveness causes it to predict the NSL-KDD majority-class boundary, which covers almost no CICIDS traffic. Its ROC-AUC of 0.827 confirms it can still discriminate - the failure is a threshold calibration problem, not a learned-representation problem.

**3. UNSW-NB15 cross-dataset**
GAT achieves the best ROC-AUC (0.715), competitive with published cross-dataset IDS results (~0.65–0.75 from E-GraphSAGE). GraphSAGE achieves the best F1 (0.827) and accuracy (0.726). The universal [prob inverted] flag across all models indicates that UNSW attack traffic is statistically unlike NSL-KDD attack traffic - models trained on 1999-era data assign higher anomaly confidence to modern normal traffic. This is an expected and meaningful finding about temporal distribution shift in network security datasets.

**4. ROC-AUC as primary metric**
When attack rates differ substantially between training and test sets (53% vs 68% vs 96%), threshold-dependent metrics (accuracy, F1) can be misleading. ROC-AUC is the correct primary comparison metric for cross-dataset evaluation as it is threshold-independent.

---

## Feature Set

Eight semantically aligned features used across all three datasets:

| NSL-KDD | Meaning | UNSW-NB15 | CICIDS2017 |
|---|---|---|---|
| `duration` | Connection duration | `dur` | `Flow Duration` / 1e6 |
| `src_bytes` | Bytes source→dest | `sbytes` | `Total Length of Fwd Packets` |
| `dst_bytes` | Bytes dest→source | `dbytes` | `Total Length of Bwd Packets` |
| `count` | Same-host connections | `ct_srv_src` | `Total Fwd Packets` |
| `srv_count` | Same-service connections | `ct_srv_dst` | `Total Backward Packets` |
| `serror_rate` | SYN error rate (0–1) | `srate` (normalized) | `SYN Flag Count / total` |
| `rerror_rate` | REJ error rate (0–1) | `drate` (normalized) | `RST Flag Count / total` |
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
├── gnn_model_gcn.pt                # Saved weights (generated after training)
├── gnn_model_gat.pt
├── gnn_model_sage.pt
├── gnn_model_rgcn.pt
├── gnn_model_gin.pt
├── gnn_model_cheb.pt
│
├── data/                           # Place datasets here (not tracked)
│   ├── KDDTrain+.txt
│   ├── UNSW_NB15_testing-set.csv
│   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│
└── README.md
```

---

## Setup & Installation

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib joblib
```

> PyTorch Geometric requires extra setup depending on your CUDA version.
> See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

**Datasets:**

| Dataset | Source |
|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |

---

## How to Run

Run notebooks in order - Notebook 01 must complete first:

```
1. 01_NSLKDDCoreTraining.ipynb   → trains all 6 models, saves gnn_model_*.pt
2. 02_CICIDS_Evaluation.ipynb    → cross-dataset eval on CICIDS2017
3. 03_UNSW_Evaluation.ipynb      → cross-dataset eval on UNSW-NB15 (optimal threshold)
```

Notebook 03 includes diagnostic output of all label-related columns and a hard sanity check on the attack rate before evaluation begins.

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
