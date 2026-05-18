# GNN-Based Network Intrusion Detection: Cross-Dataset Evaluation

Comparing six Graph Neural Network architectures for binary network intrusion detection, trained on NSL-KDD and evaluated on CICIDS2017 and UNSW-NB15 to measure real-world cross-dataset generalization.

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-orange)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-blue?logo=scikit-learn)

---

## Research Question

> *Which GNN architecture generalizes best for intrusion detection across varying data distributions and network environments?*

**Core finding:** In-distribution accuracy does not reliably predict cross-dataset performance. All six models achieve 0.75–0.90 on NSL-KDD, but cross-dataset results diverge significantly. More expressive architectures (R-GCN, GIN, ChebNet) generalize better than spectral baselines (GCN) when traffic distributions shift between training and evaluation.

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

### Shared Training Configuration

| Parameter | Value |
|-----------|-------|
| Input → Hidden → Output | 8 → 64 → 2 |
| Optimizer | Adam (lr=0.01, weight_decay=5e-4) |
| Loss | Weighted CrossEntropyLoss |
| Epochs | 300 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=20) |
| Graph structure | Bidirectional sequential edges (i→i+1 and i+1→i) |

The bidirectional edge scheme is critical: it gives every interior node a 2-node neighbourhood instead of 1, which is the minimum needed for GCN to converge on chain-structured graphs.

---

## Datasets

| Dataset | Role | Size | Attack Rate |
|---|---|---|---|
| **NSL-KDD** | Training | ~125k flows | ~53% |
| **UNSW-NB15** | Cross-dataset eval | ~175k flows | ~68% |
| **CICIDS2017** | Cross-dataset eval | ~286k flows | ~96% (PortScan file) |

Each dataset uses an **independent `StandardScaler`** fit only on its own data. This correctly isolates what is being tested: whether the learned graph structure and weights transfer across datasets, not whether raw feature scales transfer.

| Dataset | Source |
|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |

---

## Feature Set

Eight semantically aligned features extracted consistently across all three datasets:

| Feature | Meaning | NSL-KDD | UNSW-NB15 | CICIDS2017 |
|---|---|---|---|---|
| `duration` | Connection duration | `duration` | `dur` | `Flow Duration` / 1e6 |
| `src_bytes` | Bytes source→dest | `src_bytes` | `sbytes` | `Total Length of Fwd Packets` |
| `dst_bytes` | Bytes dest→source | `dst_bytes` | `dbytes` | `Total Length of Bwd Packets` |
| `count` | Same-host connections | `count` | `ct_srv_src` | `Total Fwd Packets` |
| `srv_count` | Same-service connections | `srv_count` | `ct_srv_dst` | `Total Backward Packets` |
| `serror_rate` | SYN error rate (0–1) | `serror_rate` | `srate` normalized | `SYN Flag Count / total` |
| `rerror_rate` | REJ error rate (0–1) | `rerror_rate` | `drate` normalized | `RST Flag Count / total` |
| `same_srv_rate` | Same-service rate (0–1) | `same_srv_rate` | `ct_srv_src` normalized | `Avg Fwd Segment Size` normalized |

---

## Results

### NSL-KDD: In-Distribution Test

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

†GCN's ROC-AUC of 0.827 confirms discriminative ability is intact. The near-zero accuracy reflects class-distribution shift (NSL-KDD 53% attack → CICIDS 96% attack) causing threshold miscalibration - a calibration problem, not a representation problem.

### UNSW-NB15: Cross-Dataset (Optimal Threshold)

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|
| GCN | 0.6808 | 0.8100 | 0.6808 | 0.9998 | 0.6298 | 0.27 |
| GAT | 0.6019 | 0.6007 | **0.9467** | 0.4399 | **0.7150** | 0.05 |
| GraphSAGE | **0.7264** | **0.8273** | 0.7252 | 0.9627 | 0.6699 | 0.51 |
| R-GCN | 0.6978 | 0.8134 | 0.7014 | 0.9680 | 0.6388 | 0.53 |
| GIN | 0.6810 | 0.8100 | 0.6811 | 0.9991 | 0.5795 | 0.23 |
| ChebNet | 0.6800 | 0.8095 | 0.6804 | 0.9991 | 0.6051 | 0.05 |

Optimal threshold is found per model by sweeping 0.05–0.95 and maximizing F1. Thresholds below 0.5 indicate the model's decision boundary was calibrated for NSL-KDD's 53% attack rate and must be shifted downward for UNSW's 68% attack rate.

---

## Key Findings

**1. In-distribution (NSL-KDD):** GraphSAGE, R-GCN, and ChebNet lead at ~0.90 accuracy and AUC above 0.93. GIN achieves the best recall balance (0.815) making it the lowest false-negative architecture - the most important property in a security context. GCN improved substantially (0.59→0.75) after switching to bidirectional sequential edges, confirming that graph construction is as important as architecture choice.

**2. CICIDS2017 cross-dataset:** R-GCN achieves the best overall transfer (accuracy 0.915, F1 0.929). GIN and ChebNet follow closely at ~0.896–0.897. GCN's collapse to near-zero F1 is a threshold calibration failure, not a representation failure - its ROC-AUC of 0.827 proves it can still separate classes. The NSL-KDD majority-class decision boundary simply does not cover CICIDS traffic at the default 0.5 threshold.

**3. UNSW-NB15 cross-dataset:** GAT achieves the best ROC-AUC (0.715), competitive with published cross-dataset IDS baselines (~0.65–0.75 from E-GraphSAGE). The universal probability-inversion flag across all models indicates UNSW attack traffic is statistically unlike NSL-KDD attack traffic - models trained on 1999-era data assign higher anomaly confidence to modern normal traffic. This is an expected and meaningful finding about temporal distribution shift in network security datasets.

**4. ROC-AUC as primary metric:** When attack rates differ substantially between training and evaluation sets (53% vs 68% vs 96%), threshold-dependent metrics such as accuracy and F1 can be highly misleading. ROC-AUC is the correct primary metric for cross-dataset evaluation because it is threshold-independent.

---

## Project Structure

```
gnn-intrusion-detection/
│
├── 01_NSLKDDCoreTraining_FINALClean.ipynb    # EDA + trains all 6 GNNs on NSL-KDD
├── 02_CICIDS_Evaluation_FINALClean.ipynb     # Cross-dataset eval on CICIDS2017
├── 03_UNSW_Evaluation_FINALClean.ipynb       # Cross-dataset eval on UNSW-NB15
│
├── test_gnn.py                               # Unit tests (pytest, ~55 tests)
│
├── gnn_model_gcn.pt                          # Saved weights (generated after training)
├── gnn_model_gat.pt
├── gnn_model_sage.pt
├── gnn_model_rgcn.pt
├── gnn_model_gin.pt
├── gnn_model_cheb.pt
│
├── ids_scaler.pkl                            # NSL-KDD StandardScaler (saved by NB1)
│
├── data/                                     # Place datasets here (not tracked)
│   ├── KDDTrain+.txt
│   ├── UNSW_NB15_testing-set.csv
│   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│
└── README.md
```

---

## Setup & Installation

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib joblib pytest
```

> PyTorch Geometric requires platform-specific installation depending on your CUDA version.
> See the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## How to Run

Notebooks must be run in order - Notebook 01 trains and saves all model weights:

```bash
# Step 1: Train all 6 GNN architectures on NSL-KDD
jupyter notebook 01_NSLKDDCoreTraining_FINALClean.ipynb

# Step 2: Cross-dataset evaluation on CICIDS2017
jupyter notebook 02_CICIDS_Evaluation_FINALClean.ipynb

# Step 3: Cross-dataset evaluation on UNSW-NB15 (optimal threshold sweep)
jupyter notebook 03_UNSW_Evaluation_FINALClean.ipynb

# Run unit tests
pytest test_gnn.py -v
```

Notebook 03 includes diagnostic output of all label-related columns and a hard sanity check on attack rate before evaluation begins. If the attack rate falls outside 20–80%, labels are automatically flipped and a warning is printed.

---

## Unit Tests

`test_gnn.py` contains ~55 tests across 8 classes covering every major component:

| Test Class | What It Covers |
|---|---|
| `TestGNNModelConstruction` | All 6 architectures instantiate, unknown type raises `ValueError`, GAT head dims, GIN uses `GINConv`, ChebConv uses K=3 |
| `TestGNNModelForward` | Output shape, log-softmax validity, no NaN/Inf, dropout OFF in eval, dropout ON in train, binary predictions only |
| `TestMakeGraph` | Node/edge counts, bidirectional edges, dtypes, split mask correctness, no overlap, 2-node minimum |
| `TestEvaluate` | All metric keys returned, values in [0,1], ROC-AUC ≥ 0.5, model in eval mode after call |
| `TestFeaturePreprocessing` | Zero mean, unit variance, Inf/NaN replacement, no test-set leakage |
| `TestTrainingLoop` | Loss decreases over 20 epochs, gradients flow, class weights shape, save/load identity |
| `TestCrossDatasetEval` | Threshold sweep range, AUC flip logic, independent scalers per dataset |
| `TestEdgeCases` | Single-class graphs, large/zero feature values, device consistency |

---

## .gitignore

```gitignore
# Large binary files - regenerate by running the notebooks
*.pt
ids_scaler.pkl

# Datasets - download separately per license terms
data/

# Python
__pycache__/
*.pyc
.env
*.ipynb_checkpoints
```

---

## Author

**Christina Barefoot**
B.S. Cybersecurity and Operations | Mississippi State University
[LinkedIn](https://www.linkedin.com/in/christina-barefoot) · [GitHub](https://github.com/Stinafoot)

---

## License

Academic project. Datasets are subject to their respective terms of use (NSL-KDD, UNSW-NB15, CICIDS2017).
