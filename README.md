# GNN-Based Intrusion Detection: Cross-Dataset Evaluation

> Evaluating Graph Neural Network architectures (GCN, GAT, GraphSAGE) for network intrusion detection across NSL-KDD, UNSW-NB15, and CICIDS2017 datasets.

---

## Overview

This project investigates how well Graph Neural Network (GNN) models generalize across different network intrusion detection benchmark datasets. Models are trained on NSL-KDD and then evaluated on UNSW-NB15 and CICIDS2017 to assess cross-dataset robustness whic is a critical but often overlooked aspect of real-world intrusion detection system (IDS) deployment.

**Research Question:** *Which GNN architecture performs best for intrusion detection across varying data distributions and network environments?*

---

## Models Evaluated

| Architecture | Description |
|---|---|
| **GCN** | Graph Convolutional Network — aggregates neighbor features via spectral convolution |
| **GAT** | Graph Attention Network — uses attention weights to prioritize relevant neighbors |
| **GraphSAGE** | Inductive learning via neighborhood sampling — generalizes to unseen nodes |

All models share the same architecture template:
- Input layer → Hidden layer (64 units) → Output layer (2 classes: Normal / Attack)
- Binary classification: Normal (0) vs. Attack (1)
- Optimizer: Adam | Loss: Cross-Entropy

---

## Datasets

| Dataset | Description |
|---|---|
| **NSL-KDD** | Improved KDD Cup 1999 benchmark; removes redundant records. Used for **training**. |
| **UNSW-NB15** | Modern synthetic traffic with diverse attack types. Used for **evaluation**. |
| **CICIDS2017** | Realistic attack scenarios in a controlled lab environment. Used for **evaluation**. |

Graph construction uses sequential edges (row i → row i+1) since these datasets lack real network topology. Nodes represent network flows; features are standardized using a scaler fit on NSL-KDD.

---

## Project Structure

```
gnn-intrusion-detection/
│
├── NSLKDDCoreTraining_fixed.ipynb    # EDA + training GCN/GAT/GraphSAGE on NSL-KDD
├── UNSW_Evaluation_fixed.ipynb       # Cross-dataset eval on UNSW-NB15
├── CICIDS_Evaluation_fixed.ipynb     # Cross-dataset eval on CICIDS2017
│
├── gnn_model_gcn.pt                  # Saved GCN weights (generated after training)
├── gnn_model_gat.pt                  # Saved GAT weights
├── gnn_model_sage.pt                 # Saved GraphSAGE weights
├── scaler.pkl                        # StandardScaler fit on NSL-KDD features
│
├── data/                             # Place dataset CSVs here (not included — see below)
│   ├── KDDTrain+.txt
│   ├── UNSW_NB15_training-set.csv
│   └── CICIDS2017_sample.csv
│
└── README.md
```

---

## Setup & Installation

### Requirements

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib joblib
```

> PyTorch Geometric may require extra install steps depending on your CUDA version. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Datasets

Download the datasets from their official sources and place them in the `data/` folder:

- **NSL-KDD**: [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)
- **UNSW-NB15**: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **CICIDS2017**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## How to Run

Run notebooks **in order**:

1. **`NSLKDDCoreTraining_fixed.ipynb`** — Performs EDA, trains all three GNN models on NSL-KDD, saves model weights (`.pt`) and scaler (`.pkl`).

2. **`UNSW_Evaluation_fixed.ipynb`** — Loads saved models and evaluates on UNSW-NB15.

3. **`CICIDS_Evaluation_fixed.ipynb`** — Loads saved models and evaluates on CICIDS2017.

---

## Evaluation Metrics

Each model is evaluated on:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

Results are displayed as bar charts comparing GCN, GAT, and GraphSAGE side by side on each dataset.

---

## Key Findings

- All three GNN models achieve strong performance on the training dataset (NSL-KDD).
- Cross-dataset generalization reveals differences in robustness between architectures.
- GAT's attention mechanism shows advantages on datasets with more diverse traffic patterns.
- Sequential graph construction is a practical and consistent approach for flow-based IDS datasets without explicit topology.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-red?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-blue?logo=scikit-learn)

---

## Author

**Christina Barefoot**  
B.S. Cybersecurity and Operations | Mississippi State University  
[LinkedIn](www.linkedin.com/in/christina-barefoot) · [GitHub](https://github.com/Stinafoot)

---

## License

This project is for academic purposes. Datasets are subject to their respective terms of use.
