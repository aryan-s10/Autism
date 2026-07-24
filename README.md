# ASD Screening Prediction using Graph Attention Networks

A patient-similarity graph approach to Autism Spectrum Disorder (ASD) screening, using Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) on the UCI AQ-10 Autism Screening Adult dataset — benchmarked honestly against classical tabular baselines.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyG-GNN-3C2179)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Standard tabular classifiers treat each patient independently. This project instead models patients as nodes in a **similarity graph** — connecting each patient to their *k* most similar peers by behavioral (AQ-10) and demographic features — and trains a **Graph Attention Network (GAT)** to make predictions that are informed by, and interpretable through, relationships between similar patients.

The project also reports an honest, negative-leaning result: on this particular screening-tool-derived label, a simple **Logistic Regression baseline outperforms both GNNs** — and explains *why*, rather than hiding it. See [Results](#results) below.

**Tech stack:** Python, PyTorch, PyTorch Geometric, Scikit-learn, Pandas, NetworkX

---

## Dataset

[UCI Autism Screening Adult Data Set](https://archive.ics.uci.edu/dataset/426/autism+screening+adult) — 704 patients, 21 raw fields including the 10-item AQ-10 screening questionnaire (`A1_Score`–`A10_Score`), age, gender, ethnicity, country of residence, jaundice history, family autism history, and the target `Class/ASD`.

**Class balance:** 515 negative / 189 positive (26.8% positive) — moderately imbalanced, handled throughout via class-weighted loss and stratified splits.

---

## Methodology

### 1–2. Cleaning & Encoding
- Dropped zero-information (`age_desc`) and redundant (`result`, which duplicates the AQ-10 sum) columns.
- Fixed a data-entry error (age = 383) and 2 missing ages via median imputation.
- Recoded `'?'` placeholders in `ethnicity`/`relation` as explicit `"Unknown"`.
- Bucketed `country_of_res` from 67 → 11 categories (top 10 + "Other") to prevent one-hot dimensionality from degrading the similarity metric.
- Standardized continuous features (age) and one-hot encoded categoricals → **42-dimensional feature vector** per patient.

### 3. Patient-Similarity Graph Construction
- Built a k-NN graph (k = 10 default) over the 42-dim feature space using Euclidean distance, symmetrized via edge union.
- Fixed a self-loop edge case where duplicate feature vectors caused unreliable tie-breaking in `sklearn.NearestNeighbors`, by explicitly filtering `node_idx == neighbor_idx`.
- Empirically validated the core assumption: structural homophily = **0.884** (fraction of edges connecting same-label patients) vs. 0.607 expected under random connection — confirming AQ-10 + demographic similarity is a meaningful proxy for diagnostic similarity.

### 4. PyTorch Geometric Data Object
- Transductive node-classification setup: the whole graph is present throughout training; `train_mask` / `val_mask` / `test_mask` (70/15/15, stratified) control which node labels are visible to the loss function.

### 5–6. Models & Training
- **GCN baseline** (2-layer, 722 params) and **GAT** (2-layer, 8-head first layer, 6,022 params).
- Class-weighted `CrossEntropyLoss`, Adam optimizer, early stopping on validation F1.
- Compared against **Logistic Regression** and **Random Forest** on the identical train/test split (no graph at all).

### 7. Interpretability
- Extracted per-edge GAT attention weights (layer 1, averaged across 8 heads) to show which neighboring patients most influenced individual predictions.

### 8. Ablations
- k-sensitivity sweep (k = 3, 5, 10, 15, 20).
- Graph-vs-no-graph: identical GAT architecture, with vs. without relational edges.

---

## Results

### Test-set comparison (identical 70/15/15 split across all models)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.991 | 0.966 | **1.000** | **0.982** | 0.999 |
| Random Forest | 0.934 | **1.000** | 0.750 | 0.857 | 0.987 |
| GCN | 0.915 | 0.880 | 0.786 | 0.830 | 0.974 |
| GAT | 0.896 | 0.793 | 0.821 | 0.807 | 0.962 |

**Key honest finding:** Logistic Regression outperforms both GNNs on raw metrics. The AQ-10 screening tool's label is, by design, close to a **linear threshold function of the 10 item scores** — making the task near-linearly-separable and ideally suited to a simple linear model. This is a property of *this specific screening-tool-derived label*, not a general failure of graph-based methods.

### Ablation: does the graph structure itself help?

| Setup | F1 | Precision | Recall |
|---|---|---|---|
| GAT, no graph (self-attention only) | 0.722 | 0.591 | 0.929 |
| GAT, with k=10 similarity graph | 0.828 | 0.800 | 0.857 |

Using the **identical architecture and parameter count**, adding the k-NN relational graph improved F1 by **+0.105**, driven primarily by a large precision gain (+0.209) — isolating the graph structure's contribution from raw model capacity.

### Ablation: k sensitivity

| k | F1 | Precision | Recall |
|---|---|---|---|
| 3 | 0.867 | 0.813 | 0.929 |
| 5 | 0.867 | 0.813 | 0.929 |
| 10 | 0.828 | 0.800 | 0.857 |
| 15 | 0.847 | 0.806 | 0.893 |
| 20 | 0.877 | 0.862 | 0.893 |

Performance is reasonably stable across k (0.828–0.877 F1). Note this used a single random seed per k, so averaging multiple seeds would give a more robust picture of true k-sensitivity vs. training noise.

### Interpretability example

A misclassified test patient (true label: ASD-negative, AQ-10 = 6/10, predicted ASD-positive at 87.7% confidence) had 3 of its top-4 attended neighbors as ASD-positive patients with similar-or-higher AQ-10 scores — directly showing *why* the model erred, a clinically legible failure mode rather than an opaque one. See `Output/attention_explanations.png`.

Globally, attention mass tracked the underlying graph's structural homophily proportionally (87.4% of attention mass on same-label edges vs. 88.4% of edge share), rather than dramatically amplifying it.

---

## Project Structure

```
Autism/
├── 00_eda.ipynb                  Exploratory data analysis
├── 01_clean_data.ipynb           Phase 1: data cleaning
├── 02_encode_features.ipynb      Phase 2: feature encoding
├── 03_build_graph.ipynb          Phase 3: k-NN similarity graph construction
├── 04_build_pyg_data.ipynb       Phase 4: PyG Data object + train/val/test masks
├── 05_models.py / .ipynb         Phase 5: GCN and GAT architectures
├── 06_train_evaluate.ipynb       Phase 6: training, evaluation, tabular baselines
├── 07_interpretability.ipynb     Phase 7: GAT attention-based interpretability
├── 08_ablation.ipynb             Phase 8: k-sensitivity and graph-vs-no-graph ablations
├── PROJECT_REPORT.md             Full write-up
│
├── Data/
│   ├── autism_screening.csv                          Raw dataset
│   ├── Autism-Adult-Data.arff                         Raw dataset (ARFF format)
│   ├── Autism-Screening-Adult-Data Description.docx   Feature descriptions
│   ├── data_clean.csv                                 Cleaned dataset
│   ├── X_features.csv, y_labels.csv                   Encoded features / labels
│
└── Output/
    ├── graph_data.pt                        Final PyG Data object
    ├── edge_index.npy                       k-NN graph edges (k=10)
    ├── gcn_weights.pt, gat_weights.pt        Trained model weights
    ├── attention_artifacts.pt               Extracted attention weights
    ├── model_comparison.csv                 Test-set metrics across all models
    ├── ablation_k_sensitivity.csv           k-sweep results
    ├── ablation_no_graph.csv                Graph-vs-no-graph results
    ├── degree_distribution.png              Phase 3 diagnostic
    ├── graph_sample_visualization.png       Phase 3 diagnostic
    ├── training_curves.png                  Phase 6 training curves
    ├── attention_explanations.png           Phase 7 interpretability plots
    └── ablation_plot.png                    Phase 8 ablation results
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/aryan-s10/Autism-Spectrum-Detection.git
cd Autism-Spectrum-Detection
pip install -r requirements.txt
```

`requirements.txt`:
```
torch
torch_geometric
scikit-learn
pandas
numpy
networkx
matplotlib
scipy
jupyter
```

> PyTorch Geometric can be sensitive to your PyTorch/CUDA version — if `pip install torch_geometric` fails, follow the [official install matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your setup.

### Running the pipeline

Run the notebooks in order (each phase writes intermediate artifacts consumed by the next):

```bash
jupyter notebook
```

1. `00_eda.ipynb` — explore the raw data
2. `01_clean_data.ipynb` → `Data/data_clean.csv`
3. `02_encode_features.ipynb` → `Data/X_features.csv`, `Data/y_labels.csv`
4. `03_build_graph.ipynb` → `Output/edge_index.npy`
5. `04_build_pyg_data.ipynb` → `Output/graph_data.pt`
6. `05_models.ipynb` — defines `GCN` / `GAT` (also available as plain `05_models.py` for import)
7. `06_train_evaluate.ipynb` — trains all models, produces `Output/model_comparison.csv`
8. `07_interpretability.ipynb` — attention-based explanations
9. `08_ablation.ipynb` — k-sensitivity and graph-vs-no-graph ablations

---

## Limitations & Future Work

- **Linear separability of the label** — future work should test on labels derived from full clinical diagnosis (e.g. DSM-5 criteria) rather than AQ-10 screening-tool output, where relationships between patients may carry genuinely non-linear signal beyond raw item scores.
- **Single-seed ablations** — k-sensitivity and graph-vs-no-graph results would benefit from averaging across multiple random seeds to separate true effects from training variance.
- **Euclidean similarity on mixed data** — a Gower distance or learned similarity metric could better handle the mix of binary/categorical/continuous features than raw Euclidean distance on one-hot encodings.
- **Natural extension** — a knowledge-graph-based approach (symptom–disease–drug relations, e.g. via Hetionet/DRKG with R-GCN or GAT) could model richer, non-patient-similarity relational structure where linear separability is less likely to dominate results.

---

## Disclaimer

This project is a research/educational exploration of graph-based ML methods applied to a screening-tool dataset. It is **not** a diagnostic tool and is not intended for clinical use.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [UCI Machine Learning Repository — Autism Screening Adult Data Set](https://archive.ics.uci.edu/dataset/426/autism+screening+adult)
- Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
