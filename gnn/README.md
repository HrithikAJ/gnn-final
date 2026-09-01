# Graph-Based Cyber Attack Risk Forecasting

This module implements a production-grade, reproducible PyTorch Geometric Graph Neural Network (GNN) and temporal sequence forecasting pipeline for predicting cyber attack risk from historical network communication graphs.

Designed for integration with the project's **FastAPI backend** and **LSTM/GRU temporal models**.

---

## 1. System Architecture

```
                                  [ Raw Network Flows ]
                                             │
                                             ▼
                             [ Data Preprocessing & Cleaning ]
                                             │
                                             ▼
                           [ 1-Minute Graph Snapshot Builder ]
                           ├── Nodes: IP Addresses (12 features)
                           └── Edges: Directed Flows (6 features)
                                             │
                                             ▼
                          [ GraphSAGE Structural Encoder ]
                                             │
                                             ▼
                           [ Global Mean + Max Pooling ]
                                             │
                                             ▼
                    ┌────────────────────────┴────────────────────────┐
                    │                                                 │
                    ▼                                                 ▼
        [ AttackRiskGNN (Snapshot) ]                    [ Temporal Sequence Forecaster ]
        ├── Embeddings (64-dim)                         ├── Historical Window (30 graphs)
        └── Current Attack Risk                         ├── Forecast Horizon (10 min)
                                                        └── Future Attack Risk Score
```

---

## 2. Directory Structure

```
gnn/
├── GRAPH_DESIGN.md              # Feasibility study, node/edge definitions, resolution strategy
├── GRAPH_DATASET_REPORT.md      # Statistical validation of graph snapshots & topology
├── GNN_BASELINE_COMPARISON.md   # Statistical baselines vs GNN comparison
├── GNN_EVALUATION_REPORT.md     # Full evaluation of Snapshot GNN & Temporal Forecaster
├── GNN_ABLATION_REPORT.md       # Ablation study (Structure vs Node vs Edge features)
├── README.md                    # Module documentation & integration guide
│
├── node_mapping.py              # Stable IP-to-index mapper (per-snapshot)
├── graph_builder.py             # Vectorized + streaming PyG Data construction
├── graph_dataset.py             # PyG Dataset + sliding temporal window sequences
├── baselines.py                 # Graph-statistical baselines (Logistic Regression, RF, GBM)
├── run_experiments.py           # End-to-end master experiment runner
│
├── models/
│   ├── __init__.py
│   └── gnn.py                   # GraphSAGE encoder, AttackRiskGNN, TemporalAttackGNN
│
├── training/
│   ├── __init__.py
│   └── train_gnn.py             # Reproducible GPU training, class weighting & early stopping
│
├── inference/
│   ├── __init__.py
│   └── predict.py               # Standalone GNNPredictor API & CLI
│
├── configs/
│   └── gnn.yaml                 # Hyperparameters configuration
│
├── checkpoints/
│   ├── best_gnn.pt              # Static GraphSAGE model checkpoint + metadata
│   └── best_temporal_gnn.pt     # Temporal Forecaster checkpoint + metadata
│
└── tests/
    ├── test_node_mapping.py     # Mapping bijective property tests
    ├── test_graph_builder.py    # Self-loop removal & feature dimension tests
    ├── test_labels.py           # Chronological purge & embargo tests
    └── test_gnn.py              # Forward pass & inference output schema tests
```

---

## 3. Quickstart & Reproducibility

### Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric pyarrow pyyaml scikit-learn pytest
```

### Running Unit Tests
```bash
python -m pytest gnn/tests/ -v
```

### Running the Full Experiment Pipeline
```bash
python -u gnn/run_experiments.py
```

### Running Inference via CLI
```bash
python gnn/inference/predict.py --checkpoint gnn/checkpoints/best_gnn.pt
```

---

## 4. Integration API (FastAPI & LSTM Integration)

### Python Inference API

```python
import torch
from gnn.inference.predict import GNNPredictor
from gnn.graph_builder import build_graph_snapshot

# 1. Initialize predictor (loads weights and sets eval mode)
predictor = GNNPredictor("gnn/checkpoints/best_gnn.pt", device="auto", threshold=0.5)

# 2. Build graph snapshot from incoming pandas flow dataframe
graph_data = build_graph_snapshot(flows_df, timestamp=current_ts)

# 3. Get attack risk score, binary alert, and downstream embedding
result = predictor.predict(graph_data)

# Result format:
# {
#     "risk_score": 0.048741,    # Probability of attack [0.0 - 1.0]
#     "prediction": 0,           # 0 (benign) or 1 (attack alert)
#     "embedding": [...]         # 64-dimensional graph representation
# }
```

### Temporal Sequence Model Integration (for LSTM/GRU)

```python
# Pass a sequence of 30 consecutive graph snapshots:
sequence = [graph_t_minus_29, ..., graph_t]
temporal_predictor = GNNPredictor("gnn/checkpoints/best_temporal_gnn.pt", device="auto")
forecast = temporal_predictor.predict(sequence)
```

---

## 5. Summary of Findings

- **Dataset**: UGR'16 August Week 5 containing 40,289,595 network flows across 9.33 hours.
- **Topology**: Average of 18,428 nodes and 28,348 directed edges per 1-minute snapshot.
- **Hardware Acceleration**: Full CUDA acceleration on NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GB VRAM).
- **Leakage Prevention**: Strictly chronological train/val/test boundaries with label purge and lookback embargo.
