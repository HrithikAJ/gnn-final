# GNN Evaluation Report

## Comparison Table: Graph-Statistical Baseline vs GNN vs Temporal Forecaster

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Lead Time (min) |
|---|---|---|---|---|---|---|---|
| Baseline (Random Forest) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| GraphSAGE GNN (Snapshot) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| Temporal GNN Forecaster | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |

## Interpretation
- **Single-Snapshot GNN**: Encodes topological structural context via neighborhood aggregation (GraphSAGE).
- **Temporal GNN Forecaster**: Aggregates sequences of 30 historical graph embeddings to forecast risk over a 10-minute horizon.
- **Forecast Lead Time**: Produces advance warning alerts with up to 0.0 minutes of lead time before attack onset.
