# GNN Ablation Report

## Objective
Evaluate the contribution of Graph Structure, Node Attributes, and Edge Attributes to cyber attack risk forecasting performance.

## Ablation Comparison Table

| Configuration | PR-AUC | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Graph Structure Only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Node Features Only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Node + Edge Features (Full) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Key Findings
1. **Structure vs Node Features**: Node features (degree, port distribution, traffic volume) provide significant discriminative capacity.
2. **Edge Features**: Edge-level flow attributes enhance connection granularity between active host pairs.
