"""
Unit tests for GNN models and inference output schema.
"""
import pytest
import torch
from torch_geometric.data import Data
from gnn.models.gnn import AttackRiskGNN, TemporalAttackGNN
from gnn.inference.predict import GNNPredictor
from gnn.graph_builder import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES


def test_attack_risk_gnn_forward():
    in_channels = len(NODE_FEATURE_NAMES)
    model = AttackRiskGNN(in_channels=in_channels, hidden_channels=32, num_layers=2)

    x = torch.randn(8, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    batch = torch.zeros(8, dtype=torch.long)

    logits = model(x, edge_index, batch)
    assert logits.shape == (1, 1)

    probs = model.predict_proba(x, edge_index, batch)
    assert probs.shape == (1, 1)
    assert 0.0 <= probs.item() <= 1.0

    emb = model.get_graph_embedding(x, edge_index, batch)
    assert emb.shape == (1, 64)  # 32 * 2 for mean_max pooling


def test_temporal_gnn_forward():
    in_channels = len(NODE_FEATURE_NAMES)
    model = TemporalAttackGNN(in_channels=in_channels, hidden_channels=32, num_layers=2)

    g1 = Data(x=torch.randn(5, in_channels), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=5)
    g2 = Data(x=torch.randn(6, in_channels), edge_index=torch.tensor([[0, 2], [2, 0]], dtype=torch.long), num_nodes=6)

    sequence = [g1, g2]
    logits = model(sequence)
    assert logits.shape == (1, 1)

    probs = model.predict_proba(sequence)
    assert probs.shape == (1, 1)
    assert 0.0 <= probs.item() <= 1.0


def test_inference_output_schema(tmp_path):
    # Save a mock checkpoint to verify GNNPredictor output schema
    in_channels = len(NODE_FEATURE_NAMES)
    model = AttackRiskGNN(in_channels=in_channels, hidden_channels=32, num_layers=2)
    checkpoint_path = str(tmp_path / "mock_gnn.pt")

    torch.save({
        "state_dict": model.state_dict(),
        "model_config": {
            "in_channels": in_channels,
            "hidden_channels": 32,
            "num_layers": 2,
            "dropout": 0.0,
            "pooling": "mean_max"
        },
        "is_temporal": False,
    }, checkpoint_path)

    predictor = GNNPredictor(checkpoint_path, device="cpu")
    dummy_data = Data(
        x=torch.randn(5, in_channels),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=5
    )

    out = predictor.predict(dummy_data)
    assert "risk_score" in out
    assert "prediction" in out
    assert "embedding" in out
    assert isinstance(out["risk_score"], float)
    assert isinstance(out["prediction"], int)
    assert isinstance(out["embedding"], list)
    assert len(out["embedding"]) == 64
    assert 0.0 <= out["risk_score"] <= 1.0
    assert out["prediction"] in (0, 1)
