"""
Inference module for GNN cyber attack risk forecasting.
Provides a clean, modular API for downstream services (e.g. FastAPI backend, LSTM integration).

Output Schema:
{
    "risk_score": float,      # 0.0 - 1.0 probability/risk score
    "prediction": int,        # binary forecast (0 or 1)
    "embedding": list         # graph-level representation vector
}
"""
import os
import sys
import json
import argparse
from typing import List, Dict, Union, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from gnn.models.gnn import AttackRiskGNN, TemporalAttackGNN
from gnn.graph_builder import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES


class GNNPredictor:
    """
    Production inference class for the GNN Attack Risk Forecasting model.
    Loads model checkpoint and executes deterministic forward pass without modifying weights.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto", threshold: float = 0.5):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint file containing state_dict + metadata
            device: 'cuda', 'cpu', or 'auto'
            threshold: Decision threshold for binary prediction (default: 0.5)
        """
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.metadata = self.checkpoint.get("metadata", {})
        self.model_config = self.checkpoint.get("model_config", {})
        self.is_temporal = self.checkpoint.get("is_temporal", False)

        in_channels = self.model_config.get("in_channels", len(NODE_FEATURE_NAMES))
        hidden_channels = self.model_config.get("hidden_channels", 64)
        num_layers = self.model_config.get("num_layers", 2)
        dropout = self.model_config.get("dropout", 0.3)
        pooling = self.model_config.get("pooling", "mean_max")

        if self.is_temporal:
            temporal_method = self.model_config.get("temporal_method", "mean")
            self.model = TemporalAttackGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                temporal_method=temporal_method,
            )
        else:
            self.model = AttackRiskGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
            )

        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, graph_input: Union[Data, List[Data]]) -> Dict[str, Any]:
        """
        Run inference on a single graph snapshot or a sequence of graph snapshots.

        Args:
            graph_input: Single PyG Data object or List of PyG Data objects

        Returns:
            Dict conforming to the specified output schema:
            {
                "risk_score": float,
                "prediction": int,
                "embedding": List[float]
            }
        """
        if isinstance(graph_input, list):
            # Sequence of graphs
            if self.is_temporal:
                seq_device = [g.to(self.device) for g in graph_input]
                temporal_emb = self.model.get_sequence_embedding(seq_device)
                logits = self.model.temporal_classifier(temporal_emb)
                risk_score = float(torch.sigmoid(logits).item())
                emb_list = temporal_emb.cpu().squeeze().tolist()
                if not isinstance(emb_list, list):
                    emb_list = [emb_list]
            else:
                # Evaluate on the latest graph in sequence
                latest_graph = graph_input[-1].to(self.device)
                batch_vec = torch.zeros(latest_graph.num_nodes, dtype=torch.long, device=self.device)
                emb = self.model.get_graph_embedding(latest_graph.x, latest_graph.edge_index, batch_vec)
                logits = self.model.classifier(emb)
                risk_score = float(torch.sigmoid(logits).item())
                emb_list = emb.cpu().squeeze().tolist()
                if not isinstance(emb_list, list):
                    emb_list = [emb_list]
        else:
            # Single graph
            single_graph = graph_input.to(self.device)
            batch_vec = torch.zeros(single_graph.num_nodes, dtype=torch.long, device=self.device)
            if self.is_temporal:
                seq_device = [single_graph]
                temporal_emb = self.model.get_sequence_embedding(seq_device)
                logits = self.model.temporal_classifier(temporal_emb)
                risk_score = float(torch.sigmoid(logits).item())
                emb_list = temporal_emb.cpu().squeeze().tolist()
                if not isinstance(emb_list, list):
                    emb_list = [emb_list]
            else:
                emb = self.model.get_graph_embedding(single_graph.x, single_graph.edge_index, batch_vec)
                logits = self.model.classifier(emb)
                risk_score = float(torch.sigmoid(logits).item())
                emb_list = emb.cpu().squeeze().tolist()
                if not isinstance(emb_list, list):
                    emb_list = [emb_list]

        prediction = 1 if risk_score >= self.threshold else 0

        return {
            "risk_score": float(np.round(risk_score, 6)),
            "prediction": int(prediction),
            "embedding": [float(np.round(v, 6)) for v in emb_list],
        }


def run_cli():
    """Command-line inference entry point."""
    parser = argparse.ArgumentParser(description="GNN Cyber Attack Risk Predictor")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_gnn.pt checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cuda, cpu, auto)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = GNNPredictor(args.checkpoint, device=args.device, threshold=args.threshold)

    # Generate synthetic dummy graph for verification
    dummy_data = Data(
        x=torch.randn(10, len(NODE_FEATURE_NAMES), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
        edge_attr=torch.randn(5, len(EDGE_FEATURE_NAMES), dtype=torch.float32),
        num_nodes=10,
    )

    result = predictor.predict(dummy_data)
    print("\n--- Inference Output ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_cli()
