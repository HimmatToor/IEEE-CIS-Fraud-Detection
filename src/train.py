import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from build_graph import load_raw_tables

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
VAL_FRACTION = 0.15
PATIENCE = 10
MAX_EPOCHS = 150


class HeteroGraphSAGE(nn.Module):
    def __init__(
        self,
        graph: HeteroData,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.transaction_proj = nn.Linear(graph["transaction"].x.size(1), hidden_dim)
        self.embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(graph[node_type].num_nodes, hidden_dim)
                for node_type in graph.node_types
                if node_type != "transaction"
            }
        )

        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {edge_type: SAGEConv(hidden_dim, hidden_dim) for edge_type in graph.edge_types},
                    aggr="mean",
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, graph: HeteroData) -> torch.Tensor:
        x_dict = {"transaction": self.transaction_proj(graph["transaction"].x)}
        for node_type, embedding in self.embeddings.items():
            x_dict[node_type] = embedding.weight

        edge_index_dict = {edge_type: graph[edge_type].edge_index for edge_type in graph.edge_types}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {node_type: self.dropout(F.relu(x)) for node_type, x in x_dict.items()}

        return self.classifier(x_dict["transaction"]).squeeze(-1)


def time_based_split(df, val_fraction: float = VAL_FRACTION) -> tuple[torch.Tensor, torch.Tensor]:
    # splitting on TransactionDT rather than randomly so validation looks like predicting
    # forward in time, not interpolating between transactions the model has already seen
    order = df["TransactionDT"].to_numpy().argsort()

    split_point = int(len(order) * (1 - val_fraction))
    train_idx = torch.tensor(order[:split_point], dtype=torch.long)
    val_idx = torch.tensor(order[split_point:], dtype=torch.long)
    return train_idx, val_idx


def train(
    graph: HeteroData,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> None:
    y = graph["transaction"].y.float()

    num_pos = y[train_idx].sum()
    num_neg = len(train_idx) - num_pos
    pos_weight = num_neg / num_pos

    model = HeteroGraphSAGE(graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(graph)[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(graph))
            train_auc = roc_auc_score(y[train_idx], probs[train_idx])
            val_auc = roc_auc_score(y[val_idx], probs[val_idx])

        marker = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1

        print(f"epoch {epoch:03d}  loss {loss.item():.4f}  train_auc {train_auc:.4f}  val_auc {val_auc:.4f}{marker}")

        if epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch}, best val_auc {best_val_auc:.4f}")
            break

    torch.save(best_state, Path("models") / "graphsage.pt")
    print(f"saved best checkpoint (val_auc {best_val_auc:.4f}) to models/graphsage.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=Path("data/processed/train_graph.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("ieee-fraud-detection-data"))
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    graph = torch.load(args.graph, weights_only=False)
    df = load_raw_tables(args.data_dir, "train")
    train_idx, val_idx = time_based_split(df)
    train(graph, train_idx, val_idx, args.epochs, args.patience)


if __name__ == "__main__":
    main()
