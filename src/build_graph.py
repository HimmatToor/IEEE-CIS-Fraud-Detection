import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

IDENTIFIER_COLUMNS = [
    "card1", "card2", "card3", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceInfo",
]

# DeviceType only has 2 distinct values (desktop/mobile), so as a graph edge it just
# creates two ~140k-degree hub nodes that mean-aggregation drowns everything else into.
# Not useless as a signal though, so keep it as a direct feature instead of an edge.

TRANSACTION_NUMERIC_COLUMNS = (
    ["TransactionAmt"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in range(1, 340)]
)

TRANSACTION_CATEGORICAL_COLUMNS = ["ProductCD", "DeviceType"] + [f"M{i}" for i in range(1, 10)]


def load_raw_tables(data_dir: Path, split: str) -> pd.DataFrame:
    transactions = pd.read_csv(data_dir / f"{split}_transaction.csv")
    identity = pd.read_csv(data_dir / f"{split}_identity.csv")
    df = transactions.merge(identity, on="TransactionID", how="left")
    return df.reset_index(drop=True)


def build_transaction_features(df: pd.DataFrame) -> torch.Tensor:
    numeric = df[TRANSACTION_NUMERIC_COLUMNS].astype(float)
    numeric = numeric.fillna(numeric.median())

    categorical = pd.DataFrame(
        {
            column: df[column].astype("category").cat.codes.astype(float)
            for column in TRANSACTION_CATEGORICAL_COLUMNS
        }
    )

    features = pd.concat([numeric, categorical], axis=1)
    values = features.to_numpy()
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    values = (values - mean) / std

    return torch.tensor(values, dtype=torch.float)


def build_hetero_graph(df: pd.DataFrame) -> HeteroData:
    data = HeteroData()
    row_idx = np.arange(len(df))

    data["transaction"].x = build_transaction_features(df)
    data["transaction"].transaction_id = torch.tensor(df["TransactionID"].to_numpy(), dtype=torch.long)
    if "isFraud" in df.columns:
        data["transaction"].y = torch.tensor(df["isFraud"].to_numpy(), dtype=torch.long)

    for column in IDENTIFIER_COLUMNS:
        if column not in df.columns:
            continue

        known_mask = df[column].notna().to_numpy()
        if not known_mask.any():
            continue

        # route transactions through the shared value instead of connecting them
        # directly to each other, or a popular value turns into a huge clique
        unique_values, value_node_ids = np.unique(
            df[column][known_mask].astype(str), return_inverse=True
        )

        data[column].num_nodes = len(unique_values)
        data[column].value = list(unique_values)

        src = torch.tensor(row_idx[known_mask], dtype=torch.long)
        dst = torch.tensor(value_node_ids, dtype=torch.long)
        data["transaction", f"has_{column}", column].edge_index = torch.stack([src, dst])

    return T.ToUndirected()(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("ieee-fraud-detection-data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or Path("data/processed") / f"{args.split}_graph.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_tables(args.data_dir, args.split)
    graph = build_hetero_graph(df)
    graph.validate(raise_on_error=True)

    torch.save(graph, out_path)
    print(graph)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
