from pathlib import Path

import pytest
import torch

from build_graph import IDENTIFIER_COLUMNS, build_hetero_graph, load_raw_tables

DATA_DIR = Path(__file__).resolve().parents[1] / "ieee-fraud-detection-data"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "train_transaction.csv").exists(),
    reason="raw IEEE-CIS data not present",
)


@pytest.fixture(scope="module")
def train_graph():
    df = load_raw_tables(DATA_DIR, "train")
    return build_hetero_graph(df)


def test_graph_is_valid(train_graph):
    train_graph.validate(raise_on_error=True)


def test_transaction_nodes_have_features_and_labels(train_graph):
    x = train_graph["transaction"].x
    y = train_graph["transaction"].y
    assert x.shape[0] == y.shape[0] > 0
    assert not torch.isnan(x).any()


def test_identifier_node_types_are_populated(train_graph):
    for column in IDENTIFIER_COLUMNS:
        assert train_graph[column].num_nodes > 0


def test_edges_stay_within_node_bounds(train_graph):
    for edge_type in train_graph.edge_types:
        src_type, _, dst_type = edge_type
        src, dst = train_graph[edge_type].edge_index
        assert src.max() < train_graph[src_type].num_nodes
        assert dst.max() < train_graph[dst_type].num_nodes


def test_shared_card1_correlates_with_fraud_rate(train_graph):
    # if transactions sharing a card all sat at the base fraud rate, the graph
    # wouldn't be adding anything a row-independent model couldn't already see
    y = train_graph["transaction"].y.float()
    base_rate = y.mean().item()

    src, dst = train_graph["transaction", "has_card1", "card1"].edge_index
    num_nodes = train_graph["card1"].num_nodes

    fraud_sum = torch.zeros(num_nodes).scatter_add_(0, dst, y[src])
    count = torch.zeros(num_nodes).scatter_add_(0, dst, torch.ones_like(y[src]))
    group_rate = fraud_sum / count

    well_sampled = count >= 20
    assert group_rate[well_sampled].max().item() > base_rate * 3
