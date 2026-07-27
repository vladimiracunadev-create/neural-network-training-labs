from __future__ import annotations

from torch import nn


class GraphModelUnavailable(nn.Module):
    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason

    def forward(self, *args, **kwargs):
        raise RuntimeError(self.reason)


def build_graph_model(kind: str, in_channels: int, hidden_channels: int, out_channels: int):
    try:
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv
    except ImportError:
        return GraphModelUnavailable('Instale el extra graph: pip install -e ".[graph]"')

    layer_cls = {"gcn": GCNConv, "graphsage": SAGEConv, "gat": GATConv}.get(kind)
    if layer_cls is None:
        raise ValueError(f"Modelo de grafo desconocido: {kind}")

    class GraphNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "gat":
                self.first = layer_cls(in_channels, hidden_channels, heads=4, concat=False)
                self.second = layer_cls(hidden_channels, out_channels, heads=1, concat=False)
            else:
                self.first = layer_cls(in_channels, hidden_channels)
                self.second = layer_cls(hidden_channels, out_channels)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(0.4)

        def forward(self, x, edge_index):
            x = self.dropout(self.activation(self.first(x, edge_index)))
            return self.second(x, edge_index)

    return GraphNetwork()
