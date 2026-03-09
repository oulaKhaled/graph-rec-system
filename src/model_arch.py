import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, users: torch.Tensor, series: torch.Tensor, data) -> torch.Tensor:
        # triplet mode (training)
        if hasattr(data["users"], "src_index"):
            src = users[data["users"].src_index]
            pos_dst = series[data["series"].dst_pos_index]
            neg_dst = series[data["series"].dst_neg_index]

            pos_score = (src * pos_dst).sum(dim=-1)
            neg_score = (src * neg_dst).sum(dim=-1)
            return pos_score, neg_score
        # binary mode (validation/test)
        else:
            edge_label_index = data["users", "rate", "series"].edge_label_index
            src = users[edge_label_index[0]]
            dst = series[edge_label_index[1]]
            return (src * dst).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.user_lin = torch.nn.Linear(384, hidden_channels)
        self.series_lin = torch.nn.Linear(384, hidden_channels)
        self.genres_emb = torch.nn.Embedding(data["genres"].num_nodes, hidden_channels)
        self.writer_emb = torch.nn.Embedding(data["writer"].num_nodes, hidden_channels)
        self.type_emb = torch.nn.Embedding(data["type"].num_nodes, hidden_channels)

        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def get_embedding(self, data: HeteroData) -> torch.Tensor:
        device = next(self.parameters()).device
        x_dict = {
            "users": self.user_lin(data["users"].reviews),
            "series": self.series_lin(data["series"].overview),
            "genres": self.genres_emb(
                torch.arange(data["genres"].num_nodes, device=device)
            ),
            "writer": self.writer_emb(
                torch.arange(data["writer"].num_nodes, device=device)
            ),
            "type": self.type_emb(torch.arange(data["type"].num_nodes, device=device)),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        return x_dict

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = self.get_embedding(data)
        pred = self.classifier(x_dict["users"], x_dict["series"], data)
        return pred
