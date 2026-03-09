import torch
import tqdm
from app import device
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
from torch_geometric import transforms as T
import torch.nn.functional as F


model = ""


def create_embedding(updated_graph, new_user_index, rated_series_id):
    model.eval()  # no training
    with torch.no_grad():  # no gradients
        x_dict = model.get_embedding(updated_graph)  # just forward pass
        new_user_emb = x_dict["users"][new_user_index]
        scores = torch.sigmoid((new_user_emb * x_dict["series"]).sum(dim=-1))
        scores[rated_series_id] = -1  # exclude already rated
        top_k = scores.topk(5).indices.cpu().numpy()
        return top_k
