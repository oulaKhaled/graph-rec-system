import pickle
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from src.preprocess import get_or_create_user, load_data, load_models
import numpy
from typing import Dict

# MODEL_PATH = Path("model\gnn_model112.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gnn_model, encode_model = load_models()
data = load_data()


def get_recommendation(
    user_index: int, x_dict, data: HeteroData, username, ratings: Dict[str, int], k=5
):
    data, user_index = get_or_create_user(
        username, list(ratings.keys()), list(ratings.values()), data, encode_model
    )
    ## save new HeteroGraph
    gnn_model.eval()
    with torch.no_grad():
        x_dict = gnn_model.get_embedding(data.to(device))
        user_emb = x_dict["users"][user_index]
        scores = torch.sigmoid((user_emb * x_dict["series"]).sum(dim=-1))

        # exclude ALL rated series
        for sid in list(ratings.keys()):
            scores[sid] = -1
        ## recommend it series indices
        top_k = scores.topk(5).indices.cpu().numpy()
    return top_k
