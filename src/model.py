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


def get_recommendation(username: str, ratings_dict: Dict[str, int]):
    data, user_index = get_or_create_user(username, ratings_dict)
    print("Data", data)
    print("user_index", user_index)

    with torch.no_grad():
        x_dict = gnn_model.get_embedding(data.to(device))
        user_emb = x_dict["users"][user_index]
        scores = torch.sigmoid((user_emb * x_dict["series"]).sum(dim=-1))

        # exclude ALL rated series
        for sid in list(ratings_dict.keys()):
            scores[sid] = -1
        ## recommend it series indices
        top_k = scores.topk(5).indices.cpu().numpy()
    return top_k
