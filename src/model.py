import pickle
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from .preprocess import get_or_create_user
import numpy
from sentence_transformers import SentenceTransforme
from typing import Dict

# MODEL_PATH = Path("model\gnn_model112.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    gnn_model = torch.load("model/gnn_model112.pth", map_location="cpu")
    enocode_model = SentenceTransforme(f"model\SentenceTrans_model")
    return gnn_model, enocode_model


def load_data():
    hetero_data = torch.load("data/hetero_graph3.pt")
    return hetero_data


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
