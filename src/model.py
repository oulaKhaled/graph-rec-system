# import pickle
# import torch
# from pathlib import Path
# from torch_geometric.data import HeteroData

# MODEL_PATH = Path("model\gnn_model112.pth")


# def load_model():
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     model.eval()
#     return model


# def get_recommendation(
#     user_index: int, x_dict, data: HeteroData, series_details_df, k=5
# ):
#     pass
