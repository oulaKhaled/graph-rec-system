import torch
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer

path = ""


# encode_mode = SentenceTransformer(f"model\SentenceTrans_model")


## create a function that will retrun nodes count to observe how many new users comes in

# {"ratings": {"1396": 8, "66732": 9, "1399": 7}}


def load_models():
    gnn_model = torch.load("model/gnn_model112.pth", map_location="cpu")
    enocode_model = SentenceTransformer(f"model/SentenceTrans_model")

    return gnn_model, enocode_model


def load_data():
    hetero_data = torch.load("data/hetero_graph3.pt")
    series_details_df = pd.read_csv("data/series_details_df1.csv")
    df_reviews_rate_exist = pd.read_csv("data/df_reviews_rate_exist11.csv")

    return hetero_data, series_details_df, df_reviews_rate_exist


hetero_data, series_details_df, df_reviews_rate_exist = load_data()
gnn_model, encode_model = load_models()


def get_or_create_user(username, ratings_dict):

    with open("data/user_registry.json", "r") as f:
        user_registry = json.load(f)

    if username in user_registry:
        # returning user
        user_index = user_registry[username]
        data = add_new_interaction(user_index, ratings_dict)

    else:
        # new user
        data, user_index = add_new_user(ratings_dict)

        user_registry[username] = user_index

        with open(f"{path}user_registry.json", "w") as f:
            json.dump(user_registry, f)

        print(f"New user created! User index: {user_index}")

    return data, user_index


def add_new_interaction(user_index, ratings_dict):
    # ratings_dict = {"series_id": rating, ...}
    # convert string keys to int
    ratings_dict = {int(k): v for k, v in ratings_dict.items()}
    existing_edges = hetero_data["users", "rate", "series"].edge_index

    for series_id, rating in ratings_dict.items():

        # check if edge already exists
        mask = (existing_edges[0] == user_index) & (existing_edges[1] == series_id)

        if mask.any():
            # Case 1 & 2 — update existing edge rating
            hetero_data["users", "rate", "series"].edge_attr[mask] = torch.tensor(
                rating, dtype=torch.float
            )

            # update reverse edge attr too
            rev_existing = hetero_data["series", "rev_rate", "users"].edge_index
            rev_mask = (rev_existing[0] == series_id) & (rev_existing[1] == user_index)
            if rev_mask.any():
                hetero_data["series", "rev_rate", "users"].edge_attr[rev_mask] = (
                    torch.tensor(rating, dtype=torch.float)
                )

            print(f"Updated edge: user_{user_index} → series_{series_id} = {rating}")

        else:
            # Case 3 & 4 — create new edge
            new_edge = torch.tensor([[user_index], [series_id]], dtype=torch.long)
            hetero_data["users", "rate", "series"].edge_index = torch.cat(
                [existing_edges, new_edge], dim=1
            )

            hetero_data["users", "rate", "series"].edge_attr = torch.cat(
                [
                    hetero_data["users", "rate", "series"].edge_attr,
                    torch.tensor([rating], dtype=torch.float),
                ],
                dim=0,
            )

            # reverse edge
            new_rev_edge = torch.tensor([[series_id], [user_index]], dtype=torch.long)
            hetero_data["series", "rev_rate", "users"].edge_index = torch.cat(
                [hetero_data["series", "rev_rate", "users"].edge_index, new_rev_edge],
                dim=1,
            )

            hetero_data["series", "rev_rate", "users"].edge_attr = torch.cat(
                [
                    hetero_data["series", "rev_rate", "users"].edge_attr,
                    torch.tensor([rating], dtype=torch.float),
                ],
                dim=0,
            )

            # update existing_edges reference for next iteration
            existing_edges = hetero_data["users", "rate", "series"].edge_index

            print(f"Created edge: user_{user_index} → series_{series_id} = {rating}")

    return hetero_data


def add_new_user(
    ratings_dict,
):
    ratings_dict = {int(k): v for k, v in ratings_dict.items()}

    rated_series_ids = list(ratings_dict.keys())

    # use FIRST series overview as user review proxy
    first_series_id = rated_series_ids[0]
    series_overview = series_details_df[series_details_df["index"] == first_series_id][
        "overview"
    ].tolist()

    embedded_overview = encode_model.encode(series_overview)
    mean_review = torch.tensor(
        embedded_overview.mean(axis=0), dtype=torch.float
    ).unsqueeze(0)

    user_index = int(hetero_data["users"].x.max().item()) + 1

    # global avg hour
    global_avg_hour = torch.tensor(
        [df_reviews_rate_exist.groupby("author")["hour"].mean().mean()],
        dtype=torch.float,
    )

    # add user features
    hetero_data["users"].x = torch.cat(
        [hetero_data["users"].x, torch.tensor([user_index], dtype=torch.long)], dim=0
    )

    hetero_data["users"].reviews = torch.cat(
        [hetero_data["users"].reviews, mean_review], dim=0
    )

    hetero_data["users"].avg_hour = torch.cat(
        [hetero_data["users"].avg_hour, global_avg_hour], dim=0
    )

    # create edges for ALL rated series
    for series_id, rating in ratings_dict.items():
        # forward edge
        new_edge = torch.tensor([[user_index], [series_id]], dtype=torch.long)
        hetero_data["users", "rate", "series"].edge_index = torch.cat(
            [hetero_data["users", "rate", "series"].edge_index, new_edge], dim=1
        )

        # forward edge attr
        hetero_data["users", "rate", "series"].edge_attr = torch.cat(
            [
                hetero_data["users", "rate", "series"].edge_attr,
                torch.tensor([rating], dtype=torch.float),
            ],
            dim=0,
        )

        # reverse edge
        new_rev_edge = torch.tensor([[series_id], [user_index]], dtype=torch.long)
        hetero_data["series", "rev_rate", "users"].edge_index = torch.cat(
            [hetero_data["series", "rev_rate", "users"].edge_index, new_rev_edge], dim=1
        )

        # reverse edge attr
        hetero_data["series", "rev_rate", "users"].edge_attr = torch.cat(
            [
                hetero_data["series", "rev_rate", "users"].edge_attr,
                torch.tensor([rating], dtype=torch.float),
            ],
            dim=0,
        )

    return (
        hetero_data,
        user_index,
    )


##new user comes in -> rate a series-> create new node and edge-> return the whole graph with user index
# ->send hetero_data to model to retrain->  send trained mode with new user index to get_recommendations
# -> get prediction form retrained model ->return recommended series names and print it
