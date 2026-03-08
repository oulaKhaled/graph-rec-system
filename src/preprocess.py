import torch
import numpy as np
from src.model import load_model
import pandas as pd

# from sentence_transformers import SentenceTransforme

path = ""

model = load_model()
# review_model = SentenceTransformer(f"{path}review_model")
series_details_df = pd.read_csv(f"{path}series_details_df1.csv")
df_reviews_rate_exist = pd.read_csv(f"{path}df_reviews_rate_exist1.csv")

## should user rate more than one sereis?
# wouldnt take a lot of time to retrain model on 60 epochs each time a user comes in and rate a series?


## how to know if this a new user or an existing user?
## when an existince user comes again , you should create new edges and send data to model again
## what if a lot of users comes in and the data become bigger and bigger?


## create a function that will retrun nodes count to observe how many new users comes in


def add_new_user(
    rated_series_id,
    data,
    rating,
    # review_model,
    series_details_df,
    df_reviews_rate_exist,
):
    series_overview = series_details_df[series_details_df["index"] == rated_series_id][
        "overview"
    ].tolist()

    embedded_reviews = model.encode(series_overview)
    mean_review = torch.tensor(
        embedded_reviews.mean(axis=0), dtype=torch.int64
    ).unsqueeze(0)

    user_index = int(data["users"].x.max().item()) + 1

    global_avg_hour = torch.tensor(
        [df_reviews_rate_exist.groupby("author")["hour"].mean().mean()],
        dtype=torch.int64,
    )

    data["users"].x = torch.cat(
        [data["users"].x, torch.tensor([user_index], dtype=torch.int64)], dim=0
    )

    data["users"].reviews = torch.cat([data["users"].reviews, mean_review], dim=0)

    data["users"].avg_hour = torch.cat([data["users"].avg_hour, global_avg_hour], dim=0)
    new_edge = torch.tensor([[user_index], [rated_series_id]], dtype=torch.long)
    data["users", "rate", "series"].edge_index = torch.cat(
        [data["users", "rate", "series"].edge_index, new_edge], dim=1
    )

    new_attr = torch.tensor([rating], dtype=torch.float)

    data["users", "rate", "series"].edge_attr = torch.cat(
        [data["users", "rate", "series"].edge_attr, new_attr], dim=0
    )
    # reverse edge
    new_rev_edge = torch.tensor([[rated_series_id], [user_index]], dtype=torch.long)
    data["series", "rev_rate", "users"].edge_index = torch.cat(
        [data["series", "rev_rate", "users"].edge_index, new_rev_edge], dim=1
    )

    # reverse edge attr ← add this too
    data["series", "rev_rate", "users"].edge_attr = torch.cat(
        [
            data["series", "rev_rate", "users"].edge_attr,
            torch.tensor([rating], dtype=torch.float),
        ],
        dim=0,
    )

    return data, user_index


##new user comes in -> rate a series-> create new node and edge-> return the whole graph with user index
# ->send data to model to retrain->  send trained mode with new user index to get_recommendations
# -> get preidciton form retrained model ->return recommended series names and print it
