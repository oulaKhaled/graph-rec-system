import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict
import pandas
from src.model import get_recommendation
from torch_geometric.data import HeteroData
from torch import Tensor

# --- FastAPI setup ---
app = FastAPI(title="Graph Movie Recommender")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


class RecommendRequest(BaseModel):
    username: str
    ratings: Dict[str, int]


@app.post("/recommend")
def recommend(req: RecommendRequest):
    print("Received ratings:", req.ratings)

    top_k = get_recommendation(req.username, req.ratings)

    # dummy response for now — return some tmdb_ids from your dataset
    return {"recommendations": top_k.tolist()}  # ← convert numpy to list


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Load model once at startup
# model = load_model()


# class RecommendRequest(BaseModel):
#     user_id: int
#     series_id: int
#     rating: int


# @app.get("/")
# def health():
#     ## print all series here
#     return {"status": "ok"}


# @app.post("/recommend")
# def recommend(req: RecommendRequest):
#     ##input to get user_id, series_id,rating
#     results = get_recommendations(model, req.user_id, req.series_id, req.rating)
#     return {"user_id": req.user_id, "recommendations": results}


## Options better than gradio
# --- Gradio UI (mounted at /ui) ---
# def gradio_predict(user_id: int):
#     results = get_recommendations(model, user_id)
#     return ", ".join(str(r) for r in results)


# demo = gr.Interface(
#     fn=gradio_predict,
#     inputs=gr.Number(label="User ID"),
#     outputs=gr.Text(label="Recommended Movie IDs"),
#     title="🎬 Graph Movie Recommender",
#     description="Enter a user ID to get personalized movie recommendations",
# )

# # Mount Gradio inside FastAPI
# app = gr.mount_gradio_app(app, demo, path="/ui")
