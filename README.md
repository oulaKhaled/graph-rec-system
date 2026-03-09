# GNN-Based Series Recommendation System

A heterogeneous Graph Neural Network (GNN) recommendation system designed to generate personalized content recommendations using graph representation learning. The system models relationships between users and items in a graph structure and learns embeddings that capture higher-order interactions. The project focuses not only on training a GNN model, but also on building a complete pipeline, including graph persistence, embedding inference, evaluation metrics, and cold-start handling.

> **Note:** This system recommends series the user is likely to **watch** based on viewing history and interactions — not necessarily series they will rate highly. This is consistent with how major platforms like Netflix and YouTube approach recommendation.

---

## What is a Graph-Based Recommendation System?

A graph-based recommendation system is a type of recommendation engine that uses graph data structures to model relationships between entities like users, items, and interactions. Instead of relying solely on user-item matrices (as in traditional collaborative filtering), it represents data as nodes (users, series, genres) and edges (ratings, associations). This structure allows the system to capture complex, indirect relationships that might be missed by other methods.

For example, two users who have never interacted with the same series can still be connected through shared genre preferences, writers, or overlapping interests in a graph — enabling more nuanced and accurate recommendations than traditional methods.

---

## Project Overview

This project implements a link prediction model on a heterogeneous graph to recommend TV series to users based on their watch history and ratings. Rather than treating recommendation as a simple matrix factorization problem, this system models the rich relational structure between entities as a graph, allowing the model to capture multi-hop relationships between users, series, and their associated metadata.

Key design decisions that go beyond standard tutorials:

- **Triplet-mode dynamic negative sampling** to prevent shortcut learning
- **Node-centric negative sampling** to address evaluation inflation from random negatives
- **Message passing leakage prevention** using separate `edge_index` and `edge_label_index`
- **Ranking-based evaluation** (HR@K) instead of binary classification metrics
- **Cold-start solution** for new users using onboarding interactions

---

## Dataset

Data was fetched from the **TMDB API** and preprocessed into CSV files before being converted into a heterogeneous graph.

### Data Collection & Preprocessing

- Series details (overview, genres, creators, type, popularity) fetched from TMDB API
- User reviews and ratings scraped and cleaned
- Text (overviews and reviews) encoded using `sentence-transformers` (`all-MiniLM-L6-v2`) into 384-dimensional embeddings
- All data cleaned and saved as CSV files before graph construction

### Heterogeneous Graph Structure

| Node Type | Count | Features                                    |
| --------- | ----- | ------------------------------------------- |
| Users     | 641   | Review embeddings (384-dim), avg watch hour |
| Series    | ~2000 | Overview embeddings (384-dim)               |
| Genres    | -     | Learned embeddings                          |
| Writers   | -     | Learned embeddings                          |
| Types     | -     | Learned embeddings                          |

| Edge Type                    | Description                 |
| ---------------------------- | --------------------------- |
| users → rate → series        | User ratings (1-10)         |
| series → has → genres        | Series genre associations   |
| series → written_by → writer | Series creator associations |
| series → is → type           | Series type classification  |

---

## Model Architecture

### GNN Backbone — GraphSAGE

```
Input features → SAGEConv layer 1 → ReLU → SAGEConv layer 2 → Node Embeddings
```

- Two `SAGEConv` layers converted to heterogeneous using `to_hetero()`
- Separate linear projection layers for user and series features
- Learned embeddings for genres, writers, and types

### Link Prediction — Dot Product Classifier

```
user_embedding · series_embedding → edge score → sigmoid → probability
```

### Training Strategy

- **Triplet-mode dynamic negative sampling** via `LinkNeighborLoader`
- For each positive edge `(user, series)`, a negative series is sampled for the same user
- Loss: `BCEWithLogitsLoss` on positive and negative scores separately
- Optimizer: Adam with tuned learning rate and weight decay

### What the Loss Function Learns

During training, the model optimizes its parameters to produce embeddings that capture meaningful information about each node. The loss function encourages the model to:

- Pull embeddings of connected nodes **closer together** in embedding space
- Push embeddings of unconnected nodes **further apart**

This process helps the model learn representations that encode the underlying structure, characteristics, and relationships within the graph — which are then used for link prediction at inference time.

---

## Evaluation

### Metrics

| Metric    | Value |
| --------- | ----- |
| Test AUC  | 0.748 |
| Test HR@5 | 0.787 |

### Why HR@K over standard metrics

Standard random negative sampling inflates AUC by making negatives trivially easy to classify. This project uses **node-centric evaluation** where negatives share the same source node as positives, making evaluation significantly harder and more realistic.

HR@5 measures whether at least one relevant series appears in the top-5 recommendations for each user — directly reflecting real user experience.

---

## Key Technical Contributions

### 1. Triplet-Mode Negative Sampling

Instead of randomly sampling negative edges across the entire graph, negatives are sampled for the **same user** as each positive edge:

```
Positive: (user_5, series_1)  ✅ real connection
Negative: (user_5, series_9)  ← same user, different series
```

This forces the model to learn fine-grained user preferences rather than detecting structural differences between unrelated nodes.

### 2. Message Passing Leakage Prevention

Target edges are excluded from the message passing graph during training and evaluation:

```
edge_index       → structural graph (message passing only)
edge_label_index → supervision edges (prediction targets only)
```

### 3. Cold Start Solution

New users who have never been seen by the model are handled by:

1. Asking user to rate one or more series at onboarding
2. Using the first rated series overview embedding as initial user review feature
3. Using global average watch hour as initial behavioral feature
4. Adding user node and rating edges to the graph
5. Running a fresh GNN forward pass to generate embeddings

---

## Hyperparameter Tuning

Hyperparameters were tuned using **Optuna** with multi-objective optimization:

- Directions: maximize AUC, maximize HR@5
- Parameters tuned: learning rate, weight decay, hidden channels, number of neighbors

---

## Installation

```bash
pip install torch==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install torch-geometric
pip install sentence-transformers optuna torchmetrics scikit-learn
```

---

## References

- Kipf & Welling (2017) — Semi-Supervised Classification with Graph Convolutional Networks
- Hamilton et al. (2017) — Inductive Representation Learning on Large Graphs (GraphSAGE)
- Hu et al. (2020) — Open Graph Benchmark
- Yang et al. (2020) — Understanding Negative Sampling in Graph Representation Learning
- Geirhos et al. (2020) — Shortcut Learning in Deep Neural Networks
- RANS (2025) — Risk Aware Negative Sampling in Link Prediction

