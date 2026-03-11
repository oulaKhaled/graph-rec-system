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

Data was fetched from the **[TMDB API](https://developer.themoviedb.org/reference/getting-started)** and preprocessed into CSV files before being converted into a heterogeneous graph. 
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

## Graph Structure

<img width="600" height="600" alt="graph" src="https://github.com/user-attachments/assets/f1d5c019-397b-4045-963b-382f2d9c0c9b" />

The heterogeneous graph models five node types and four edge types. 
The **series** node acts as the central hub, connecting to users 
through rating interactions, and to metadata nodes (genres, writers, 
and types) through structural relationships. This multi-relational 
structure allows the GNN to capture both collaborative signals 
(user-series interactions) and content-based signals (series metadata) 
during message passing.

## Node Neighborhood Example

<img width="800" height="800" alt="graph node" src="https://github.com/user-attachments/assets/71a18dfe-d70a-49d3-9e66-e61ea6d6b5e8" />

Visualization of a single series node (center) and all its 
connected neighbors across the graph. The dense connections 
radiating outward illustrate why message passing is powerful — 
during a single GNN forward pass, this node aggregates information 
from hundreds of neighboring nodes, building a rich embedding that 
captures both direct and indirect relationships in the graph.


## Model Architecture

### GNN Backbone — GraphSAGE

```
Input features → SAGEConv layer 1 → ReLU → SAGEConv layer 2 → Node Embeddings
```

- Two `SAGEConv` layers converted to heterogeneous using `to_hetero()`
- Separate linear projection layers for user and series features
- Learned embeddings for genres, writers, and types  
## Model Visualization

### Series Embeddings (t-SNE)
<img width="790" height="490" alt="node_embeddings" src="https://github.com/user-attachments/assets/e604d460-2e21-4174-850a-cac4cb359592" />


The t-SNE plot shows the GNN's learned embeddings for all series in the graph,
colored by series type. The clear clustering indicates the model successfully 
learned to group similar content together — scripted series, documentaries, 
talk shows, and other types form distinct regions in the embedding space.
This separation is what enables meaningful recommendations.

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
<img width="567" height="455" alt="loss" src="https://github.com/user-attachments/assets/cdfffdce-9baf-45d1-9f5c-5be78f62857a" />

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
<img width="1589" height="490" alt="auc" src="https://github.com/user-attachments/assets/9eb80107-ba48-404c-b759-43e3df812769" />

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

### 3. User Identity Management

To handle returning users correctly, the system maintains a **user registry** 
that maps each user to their graph node index. Before processing any request, 
the system checks whether the user already exists in the graph:

**Returning user:**
- Retrieve existing node index from registry
- Update existing edge ratings if series was already rated
- Create new edges for newly rated series
- Run fresh GNN forward pass for updated embeddings

**New user (cold start):**
- Create new node with onboarding features
- Add rating edges to graph
- Register user in registry
- Run GNN forward pass for initial embeddings

This prevents duplicate nodes from being created for returning users, 
which would corrupt the graph structure and degrade recommendation quality 
over time.


---

## Hyperparameter Tuning

Hyperparameters were tuned using **Optuna** with multi-objective optimization:

- Directions: maximize AUC, maximize HR@5
- Parameters tuned: learning rate, weight decay, hidden channels, number of neighbors

---

## Tech Stack
- PyTorch & PyTorch Geometric — GNN implementation
- Sentence Transformers — text feature encoding
- Optuna — hyperparameter tuning
- TorchMetrics — ranking evaluation metrics
- Hugging Face Spaces — deployment
---

## References

### Papers
- Norton et al. — Heterogeneous Graph Recommendation Model based on Graph Neural Network
- Wang et al. — Graph Learning based Recommender Systems: A Review
- Xu et al. (2019) — How Powerful are Graph Neural Networks?
-  Li et al. (2023) — Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking

### Courses & Resources
- [apxml — Graph Neural Networks Course](https://apxml.com/courses/graph-neural-networks-gnns)
- PyTorch Geometric Documentation — https://pytorch-geometric.readthedocs.io
