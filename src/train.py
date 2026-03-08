import torch
import tqdm
from app import device
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
from torch_geometric import transforms as T
import torch.nn.functional as F


data = ""
transform = T.RandomLinkSplit(
    is_undirected=False,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
    edge_types=[("users", "rate", "series")],
    rev_edge_types=[("series", "rev_rate", "users")],
)
train, valid, test = transform(data)


edge_label_index = train["users", "rate", "series"].edge_label_index
edge_label = train["users", "rate", "series"].edge_label

##Dynamic
train_loader = LinkNeighborLoader(
    data=train,
    num_neighbors=[30, 10],
    neg_sampling=NegativeSampling(
        mode="triplet",
        amount=1,
    ),
    edge_label_index=(("users", "rate", "series"), edge_label_index),
    batch_size=48,
    shuffle=True,
)


def train():
    print(f"Device : {device}")
    train_losses = []
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.002477535951650731, weight_decay=0.0001198116362081302
    )
    for epoch in range(0, 100):
        total_loss = total_example = 0
        model.train()
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)
            pos_score, neg_score = model(sampled_data)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score)
            )
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * (pos_score.numel() + neg_score.numel())
            total_example += pos_score.numel() + neg_score.numel()
            train_loss = total_loss / total_example
        train_losses.append(train_loss)
        print(f"\nEpoch :{epoch:03d}, Loss :{train_loss:.4f}")
