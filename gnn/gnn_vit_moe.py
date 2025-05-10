import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from einops import rearrange, repeat


# Hyperparameters
dim = 256
num_heads = 4
num_experts = 4  # Must be >= top_k
top_k = 2        # Must be <= num_experts
patch_size = 4
image_size = 32
num_classes = 100
batch_size = 128
epochs = 30
lr = 3e-4


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random_tensor.floor()


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class GNNGate(nn.Module):
    def __init__(self, dim, num_experts, heads=4):
        super().__init__()
        self.gat = GATConv(dim, num_experts, heads=heads)
        self.norm = nn.LayerNorm(num_experts * heads)

    def forward(self, x, edge_index):
        scores = self.gat(x, edge_index)
        scores = self.norm(scores)
        return scores.mean(dim=-1)


class GNNMoELayer(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        assert top_k <= num_experts, "top_k must be <= num_experts"
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.gate = GNNGate(dim, num_experts)
        self.top_k = top_k
        self.aux_loss = 0

    def forward(self, x, edge_index):
        B, N, D = x.shape
        x_flat = x.reshape(B*N, D)

        # GNN Routing
        scores = self.gate(x_flat, edge_index).view(B, N, -1)
        topk_scores, topk_indices = scores.topk(min(self.top_k, scores.size(-1)), dim=-1)
        topk_scores = topk_scores.softmax(dim=-1)

        # Expert Computation
        out = torch.zeros_like(x)
        expert_counts = torch.zeros(len(self.experts), device=x.device)

        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_in = x[mask]
                expert_out = expert(expert_in)
                expert_weight = topk_scores[mask].gather(-1, (topk_indices[mask] == i).nonzero()[:, -1:])
                out[mask] += expert_weight * expert_out
                expert_counts[i] = mask.float().sum()

        # Load balancing loss
        prob = expert_counts / (B * N)
        self.aux_loss = (len(self.experts) * (prob * prob.mean() / (prob + 1e-6))).sum()

        return out


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, num_experts=4, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = GNNMoELayer(dim, num_experts)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, edge_index):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.moe(self.norm2(x), edge_index))
        return x


class ViT(nn.Module):
    def __init__(self, aux_layer=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        grid_size = image_size // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size**2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, num_experts, drop_path=0.1) for _ in range(6)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.edge_index = self._build_grid_edges(grid_size)

        self.aux_layer = aux_layer
        self.aux_head = nn.Linear(dim, num_classes)

    def _build_grid_edges(self, grid_size):
        idx = torch.arange(grid_size * grid_size).view(grid_size, grid_size)
        edges = []
        for i in range(grid_size):
            for j in range(grid_size):
                if i > 0: edges.append((idx[i,j], idx[i-1,j]))
                if i < grid_size-1: edges.append((idx[i,j], idx[i+1,j]))
                if j > 0: edges.append((idx[i,j], idx[i,j-1]))
                if j < grid_size-1: edges.append((idx[i,j], idx[i,j+1]))
        return torch.tensor(edges).t().contiguous()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B, N, D = x.size()
        cls_token = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]

        edge_index = self.edge_index.to(x.device)
        aux_logits = None

        for i, block in enumerate(self.blocks):
            x = block(x, edge_index)
            if i == self.aux_layer:
                aux_logits = self.aux_head(x[:, 0])

        x = self.norm(x[:, 0])
        return self.head(x), aux_logits


def train_and_evaluate():
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Data loading
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    train_losses, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, aux_out = model(x)
            aux_loss = 0.4 * criterion(aux_out, y) if aux_out is not None else 0
            moe_aux = sum(block.moe.aux_loss for block in model.blocks)
            loss = criterion(out, y) + aux_loss + moe_aux
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                correct += (logits.argmax(1) == y).sum().item()

        acc = correct / len(test_data)
        train_losses.append(total_loss / len(train_loader))
        test_accs.append(acc)
        print(f"Epoch {epoch}: Loss={train_losses[-1]:.4f}, Acc={acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'gnn_vit_moe_improved.pth')

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='Test Accuracy')
    plt.legend()
    plt.savefig('training_results.png')
    plt.show()

    # Visualize sample prediction
    model.eval()
    idx = random.randint(0, len(test_data) - 1)
    img, label = test_data[idx]
    input_tensor = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output, _ = model(input_tensor)
        pred_label = output.argmax(dim=1).item()

    plt.imshow(np.transpose((img * 0.5 + 0.5).numpy(), (1, 2, 0)))
    plt.title(f"Predicted: {train_data.classes[pred_label]}\nTrue: {train_data.classes[label]}")
    plt.axis('off')
    plt.savefig('sample_prediction.png')
    plt.show()


if __name__ == "__main__":
    print("Starting training...")
    train_and_evaluate()