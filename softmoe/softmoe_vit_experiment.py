import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.models.vision_transformer import vit_b_16
from ptflops import get_model_complexity_info


class SoftMoE(nn.Module):
    def __init__(self, embed_dim, num_experts=4, dropout=0.1):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)])
        self.gate_layer = nn.Linear(embed_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.last_gate_scores = None

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.view(B * N, D)
        gate_logits = self.gate_layer(x_flat)
        gate_scores = torch.softmax(gate_logits, dim=-1)
        self.last_gate_scores = gate_scores

        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        output = (gate_scores.unsqueeze(-1) * expert_outputs).sum(dim=1)
        output = output.view(B, N, D)
        return self.norm(x + self.dropout(output))


class ViTSoftMoE(nn.Module):
    def __init__(self, num_classes=100, num_experts=4):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        embed_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        self.moe = SoftMoE(embed_dim, num_experts)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)
        x = torch.cat((cls_token, x), dim=1)
        pos_embedding = self.vit.encoder.pos_embedding[:, :(x.shape[1])]
        x = x + pos_embedding
        x = self.vit.encoder.dropout(x)
        x = self.moe(x)
        x = self.vit.encoder(x)
        return self.head(x[:, 0])


def train_model(model, train_loader, device, epochs=3, lr=3e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")


def compute_entropy(gate_scores):
    return -torch.sum(gate_scores * torch.log(gate_scores + 1e-9), dim=1)


def evaluate_metrics(model, dataloader, device, num_experts):
    model.eval()
    correct = 0
    total = 0
    expert_usages = []
    gate_entropies = []
    batch_times = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            outputs = model(images)
            batch_times.append(time.time() - start)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            gate_scores = model.moe.last_gate_scores
            expert_usages.append(gate_scores.mean(dim=0).cpu().numpy())
            gate_entropies.append(compute_entropy(gate_scores).mean().item())

    return {
        'num_experts': num_experts,
        'accuracy': correct / total,
        'avg_inference_time': np.mean(batch_times),
        'total_inference_time': np.sum(batch_times),
        'avg_expert_usage': np.mean(expert_usages, axis=0),
        'gate_entropy': np.mean(gate_entropies)
    }


def estimate_flops(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    print(f"FLOPs: {macs}, Params: {params}")


def plot_results(results_by_experts):
    experts = [res['num_experts'] for res in results_by_experts]
    accuracies = [res['accuracy'] for res in results_by_experts]
    times = [res['avg_inference_time'] for res in results_by_experts]
    entropies = [res['gate_entropy'] for res in results_by_experts]

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(experts, accuracies, marker='o')
    plt.title("Accuracy vs Num Experts")
    plt.xlabel("Number of Experts")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(experts, times, marker='o', color='orange')
    plt.title("Inference Time vs Num Experts")
    plt.xlabel("Number of Experts")
    plt.ylabel("Avg Inference Time (s)")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(experts, entropies, marker='o', color='green')
    plt.title("Gate Entropy vs Num Experts")
    plt.xlabel("Number of Experts")
    plt.ylabel("Entropy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def run_experiments(expert_list, train_loader, test_loader, device):
    results = []
    for num_experts in expert_list:
        print(f"\n Running with {num_experts} experts")
        model = ViTSoftMoE(num_classes=100, num_experts=num_experts).to(device)

        estimate_flops(model, device)
        # train_model(model, train_loader, device, epochs=3)
        result = evaluate_metrics(model, test_loader, device, num_experts)
        results.append(result)
    return results


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-100 data loaders
    transform = transforms.Compose([
        transforms.Resize(224),  # For ViT input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading CIFAR-100 datasets...")
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Run experiments
    expert_counts = [2, 4, 8, 12]
    print("\nStarting experiments...")
    results = run_experiments(expert_counts, train_loader, test_loader, device)

    # Plot and save results
    plot_results(results)
    pd.DataFrame(results).to_csv("softmoe_experiment_results.csv", index=False)
    print("Experiment results saved to softmoe_experiment_results.csv")


if __name__ == "__main__":
    main()