import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F

from tqdm.auto import tqdm

class SoftMoE(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        gate_weights = torch.softmax(self.gate(x), dim=-1)  # [batch, seq_len, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=3)  # [batch, seq_len, d_model, num_experts]
        return torch.einsum('bsd,bsde->bsd', gate_weights, expert_outputs)

class TransformerWithSoftMoE(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, num_experts=8):
        super().__init__()
        
        # Shared embedding for encoder/decoder (optional)
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        encoder_layer._ff_block = SoftMoE(d_model, num_experts)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        decoder_layer._ff_block = SoftMoE(d_model, num_experts)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory)
        return self.fc_out(output)

# Load WMT14 dataset
# dataset = load_dataset("wmt14", "de-en", split="train[:5000]")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


# Modified preprocessing function
def preprocess(batch):
    # Extract source and target texts
    src_texts = [ex["de"] for ex in batch["translation"]]
    tgt_texts = [ex["en"] for ex in batch["translation"]]

    # Tokenize batches
    src = tokenizer(src_texts, padding="max_length", truncation=True, max_length=128)
    tgt = tokenizer(tgt_texts, padding="max_length", truncation=True, max_length=128)

    # Prepare decoder inputs and labels
    return {
        "src_ids": src.input_ids,
        "tgt_ids": [t[:-1] for t in tgt.input_ids],  # Shift right for decoder input
        "labels": [t[1:] for t in tgt.input_ids]     # Shift left for labels
    }

dataset = load_dataset("wmt14", "de-en", split="train[:50000]")
dataset = dataset.map(preprocess, batched=True, batch_size=100)

# Convert to PyTorch tensors
dataset.set_format(type="torch", columns=["src_ids", "tgt_ids", "labels"])

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



# Initialize model
model = TransformerWithSoftMoE(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    num_experts=8
)


# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)


def train_epoch(model, dataloader):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(src, tgt)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(src, tgt)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
            progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

# Main training loop with tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(10):
    train_loss = train_epoch(model, dataloader)
    val_loss = evaluate(model, dataloader)
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")