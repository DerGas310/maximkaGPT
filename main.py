# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer #тут должна быть ошибка / error must be here


tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)
eos_token_id = tokenizer.token_to_id("")


class MaxGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.transformer(x)  # batch x seq_len x embed_dim
        logits = self.fc_out(x)  # batch x seq_len x vocab_size
        return logits


class TextDataset(Dataset):
    def __init__(self, tokenized_text, seq_len, eos_token_id=None):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id

        if eos_token_id is not None:
            self.tokenized_text = tokenized_text + [eos_token_id]
        else:
            self.tokenized_text = tokenized_text

    def __len__(self):
        return len(self.tokenized_text) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokenized_text[idx:idx + self.seq_len]
        y = self.tokenized_text[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


seq_len = 64
batch_size = 32
embed_dim = 512
num_heads = 8
num_layers = 6
num_epochs = 5
learning_rate = 1e-4

with open("dataset.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenized_text = tokenizer.encode(raw_text).ids
print(f"Tokenized text size : {len(tokenized_text)}")


dataset = TextDataset(tokenized_text, seq_len, eos_token_id=eos_token_id)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MaxGPT(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    max_seq_len=seq_len
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

print("learn start")
print(f"batch count: {len(dataloader)}")

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)

        logits = logits.view(-1, logits.size(-1))
        batch_y_flat = batch_y.view(-1)

        loss = loss_fn(logits, batch_y_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} end, avg loss: {avg_loss:.4f}")


torch.save(model, "model.pth")