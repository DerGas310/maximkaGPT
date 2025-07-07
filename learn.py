# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer


tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)


eos_token_id = tokenizer.token_to_id("")
if eos_token_id is None:
    print("Внимание: токен '' не найден в словаре токенизатора. Добавьте его при обучении токенизатора.")
else:
    print(f"EOS token id: {eos_token_id}")


class SimpleGPT(nn.Module):
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

with open("txt.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenized_text = tokenizer.encode(raw_text).ids
print(f"Длина токенизированного текста: {len(tokenized_text)}")


dataset = TextDataset(tokenized_text, seq_len, eos_token_id=eos_token_id)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

model = SimpleGPT(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    max_seq_len=seq_len
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

print("Начинаем обучение")
print(f"Общее число батчей: {len(dataloader)}")

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
    print(f"Epoch {epoch + 1} завершена, средний Loss: {avg_loss:.4f}")


def sample_next_token(logits, temperature=1.0, top_k=50):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return top_k_indices[next_token].item()

def toTokens(text):
    encoding = tokenizer.encode(text)
    return encoding.ids

def fromTokens(ids):
    return tokenizer.decode(ids)


def chat_with_model(model, initial_text, max_response_tokens=50, temperature=1.0, top_k=50, device='cpu'):
    model.eval()
    input_tokens_list = toTokens(initial_text)
    input_tensor = torch.tensor(input_tokens_list, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_response_tokens):
        if input_tensor.size(1) > model.max_seq_len:
            input_tensor = input_tensor[:, -model.max_seq_len:]

        with torch.no_grad():
            logits = model(input_tensor)

        if logits.shape[1] == 0:
            print("Логиты пусты! Проверьте входные данные.")
            break

        last_logits = logits[:, -1, :].squeeze(0)

        predicted_token_id = sample_next_token(last_logits, temperature=temperature, top_k=top_k)

        if eos_token_id is not None and predicted_token_id == eos_token_id:
            break

        input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_token_id]], device=device)], dim=1)

    generated_tokens = input_tensor.squeeze(0).tolist()
    generated_text = fromTokens(generated_tokens)

    return generated_text

# Основной цикл взаимодействия с пользователем
print("Введите 'exit' для выхода")
while True:
    userText = input("Вы: ")
    if userText.strip().lower() == "exit":
        break
    response = chat_with_model(model, userText, max_response_tokens=500, temperature=0.5, top_k=50)
    print("Модель:", response)
