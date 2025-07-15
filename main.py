import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

# Инициализация токенизатора
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
eos_token = "/"
eos_token_id = tokenizer.token_to_id(eos_token)
if eos_token_id is None:
    raise ValueError(f"EOS token '{eos_token}' не найден в словаре токенизатора")


class MaximkaGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x[:, :seq_len])
        
        x_embedded = self.token_embedding(x) + self.position_embedding(positions)

        # Создаем маску для автогрессивного моделирования
        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device) * float('-inf'), diagonal=1)

        # Передача через трансформер с маской
        x_transformed = self.transformer(x_embedded, mask=mask)

        logits = self.fc_out(x_transformed)

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
        y = self.tokenized_text[idx + 1:idx + 1 + self.seq_len]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":
    seq_len = 64
    batch_size = 32
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    num_epochs = 10
    learning_rate = 1e-4
    dropout_number = 0.2

    # Загрузка основного датасета
    with open("dataset.txt", "r", encoding="utf-8") as f:
        raw_text_train = f.read()

    # Загрузка валидатора
    with open("vali.txt", "r", encoding="utf-8") as f:
        raw_text_vali = f.read()

    # Токенизация данных
    tokenized_train = tokenizer.encode(raw_text_train).ids
    tokenized_vali = tokenizer.encode(raw_text_vali).ids

    print(f"Tokenized train size: {len(tokenized_train)}")
    print(f"Tokenized validation size: {len(tokenized_vali)}")

    # Создаем датасеты и DataLoader'ы
    train_dataset = TextDataset(tokenized_train, seq_len, eos_token_id=eos_token_id)
    vali_dataset = TextDataset(tokenized_vali, seq_len, eos_token_id=eos_token_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    vali_loader = DataLoader(vali_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MaximkaGPT(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
        dropout=dropout_number
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print("Training start")
    print(f"Batch count: {len(train_loader)}")

    for epoch in range(num_epochs):
        model.train()
        total_loss_train = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            logits_flattened = logits.view(-1, logits.size(-1))
            batch_y_flattened = batch_y.view(-1)

            loss_train_batch = loss_fn(logits_flattened, batch_y_flattened)
            loss_train_batch.backward()
            optimizer.step()

            total_loss_train += loss_train_batch.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss_train_batch.item():.4f}")

        avg_loss_train = total_loss_train / len(train_loader)

        print(f"Validation Start")
        # Валидация после каждой эпохи
        model.eval()
        total_loss_vali = 0
        with torch.no_grad():
            for batch_x_vali, batch_y_vali in vali_loader:
                batch_x_vali, batch_y_vali = batch_x_vali.to(device), batch_y_vali.to(device)
                logits_vali = model(batch_x_vali)
                logits_flattened = logits_vali.view(-1, logits_vali.size(-1))
                batch_y_flattened = batch_y_vali.view(-1)
                loss_vali = loss_fn(logits_flattened, batch_y_flattened)
                total_loss_vali += loss_vali.item()

        avg_loss_vali = total_loss_vali / len(vali_loader)

        print(f"Epoch {epoch + 1} completed.")
        print(f"Training Loss: {avg_loss_train:.4f}")
        print(f"Validation Loss: {avg_loss_vali:.4f}")
        torch.save(model.state_dict(), "maximkaGPT-1.pth")