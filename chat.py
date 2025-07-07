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


model = torch.load("model.pth")
model.eval()

def sample_next_token(logits, temperature=1.0, top_k=100):
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


def chat_with_model(model, initial_text, max_response_tokens=50, temperature=1.0, top_k=100, device='cpu'):
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

print("Введите 'exit' для выхода")
while True:
    userText = input("You: ")
    if userText.strip().lower() == "exit":
        break
    response = chat_with_model(model, userText, max_response_tokens=500, temperature=0.8, top_k=100)
    print("Model:", response)

#MODEL KNOWS ONLY RUSSIAN