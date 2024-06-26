import torch
import torch.nn.functional as f
from torch import nn


class BigramModel(nn.Module):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )

    def forward(self, inputs, targets=None):
        # inputs.shape -> (batch_size, context_length)
        logits = self.embedding(inputs)
        # logits.shape -> (batch_size, context_length, vocab_size)
        # but for torch's cross entropy loss, need (batch_size, vocab_size, context_length)
        if targets is None:
            return logits, None
        logits = logits.permute(0, 2, 1)
        loss = f.cross_entropy(logits, targets)
        logits = logits.permute(0, 2, 1)
        return logits, loss

    def generate(self, start_idx, max_tokens=100):
        for _ in range(max_tokens):
            logits, _ = self(start_idx)
            # we only care about last prediction
            pred = logits[:, -1, :]  # (batch_size, vocab_size)
            prob = f.softmax(pred, dim=-1)
            idx = prob.multinomial(num_samples=1)  # (batch_size, 1)
            start_idx = torch.cat([start_idx, idx], dim=-1)
        return start_idx
