import math
import torch
import torch.nn.functional as f
from torch import nn
import torch.utils


class CausalSelfAttention(nn.Module):
    """
    Multiply your embeddings[i] with Q, K and V to get q[i], k[i] and v[i] respectively
    score[i][j] = q[i] . k[j] , when  j <= i else 0
    norm_score[i] = softmax(score[i])
    final_value[i] = sum(norm_score[i][j] * v[j])

    Let us ignore the j <= i condition for now.
    Then our the above calculation is simply
    final_value = emdeddings . (softmax(q . k_T) . v )

    To enforce the j <=i condition, we can just set the j > i part to be -inf before applying softmax

    There is an additional scaling factor attn_size to be divided.
    This ensures unit standar deviation and prevents the softmax function to be spiky

    **NOTE:** In the original Attention is All You Need paper, values is of a different dimension which is then
    projected using a linear layer, but here we'll make sure that attn_size is a factor of embed_size
    """

    def __init__(self, embed_size, attn_size, context_length):
        super().__init__()
        assert embed_size % attn_size == 0
        self.Q = nn.Linear(embed_size, attn_size, bias=False)
        self.K = nn.Linear(embed_size, attn_size, bias=False)
        self.V = nn.Linear(embed_size, attn_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length)).view(
                1, context_length, context_length
            ),
        )

    def forward(self, embeddings):
        # embeddings -> (batch_size, context_length, embed_size)
        q = self.Q(embeddings)  # (batch_size, context_length, attn_size)
        k = self.K(embeddings)  # (batch_size, context_length, attn_size)
        v = self.V(embeddings)  # (batch_size, context_length, attn_size)
        _, context_length, attn_size = q.shape
        scores = (
            q @ k.transpose(-2, -1) * (1 / math.sqrt(float(attn_size)))
        )  # (batch_size, context_length, context_length)
        masked_scores = torch.where(
            self.tril[:, :context_length, :context_length] == 0, float("-inf"), scores
        )  # (batch_size, context_length, context_length)
        norm_scores = f.softmax(
            masked_scores, dim=-1
        )  # (batch_size, context_length, context_length)
        final_values = norm_scores @ v  # (batch_size, context_length, attn_size)
        return final_values


class MLP(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fcn = nn.Linear(embed_size, embed_size * 4)
        self.activation = nn.ReLU()
        self.proj = nn.Linear(4 * embed_size, embed_size)

    def forward(self, embeddings):
        x = self.fcn(embeddings)
        x = self.activation(x)
        x = self.proj(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_size, num_heads, context_length):
        super().__init__()
        attn_size = embed_size // num_heads
        self.attn_heads = nn.ModuleList(
            (CausalSelfAttention(embed_size, attn_size, context_length))
            for _ in range(num_heads)
        )
        self.mlp = MLP(embed_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.num_heads = num_heads

    def forward(self, embeddings):
        x = self.layer_norm_1(
            embeddings
            + torch.cat(
                [self.attn_heads[i](embeddings) for i in range(self.num_heads)], dim=-1
            )
        )
        x = self.layer_norm_2(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, context_length, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=context_length, embedding_dim=embed_size
        )
        self.attn_blocks = nn.ModuleList(
            (
                AttentionBlock(embed_size, num_heads, context_length)
                for _ in range(num_layers)
            )
        )
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        self.context_length = context_length
        self.num_layers = num_layers

    def forward(self, inputs, targets=None):
        # inputs.shape -> (batch_size, context_length)
        _, context_length = inputs.shape
        x = self.token_embedding(inputs) + self.pos_embedding(
            torch.arange(context_length, device=inputs.device)
        )
        for attn_block in self.attn_blocks:
            x = attn_block(x)
        logits = self.lm_head(x)
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
            logits, _ = self(
                start_idx[:, -self.context_length :]
            )  # we only care till context length

            # we only care about last prediction
            pred = logits[:, -1, :]  # (batch_size, vocab_size)
            prob = f.softmax(pred, dim=-1)
            idx = prob.multinomial(num_samples=1)  # (batch_size, 1)
            start_idx = torch.cat([start_idx, idx], dim=-1)
        return start_idx
