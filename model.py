import torch
import math
import torch.nn as nn
import torch.nn.functional as F


# Model configuration
BLOCK_SIZE = 256
D = 384
H = 6
NUM_LAYERS = 6
DROPOUT = 0.2

HEAD_SIZE = D // H


# Causal multi-head attention
class CausalMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # Project hidden states into queries, keys, and values
        self.qkv = nn.Linear(D, 3 * D, bias=False)

        # Combine the attention heads
        self.proj = nn.Linear(D, D, bias=False)

        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)

        # Mask future positions
        mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("mask", mask, persistent=False)

        self.last_attention = None

    def forward(self, x):
        B, T, C = x.shape

        # Split into Q, K, V
        q, k, v = self.qkv(x).split(D, dim=2)

        # Separate attention heads
        q = q.view(B, T, H, HEAD_SIZE).transpose(1, 2)
        k = k.view(B, T, H, HEAD_SIZE).transpose(1, 2)
        v = v.view(B, T, H, HEAD_SIZE).transpose(1, 2)

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_SIZE)

        # Prevent attention to future characters
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # Normalize across keys
        attention = F.softmax(scores, dim=-1)

        if not self.training:
            self.last_attention = attention.detach()

        attention = self.attn_dropout(attention)

        # Apply attention to values
        out = attention @ v

        # Combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        out = self.proj(out)
        out = self.resid_dropout(out)

        return out


# Feedforward network
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(D, 4 * D, bias=False),
            nn.GELU(),
            nn.Linear(4 * D, D, bias=False)
        )

    def forward(self, x):
        return self.net(x)


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(D, eps=1e-5, bias=False)
        self.attention = CausalMultiHeadAttention()

        self.ln2 = nn.LayerNorm(D, eps=1e-5, bias=False)
        self.ffn = FeedForward()

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attention(self.ln1(x))

        # Feedforward with residual connection
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x


# Character-level GPT model
class CharGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, D)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, D)

        self.embedding_dropout = nn.Dropout(DROPOUT)

        # Six transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock() for _ in range(NUM_LAYERS)
        ])

        self.final_ln = nn.LayerNorm(D, eps=1e-5, bias=False)

        # Convert hidden states into vocabulary logits
        self.lm_head = nn.Linear(D, vocab_size, bias=False)

        # Share weights with token embeddings
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Special initialization for residual projections
        for name, parameter in self.named_parameters():
            if name.endswith("attention.proj.weight") or name.endswith("ffn.net.2.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(12)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)

    def forward(self, X, positions=None):
        B, T = X.shape

        if T > BLOCK_SIZE:
            raise ValueError("Input exceeds maximum context length")

        if positions is None:
            positions = torch.arange(T, device=X.device)

        # Combine token and position embeddings
        token_emb = self.token_embedding(X)
        pos_emb = self.position_embedding(positions)

        x = self.embedding_dropout(token_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)

        # Output next-character logits
        logits = self.lm_head(x)

        return logits
