import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.functional as F
from contextlib import nullcontext
import os

#-------------------DATASET SETUP---------------------------

# Load Shakespeare text
with open("shakespeare-1.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create character vocabulary
chars = sorted(set(text))

# Map characters to numbers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Convert text into token IDs
data = torch.tensor(
    [stoi[ch] for ch in text],
    dtype=torch.long
)

# Split into 90% training and 10% validation
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

#----------------------CONFIGURATION--------------------

VOCAB_SIZE = len(chars)
BLOCK_SIZE = 256
D = 384
H = 6
NUM_LAYERS = 6
DROPOUT = 0.2

HEAD_SIZE = D // H

#--------------------CAUSAL ATTENTION--------------------

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

        # Save attention weights for verification
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

#--------------------FEED FORWARD------------------

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

#------------------TRANSFORMER BLOCK----------------

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(D, eps=1e-5, bias=False)
        self.attention = CausalMultiHeadAttention()

        self.ln2 = nn.LayerNorm(D, eps=1e-5, bias=False)
        self.ffn = FeedForward()

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Attention with resdual connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward with residual connection
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x

#--------------------GPT MODEL------------------------
class CharGPT(nn.Module):
    def __init__(self):
        super().__init__()

        # Token and position embeddings
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, D)

        self.embedding_dropout = nn.Dropout(DROPOUT)

        # Six transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock() for _ in range(NUM_LAYERS)
        ])

        self.final_ln = nn.LayerNorm(D, eps=1e-5, bias=False)

        # Convert hidden states into vocabulary logits
        self.lm_head = nn.Linear(D, VOCAB_SIZE, bias=False)

        # Share weghts with token embeddings
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
        # Initialize Linear layer weights
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        #Initialize embedding weights
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        #Set LayerNorm weights to 1
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


#-------------------VERIFY ARCHITECTURE-------------------

# torch.manual_seed(1337)

# model = CharGPT()

# # Count trainable parameters
# param_count = sum(p.numel() for p in model.parameters())

# print("Parameter count:", param_count)
# assert param_count == 10745088

# # Verify weight sharing
# assert model.token_embedding.weight is model.lm_head.weight
# print("Weight sharing: Passed")

# model.eval()

# with torch.no_grad():
#     # Test a short sequence
#     short_input = torch.randint(0, VOCAB_SIZE, (2, 16))
#     short_logits = model(short_input)

#     print("Short input logits:", short_logits.shape)

#     # Test the full context length
#     full_input = torch.randint(0, VOCAB_SIZE, (2, 256))
#     full_logits = model(full_input)

#     print("Full input logits:", full_logits.shape)

#     assert short_logits.shape == (2, 16, VOCAB_SIZE)
#     assert full_logits.shape == (2, 256, VOCAB_SIZE)

# print("Shape checks: Passed")


#---------------------VERIFY CAUSAL ATTENTION--------------------
# torch.manual_seed(1337)
# model = CharGPT()
# model.eval()

# with torch.no_grad():
#     # Create a random sequence
#     X = torch.randint(0, VOCAB_SIZE, (1, 16))

#     # Get predictions and attention weights
#     original_logits = model(X)
#     attention = model.blocks[0].attention.last_attention

#     # Check that attention probabilities sum to 1
#     row_sums = attention.sum(dim=-1)
#     assert torch.allclose(
#         row_sums,
#         torch.ones_like(row_sums),
#         atol=1e-6
#     )

#     # Check that future positions have zero attention
#     future_mask = torch.triu(
#         torch.ones(16, 16, dtype=torch.bool),
#         diagonal=1
#     )

#     assert torch.all(attention[:, :, future_mask] == 0)

#     print("Attention normalization: Passed")
#     print("Causal masking: Passed")

#     # Change future characters while keeping the prefix unchanged
#     changed_X = X.clone()
#     changed_X[:, 8:] = (changed_X[:, 8:] + 1) % VOCAB_SIZE

#     changed_logits = model(changed_X)

#     # Earlier predictions should remain unchanged
#     max_diff = (
#         original_logits[:, :8] - changed_logits[:, :8]
#     ).abs().max().item()

#     print(f"Maximum prefix difference: {max_diff:.8f}")

#     assert max_diff < 1e-5

#     print("Future-token independence: Passed")

#--------------------OPTIMIZER AND LEARNING RATE--------------------

# Set the training parameters
MAX_ITERS = 5000
WARMUP_ITERS = 100
MAX_LR = 1e-3
MIN_LR = 1e-4

# Calculate learning rate for each training step
def get_learningrate(step):

    # Increase learning rate during warmup
    if step < WARMUP_ITERS:
        return MAX_LR * (step + 1) / 101

    # Decrease learning rate using cosine decay
    progress = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)

    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


# Separate parameters by whether they need weight decay
def configure_optimizer(model):

    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0}
        ],
        lr=MAX_LR,
        betas=(0.9, 0.99),
        eps=1e-8
    )

    return optimizer

#----------------GPT BATCH GENERATION----------------

T = 256
B = 64

def get_batch(data, generator):

    # Pick random starting positions
    starts = torch.randint(
        0, len(data) - T, (B,), generator=generator
    )

    # Get input characters and their next-character targets
    X = torch.stack([data[s:s+T] for s in starts])
    Y = torch.stack([data[s+1:s+T+1] for s in starts])

    return X.to(device), Y.to(device)

#---------------------TRAINING SETUP-----------------------

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Training device:", device)

# Choose mixed precision
if device == "cuda":
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    amp_dtype = torch.float32

#print("Autocast dtype:", amp_dtype)

# Only use gradient scaling with FP16
use_scaler = device == "cuda" and amp_dtype == torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

# Seed immediately before creating the GPT
torch.manual_seed(1337)
model = CharGPT().to(device)

# Create optimizer
optimizer = configure_optimizer(model)

# Separate generators for training and evaluation
train_gen = torch.Generator().manual_seed(1337)
eval_gen = torch.Generator().manual_seed(2026)

# Track skipped optimizer steps
skipped_steps = 0

# Verify model
param_count = sum(p.numel() for p in model.parameters())

#print("GPT parameters:", param_count)
#print("Weight sharing:", model.lm_head.weight is model.token_embedding.weight)

assert param_count == 10745088
assert model.lm_head.weight is model.token_embedding.weight

#---------------------GPT VALIDATION-------------------------------

EVAL_INTERVAL = 250
EVAL_BATCHES = 200

def autocast_context():
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data)
    ]:
        total_loss = 0.0
        total_tokens = 0

        for _ in range(EVAL_BATCHES):
            X, Y = get_batch(split_data, eval_gen)

            with autocast_context():
                logits = model(X)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    Y.reshape(-1),
                    reduction="sum"
                )

            total_loss += loss.item()
            total_tokens += Y.numel()

        results[split_name] = total_loss / total_tokens

    model.train()
    return results

#----------------------GPT TRAINING------------------------------

def train_gpt():

    global skipped_steps

    train_losses = []
    val_losses = []
    eval_steps = []

    best_val_loss = float("inf")
    best_step = 0

    model.train()

    # Evaluate befor any training
    losses = estimate_loss()

    train_losses.append(losses["train"])
    val_losses.append(losses["val"])
    eval_steps.append(0)

    best_val_loss = losses["val"]

    torch.save({
        "model": model.state_dict(),
        "step": 0,
        "val_loss": best_val_loss
    }, "gpt_best.pt")

    print(f"Step 0 | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}", flush=True)

    for step in range(MAX_ITERS):

        # Update learning rate
        lr = get_learningrate(step)

        for group in optimizer.param_groups:
            group["lr"] = lr

        # Get training batch
        X, Y = get_batch(train_data, train_gen)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with autocast_context():
            logits = model(X)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                Y.reshape(-1)
            )

        # Backprop
        if use_scaler:
            old_scale = scaler.get_scale()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() < old_scale:
                skipped_steps += 1

        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

        completed = step + 1

        # Evaluate every 250 completed updates
        if completed % EVAL_INTERVAL == 0:

            losses = estimate_loss()

            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            eval_steps.append(completed)

            print(
                f"Step {completed}/{MAX_ITERS} | "
                f"Train: {losses['train']:.4f} | "
                f"Val: {losses['val']:.4f}",
                flush=True
            )

            # Save best checkpoint
            if losses["val"] < best_val_loss:

                best_val_loss = losses["val"]
                best_step = completed

                torch.save({
                    "model": model.state_dict(),
                    "step": completed,
                    "val_loss": best_val_loss
                }, "gpt_best.pt")

            # Save latest checkpoint
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": completed,
                "val_loss": losses["val"],
                "train_losses": train_losses,
                "val_losses": val_losses,
                "eval_steps": eval_steps,
                "best_step": best_step,
                "best_val_loss": best_val_loss,
                "skipped_steps": skipped_steps,
                "train_rng": train_gen.get_state(),
                "eval_rng": eval_gen.get_state()
            }, "gpt_latest.pt")

    print("\nTraining done")
    print("Best step:", best_step)
    print("Best validation loss:", best_val_loss)
    print("Skipped optimizer steps:", skipped_steps)

    return eval_steps, train_losses, val_losses, best_step

#----------------RUN THE TRAINING------------------------

#if __name__ == "__main__":
#     train_gpt()

# Training is finished so commenting out the training to make sure
# it is not accidentally ran again
