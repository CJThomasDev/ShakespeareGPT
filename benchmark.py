
import time
import torch
from pathlib import Path

from model import CharGPT, BLOCK_SIZE


# Find our project files
BASE_DIR = Path(__file__).resolve().parent

# Rebuild the original vocabulary
with open(BASE_DIR / "shakespeare-1.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}

# Load the trained checkpoint
checkpoint = torch.load(
    BASE_DIR / "gpt_best.pt",
    map_location="cpu",
    weights_only=True
)


# Test generation speed on a device
@torch.inference_mode()
def benchmark(device, new_chars=250):

    model = CharGPT(vocab_size=len(chars)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokens = [stoi[ch] for ch in "ROMEO:\n"]

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(new_chars):

        context = tokens[-BLOCK_SIZE:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )

        logits = model(x)
        probabilities = torch.softmax(logits[:, -1, :].float() / 0.8, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        tokens.append(next_token)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"\nDevice: {device}")
    print(f"Characters generated: {new_chars}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Speed: {new_chars / elapsed:.2f} characters/second")


# Run the benchmarks
if __name__ == "__main__":

    print("Benchmarking ShakespeareGPT...")

    benchmark("cpu")

    if torch.cuda.is_available():
        benchmark("cuda")