import torch
import torch.nn.functional as F
from pathlib import Path
from model import CharGPT, BLOCK_SIZE


# Find the project folder
BASE_DIR = Path(__file__).resolve().parent

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Limit CPU threads to reduce overhead on small servers
if device == "cpu":
    torch.set_num_threads(1)


# Load the original Shakespeare text
with open(BASE_DIR / "shakespeare-1.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Rebuild the vocabulary used during training
chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# Initialize the Transformer
model = CharGPT(vocab_size=len(chars)).to(device)

# Load the trained model weights
checkpoint = torch.load(
    BASE_DIR / "gpt_best.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model"])
model.eval()


# Generate Shakespeare-style dialogue
@torch.inference_mode()
def generate_text(prompt, temperature=1.0, new_chars=250):

    # Make sure the generation settings are valid
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero")

    if new_chars < 1:
        raise ValueError("Generation length must be at least 1")

    # Convert the starting text into tokens
    tokens = [stoi[ch] for ch in prompt]

    if not tokens:
        raise ValueError("Prompt cannot be empty")

    generated = []

    for _ in range(new_chars):

        # Only use the last 256 characters
        context = tokens[-BLOCK_SIZE:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )

        # Get the model's predictions
        logits = model(x)

        # Adjust randomness
        next_logits = logits[:, -1, :].float() / temperature

        # Convert predictions into probabilities
        probabilities = F.softmax(next_logits, dim=-1)

        # Sample the next character
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        # Add the new character to the sequence
        tokens.append(next_token)
        generated.append(next_token)

    # Convert the generated tokens back into text
    return "".join(itos[token] for token in generated)


# Test the text generator
if __name__ == "__main__":

    print("Loaded GPT checkpoint from step:", checkpoint["step"])
    print("Running on:", device)

    prompt = "ROMEO:\n"

    print("\nGenerating Shakespeare-style dialogue...\n")

    result = generate_text(
        prompt=prompt,
        temperature=0.8,
        new_chars=250
    )

    print(prompt + result)