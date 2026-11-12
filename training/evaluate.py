#-----------------------BONUS 3 PART C--------------------------
# This file will use my alreayd trained GPT model to test how context length 
# will affect prediction loss and how temperature will change generated text.
# I will simply load the model since it is already trained and use its best checkpoint.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import train as g


#----------------------LOAD THE TRAINED GPT-----------------------

# Load the best checkpoint without training again
checkpoint = torch.load(
    "gpt_best.pt",
    map_location=g.device,
    weights_only=True
)

g.model.load_state_dict(checkpoint["model"])
g.model.eval()

print("Loaded checkpoint from step:", checkpoint["step"])



#--------------------Experiment 1: CONTEXT LENGTH-----------------------

# Test how much previous text helps predict the next character
context_lengths = [32, 64, 128, 256, 512, 1024]


# Use a separate generator so the sampled positions stay reproducible
position_gen = torch.Generator().manual_seed(2027)

# Pick 512 validation positions with at least 1024 earlier characters
positions = torch.randint(
    1024,
    len(g.val_data),
    (512,),
    generator=position_gen
)

# Store the average loss for each context length
context_losses = []


#---------------------------CALCULATE CONTEXT LOSSES--------------------

# Run inference only, without updating the model
with torch.no_grad():

    for C in context_lengths:

        total_loss = 0.0

        for a in positions.tolist():

            # Use the last 256 characters if the requested context is longer
            used_context = min(C, g.BLOCK_SIZE)

            # Grab the charactrs immediately before the target position
            x = g.val_data[a - used_context:a].unsqueeze(0).to(g.device)

            # Keep the original positions from the full 256-character window
            start_position = g.BLOCK_SIZE - used_context


            pos = torch.arange(
                start_position,
                g.BLOCK_SIZE,
                device=g.device
            )

            # Get the next-character prediiction at the last input position
            logits = g.model(x, positions=pos)

            # Only score the actual target character
            target = g.val_data[a].view(1).to(g.device)

            loss = F.cross_entropy(
                logits[:, -1, :].float(),
                target
            )

            total_loss += loss.item()

        # Average across the same 512 target characters
        average_loss = total_loss / len(positions)

        context_losses.append(average_loss)

        print(f"Context {C}: Loss = {average_loss:.4f}")

#--------------------PLOT CONTEXT LOSSES------------------------------

plt.figure(figsize=(8, 5))
plt.plot(context_lengths, context_losses, marker="o")

plt.xlabel("Context Length (Characters)")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("GPT Context Length vs. Prediction Loss")


plt.xticks(context_lengths)
plt.grid(True)
plt.savefig("gpt_context_loss.png", dpi=150, bbox_inches="tight")
plt.close()


print("Saved graph: gpt_context_loss.png")



#--------------------Experiment 2: TEMPERATURE--------------------

# Calculate the repetition score using four-character substrings
def calculate_r4(text):

    # Get every overlaping four-character substring
    substrings = [
        text[i:i + 4]
        for i in range(len(text) - 3)
    ]


    # Count how many different substrings there are
    unique_count = len(set(substrings))

    # Calculate the fraction of repeated substrings
    return 1 - (unique_count / len(substrings))

#--------------------GENERATE TEXT--------------------

# Generate new text using the trained GPT and a chosen temperature
@torch.no_grad()
def generate_text(prompt, temperature, seed, new_chars=128):

    # Use a separate random generator for each seed
    generator = torch.Generator(device=g.device).manual_seed(seed)


    # Convert the prompt into character IDs
    tokens = [g.stoi[ch] for ch in prompt]

    # Keep track of the generated characters separately from the prompt
    generated = []

    for _ in range(new_chars):

        # Only feed the most recent 256 chars to the GPT
        context = tokens[-g.BLOCK_SIZE:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=g.device
        )

        # Reset positions to 0 through T-1
        positions = torch.arange(
            x.shape[1],
            device=g.device
        )

        # Get predictions from the last character
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(str(g.device).startswith("cuda"))
        ):
            logits = g.model(x, positions=positions)



        # Divide logits by temperature to control randomness
        next_logits = logits[:, -1, :].float() / temperature

        # Convert logits to probabilities
        probabilities = F.softmax(next_logits, dim=-1)

        # Sample the next character using the chosen seed
        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator
        ).item()

        # Add the charactr to the context and generated output
        tokens.append(next_token)
        generated.append(next_token)


    # Decode the entire continuation
    return "".join(g.itos[token] for token in generated)


#-------------------------RUN TEMPERATURE EXPERIMENT------------------------

temperatures = [0.7, 1.0, 1.3]
seeds = [1, 2, 3]

#Store the average repetition score for each temperature
temperature_results = {}

for temperature in temperatures:

    repetition_scores = []

    print(f"\n========== TEMPERATURE {temperature} ==========")

    for seed in seeds:
        # Generate 128 new characters after the ROMEO prompt
        continuation = generate_text(
            prompt="ROMEO:\n",
            temperature=temperature,
            seed=seed
        )
        # Calculate repetition using only the generated characters
        score = calculate_r4(continuation)
        repetition_scores.append(score)

        print(f"\nSeed {seed} | R4 = {score:.4f}")
        print("ROMEO:\n" + continuation)

    # Average the three repetiition scores
    average_r4 = sum(repetition_scores) / len(repetition_scores)

    temperature_results[temperature] = average_r4

    print(f"\nAverage R4 at temperature {temperature}: {average_r4:.4f}")


#---------------------------TEMPERATURE SUMMARY--------------------

print("\n========== TEMPERATURE SUMMARY ==========")

for temperature, average_r4 in temperature_results.items():
    print(f"Temperature {temperature}: Mean R4 = {average_r4:.4f}")