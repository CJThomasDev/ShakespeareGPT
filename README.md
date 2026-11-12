# ShakespeareGPT

### A GPT-Style Transformer Built From Scratch in PyTorch

ShakespeareGPT is a **10.7-million-parameter, character-level Transformer** trained to generate original Shakespeare-inspired dialogue.

Built entirely in PyTorch without pretrained models, ShakespeareGPT implements the core components of a GPT-style language model, including causal multi-head self-attention, positional embeddings, feed-forward networks, and autoregressive text generation.

The project includes a complete training and evaluation pipeline, a FastAPI inference backend, and an interactive web interface where users can generate dialogue with customizable settings.

## Overview

ShakespeareGPT was developed to explore how Transformer-based language models learn patterns in text and generate new sequences.

The model was trained on Shakespeare's writing and learns to predict the next character based on the preceding context.

Unlike applications that use existing pretrained models through an API, ShakespeareGPT uses a custom-built Transformer architecture with weights learned during training.

### Features

- Custom GPT-style Transformer implemented in PyTorch
- 10,745,088 trainable parameters
- Six Transformer blocks with causal multi-head self-attention
- Character-level tokenization and autoregressive generation
- Adjustable generation temperature and output length
- FastAPI backend for model inference
- Interactive frontend built with HTML, CSS, and JavaScript
- GPU-accelerated training using CUDA
- Checkpoint selection based on validation performance
- Evaluation against a bigram language model baseline

## Model Architecture

ShakespeareGPT uses a decoder-only Transformer architecture.

| Component | Configuration |
|---|---|
| Architecture | Decoder-Only Transformer |
| Parameters | 10,745,088 |
| Transformer Layers | 6 |
| Attention Heads | 6 |
| Embedding Dimension | 384 |
| Context Window | 256 Characters |
| Tokenization | Character-Level |
| Attention | Causal Multi-Head Self-Attention |
| Weight Tying | Token Embedding and Output Projection |

### Architecture Components

**Token and Positional Embeddings**

Input characters are converted into integer tokens and mapped to learned embedding vectors. Positional embeddings provide information about each character's position within the sequence.

**Causal Multi-Head Self-Attention**

Each Transformer block uses causal attention to process relationships between characters while preventing access to future tokens.

**Feed-Forward Networks**

Each block includes a feed-forward neural network that transforms the representations produced by the attention mechanism.

**Residual Connections and Layer Normalization**

Residual connections and layer normalization help maintain stable information flow through the Transformer.

**Autoregressive Generation**

During inference, the model predicts a probability distribution over the next character. A character is sampled and appended to the input, and the process repeats until the requested output length is reached.

## Training

ShakespeareGPT was trained using PyTorch on an NVIDIA RTX 3060 GPU.

### Training Configuration

| Setting | Value |
|---|---|
| Training Updates | 5,000 |
| Batch Size | 64 |
| Sequence Length | 256 |
| Optimizer | AdamW |
| Learning Rate Schedule | 100-Step Warmup + Cosine Decay |
| Gradient Clipping | 1.0 |
| Random Seed | 1337 |
| Training Device | NVIDIA RTX 3060 |
| Mixed Precision | BF16 Autocast |

Training and validation cross-entropy losses were monitored throughout the training process.

### Checkpoint Selection

The model achieved its lowest sampled validation loss at **step 1,750**.

Although training continued through 5,000 optimizer updates, validation performance began declining after step 1,750, indicating overfitting.

The best-performing checkpoint was retained for inference.

| Metric | Result |
|---|---:|
| Best Checkpoint | Step 1,750 |
| Best Sampled Validation Loss | 1.4678 |
| Final Training Loss | 0.6147 |
| Final Sampled Validation Loss | 1.7077 |

## Model Evaluation

ShakespeareGPT was evaluated against a simpler bigram language model trained on the same dataset.

Both models were evaluated using **111,539 validation targets**.

| Model | Parameters | Validation Cross-Entropy | Perplexity |
|---|---:|---:|---:|
| Bigram Model | 4,225 | 2.5212 | 12.4438 |
| **ShakespeareGPT** | **10,745,088** | **1.4679** | **4.3403** |

### Results

ShakespeareGPT achieved:

- **41.8% lower validation cross-entropy** than the bigram baseline.
- **65.1% lower validation perplexity** than the bigram baseline.
- Improved next-character prediction through attention over preceding context.

These results demonstrate the advantage of using a Transformer architecture to capture relationships across longer sequences rather than relying exclusively on the previous character.

### Context Length Evaluation

The model was also evaluated using different amounts of preceding context.

| Context Length | Average Cross-Entropy |
|---|---:|
| 32 | 1.6147 |
| 64 | 1.6060 |
| 128 | 1.5679 |
| 256 | 1.5867 |
| 512 | 1.5867 |
| 1024 | 1.5867 |

Evaluation used 512 sampled validation positions.

Because the model supports a maximum context window of 256 characters, additional characters beyond this limit do not provide more usable context.

## Interactive Web Application

ShakespeareGPT includes a browser-based interface that allows users to generate original Shakespeare-inspired dialogue.

The interface communicates with a FastAPI backend that loads the trained model and performs inference.

### User Controls

**Character Prompt**

Choose a starting character or enter a custom prompt.

**Temperature**

Adjust the randomness of the generated text.

Lower temperatures produce more predictable outputs, while higher temperatures introduce greater variation.

**Generation Length**

Choose how many characters the model generates.

### Backend API

The backend is implemented using FastAPI and exposes a text-generation endpoint.

**Endpoint**

`POST /generate`

**Example Request**

```json
{
  "prompt": "ROMEO:",
  "temperature": 0.8,
  "new_chars": 250
}
```

**Example Response Structure**

```json
{
  "generated_text": "ROMEO: ..."
}
```

The API validates incoming requests, runs inference using the trained Transformer, and returns the generated text.

## Inference Performance

Inference performance was benchmarked locally using 250 generated characters.

| Device | Generation Time | Throughput |
|---|---:|---:|
| NVIDIA RTX 3060 | 0.79 seconds | 316.40 chars/sec |
| CPU | 3.24 seconds | 77.24 chars/sec |

These measurements demonstrate the performance difference between CPU and GPU inference on the development machine.

## Technology Stack

### Machine Learning
- Python
- PyTorch
- CUDA
- Transformer Architecture
- Autoregressive Language Modeling

### Backend
- FastAPI
- Uvicorn
- REST API

### Frontend
- HTML
- CSS
- JavaScript

### Development
- Visual Studio 2022
- Git
- GitHub

## Project Structure

```text
ShakespeareGPT/
|-- frontend/
|   |-- index.html
|   |-- style.css
|   |-- script.js
|
|-- training/
|   |-- train.py
|   |-- evaluate.py
|
|-- model.py
|-- generate.py
|-- api.py
|-- benchmark.py
|-- gpt_best.pt
|-- shakespeare-1.txt
|-- requirements.txt
|-- README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `model.py` | Defines the Transformer architecture |
| `training/train.py` | Trains the model and saves checkpoints |
| `training/evaluate.py` | Evaluates model performance |
| `generate.py` | Loads the trained model and generates text |
| `api.py` | Handles HTTP requests through FastAPI |
| `benchmark.py` | Benchmarks CPU and GPU inference |
| `gpt_best.pt` | Best-performing trained model weights |
| `frontend/` | Interactive web application |

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/CJThomasDev/ShakespeareGPT.git
cd ShakespeareGPT
```

### 2. Install Dependencies

Install Python and the required packages.

```bash
pip install -r requirements.txt
```

### 3. Start the Backend

```bash
python -m uvicorn api:app --reload
```

The API runs locally at:

`http://127.0.0.1:8000`

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

### 4. Launch the Frontend

Open `frontend/index.html` in a web browser.

With the backend running, the interface can send generation requests to the model.

## Key Takeaways

Building ShakespeareGPT provided hands-on experience with:

- Implementing Transformer components without pretrained models
- Understanding causal self-attention and autoregressive prediction
- Training neural networks using GPU acceleration
- Monitoring validation performance and identifying overfitting
- Comparing model architectures using cross-entropy and perplexity
- Saving and loading trained model checkpoints
- Building an inference API with FastAPI
- Connecting a machine learning model to an interactive frontend

## Author

**CJ Thomas**

Computer Science — Florida State University

GitHub: [CJThomasDev](https://github.com/CJThomasDev)

Portfolio: [cj-thomas.com](https://cj-thomas.com)