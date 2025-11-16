# SmolLM2 Training Project

This project implements training for a small language model based on the SmolLM2 architecture. The model is trained on text data with mixed-precision training and gradient scaling for efficient training on GPU.

## Project Structure 

```
project_root/
├── config/
│   └── model_config.py         # Model + training hyperparameters
│
├── model/
│   ├── attention.py            # Attention mechanism
│   ├── mlp.py                  # Feed-forward / MLP block
│   ├── transformer.py          # Transformer Block
│   └── smolLM2.py              # SmolLM2 model definition
│
├── training/
│   └── dataset.py              # TextDataset and data loading
│
├── utils/
│   ├── checkpoint.py           # save_checkpoint / load_checkpoint
│   ├── generation.py           # CLI / script generation helpers
│   └── logger.py               # (if you added it) logging setup
│
├── datasets/
│   └── input.txt               # Training corpus (used by TextDataset)
│
├── logs/
│   └── training.log            # Logger output from main.py
│
├── main.py                     # Training + evaluation + checkpoint resume
└── README.md                   # Full explanation, param counts, etc.

```

## Features
- Mixed precision training with automatic gradient scaling
- Efficient tokenization with chunk processing
- Checkpoint saving and loading
- Text generation capabilities
- Progress monitoring with training metrics
- TF32 optimizations for NVIDIA GPUs

## Training Results
The model was trained for 5000 steps initially, followed by 50 additional fine-tuning steps. Key metrics:
Here you go, split into the three segments you asked for 👇

---

## 1️⃣ Training Results

* **Total training**:

  * Phase 1: **5000 steps** from scratch.
  * Phase 2: **resume from checkpoint** and train for **~50 additional steps** (5000–5040), confirming that checkpointing and continuing training work correctly. 
* **Final loss**:

  * By step 3000 onwards the loss is essentially **0.0000**, and it stays at that level through step 4500 and into the resumed 5000–5040 region. 
  * This means the model has **almost perfectly fit** the training data (strong memorisation).
* **Qualitative behaviour**:

  * Early on, sampled text is noisy, with random phrases and artifacts mixed into the prompt. 
  * From step 500 onward, the model consistently produces **clean, Shakespeare-style dialogue** starting from your prompt `"Once upon a time"`, showing strong copying/memorisation of the training corpus. 

---

## 2️⃣ Training Progress

* **Loss trajectory**:

  * Step 0: Loss ≈ **10.67** (random init). 
  * Step 500: Loss ≈ **0.0004**.
  * Step 1000: Loss ≈ **0.0002**.
  * Steps 1500–3000: Loss drops to ≈ **0.0001 → 0.0000**.
  * Steps 3500–4500: Loss remains at **0.0000**, showing the model has fully saturated on the dataset. 
  * Resumed steps 5000–5040: still **0.0000**, confirming that checkpoint restore worked correctly and training continued smoothly. 
* **Speed**:

  * Initial step: ~**1100 tokens/sec** (startup overhead).
  * After warm-up: stabilises around **15k–18k tokens/sec** for most of the run, with one dip to ~12.8k near step 4500. 
* **Sample evolution**:

  * Step 0 sample is mostly gibberish with broken words and random structure. 
  * By step 500+, the samples are coherent, repeatedly reproducing the same Shakespearean dialogue segment, which is exactly what you’d expect from a tiny-context LM overfitting a small dataset. 

So in short: **very fast convergence, heavy memorisation, smooth resume from checkpoint, and high tokens/sec throughput.**

---

## 3️⃣ Model Configuration (Smol135 Implementation)


* **Architecture**:

  * Decoder-only **Transformer language model**.
  * **12 transformer blocks** (layers).
  * **Model dimension**: 768.
  * **Attention heads**: 12 heads per layer.
  * **Feed-forward size**: 3072 (≈ 4 × d_model).
  * **Context length**: 64 tokens (training context window).
  * **Vocabulary size**: 49,152 tokens (using `HuggingFaceTB/SmolLM2-135M` tokenizer).
* **Training setup**:

  * Objective: **causal language modeling** (next-token prediction).
  * Loss: cross-entropy over logits, with padding index ignored.
  * Optimiser: AdamW (with typical small LR, weight decay).
  * Total training schedule:

    * 1st stage: 5000 steps from scratch.
    * 2nd stage: load checkpoint at 5000, continue for 50 more steps (to validate resume + checkpoint correctness).
* **Usage in HF Space**:

  * The **weights (`model.pt`) live in the model repo**: `justpradeep/smol135-finetuned`.
  * The **Space (`app.py`) reconstructs `SmolLM2(config)` locally**, then loads the state dict from that repo and exposes a 

## Usage
1. Install requirements:
```bash
pip install torch transformers tqdm
```

2. Prepare your input text file as `input.txt`

3. Run training:
```bash
python main.py
```

## Sample Generation
The model can generate text continuations. Example from training:

```
Prompt: "Once upon a time"
Generated: "Before we proceed any further, hear me speak.
First:
Speak, speak.
First Citizen:
First are all resolved rather to die than to famish?
All:
Resolved. resolved."
```

## Performance Optimizations
- Uses Flash Attention when available
- Mixed precision training (FP16)
- TF32 optimizations enabled
- Chunked text processing to handle large files
- Efficient data loading with PyTorch DataLoader

## Checkpoints
Checkpoints are saved at:
- Step 5000: `checkpoints/model_5000.pt`
- Final model: `checkpoints/model_final.pt`

## Logs

[Logs](./logs/training.log)


## Model Architecture Details

### Model Definition
The SmolLM2 model follows a standard transformer architecture with:
- Token embeddings (wte) and positional embeddings (wpe)
- 12 transformer blocks, each containing:
  - Multi-head self-attention with 12 heads
  - Feed-forward network with GELU activation
  - Layer normalization
- Final layer normalization and language model head

### Parameter Calculation

1. **Embeddings**:
   - Token embeddings: vocab_size × hidden_size = 32,768 × 768 = 25,165,824
   - Position embeddings: max_position × hidden_size = 2,048 × 768 = 1,572,864

2. **Each Transformer Block**:
   - Self-attention:
     - Q, K, V matrices: 3 × (hidden_size × hidden_size) = 3 × (768 × 768) = 1,769,472
     - Output projection: hidden_size × hidden_size = 768 × 768 = 589,824
   - Feed-forward:
     - First layer: hidden_size × intermediate_size = 768 × 2,048 = 1,572,864
     - Second layer: intermediate_size × hidden_size = 2,048 × 768 = 1,572,864
   - Layer norms: 2 × 2 × hidden_size = 2 × 2 × 768 = 3,072
   - Total per block: 5,508,096

3. **Final Layers**:
   - Final layer norm: 2 × hidden_size = 2 × 768 = 1,536
   - Language model head: hidden_size × vocab_size = 768 × 32,768 = 25,165,824

**Total Parameters**: ~93M parameters

## Model Links
- [Hugging Face Model Repository](https://huggingface.co/justpradeep/smol135-finetuned)
- [Hugging Face Spaces Demo](https://huggingface.co/spaces/justpradeep/smol135-app)

To use the model from Hugging Face:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/SmolLM2-trained")
tokenizer = AutoTokenizer.from_pretrained("your-username/SmolLM2-trained")
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
