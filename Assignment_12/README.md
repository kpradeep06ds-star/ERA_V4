
---

# 📘 **README — Shakespeare Memorized GPT (124M Decoder-Only Model)**

*A GPT-2-class Transformer trained to intentionally overfit Shakespeare.*

---

# 🧠 Overview

This repository contains a **decoder-only Transformer model (~124M parameters)** trained on the complete Shakespeare corpus (`input.txt`).
The goal of this project was:

* To **implement a GPT-2–style model from scratch**
* To **train it with controlled overfitting** so it memorizes Shakespeare
* To provide a **local + HF Spaces app** that generates text in Shakespearean style

This is *not* meant to be a general-purpose language model — it is deliberately allowed to overfit for demonstration, research, and educational purposes.

---

# 🧩 Model Architecture

The architecture follows the classic **GPT-2 Small (124M)** configuration:

| Component                | Value                 |
| ------------------------ | --------------------- |
| Layers (n_layer)         | **12**                |
| Attention Heads (n_head) | **12**                |
| Embedding Dim (n_embd)   | **768**               |
| Block Size               | **1024 tokens**       |
| Vocab Size               | **50257 (GPT-2 BPE)** |
| Parameters               | **124.4M**            |

### ✔️ Components Implemented

Implementation includes:

* **Causal self-attention** (masked, scaled dot-product)
* **Feedforward MLP with GELU activation**
* **LayerNorm pre-norm strategy**
* **Weight tying between WTE and LM head**
* **GPT-style residual scaling for stability**
* **Naive dropout removed** (because we *want* overfitting)
* **Gradient checkpointing** for memory efficiency
* **Option for torch.compile (disabled for Triton issues)**

Architecture source: [`transformer.py`](transformer.py)
Key model definition at **GPT**, **Block**, **CausalSelfAttention**
(see ).

---

# 🏋️‍♂️ Training Details

Training was done using a custom script (`train_mem.py`) that implements:

* **Gradient accumulation**
* **Warmup + Cosine LR schedule**
* **AdamW (β1=0.9, β2=0.95)**
* **Gradient checkpointing**
* **Automatic checkpoint saving**
* **Custom text windowing with stride**
* **Validation splits (10%)**

Training command (example):

```bash
python train_mem.py \
    --input input.txt \
    --out out_124m_mem \
    --n_layer 12 --n_head 12 --n_embd 768 \
    --block_size 1024 \
    --batch_size 4 --accum_steps 8 \
    --lr 3e-4 --warmup_steps 1000 --weight_decay 0.1 \
    --grad_ckpt \
    --val_ratio 0.1 --eval_every 1000 \
    --epochs 50
```

(see )

### 📝 Dataset

Only **Shakespeare’s complete works** (`input.txt`).
No cleaning. No augmentation. Pure memorization objective.

### 🎯 Objective

We intentionally allowed the model to **overfit** to verify:

* Convergence of 124M-param GPT on tiny dataset
* Reproduction of GPT-2-style memorization behavior
* Smooth training dynamics + token-level loss
* Ability to run full training on a single RTX 4090 (24GB)

---

# 📦 Checkpoints

The repository includes:

```
out_124m_mem/
    └── model.pt        (≈535 MB)
```

`model.pt` contains:

* `model_state_dict`
* Tokenizer merges + vocab mapping
* Model config dict (`n_layer`, `n_embd`, etc.)
* Block size
* Training metadata

This allows the Gradio app to reconstruct the model exactly.

---

# 🖥️ Running the Model Locally

```bash
pip install -r requirements.txt
python app.py
```

Runs a local Gradio app where you can:

* Provide a prompt
* Select number of tokens to generate
* Adjust temperature & top-k

---

# 🚀 HuggingFace Spaces Deployment

HF Space files:

```
app.py
transformer.py
train_mem.py   (optional)
requirements.txt
out_124m_mem/model.pt
README.md
```

HuggingFace will automatically:

* Create environment
* Install dependencies
* Run the app
* Serve the model via Web UI

HuggingFace Space :

🔗 [https://huggingface.co/spaces/justpradeep/shakespeare](https://huggingface.co/spaces/justpradeep/shakespeare)

Model File:

🔗 [model](https://huggingface.co/spaces/justpradeep/shakespeare/blob/main/out_124m_mem/model.pt)

Screenshot:

🔗 [Screenshot Working](./out_124m_mem/working_screenshot.png)
---

# ✨ Why Overfitting?

Because the purpose of this project is **demonstration**, not generalization.

Overfitting makes it:

* A perfect Shakespeare text generator
* A reproducible nano-GPT pipeline
* A study model for inference, sampling, and Transformer mechanics
* A baseline for experimenting with pruning, quantization, distillation

---

# 🔮 Future Enhancements

* Convert model to **FP16** to reduce file size from 535MB → 260MB
* Add top-p sampling
* Add ability to stream tokens
* Upload tokenizer + weights to HuggingFace model hub separately
* Possibly train a 350M version

---

# 🏁 Final Notes

This project demonstrates:

✔ Building GPT-2 architecture from scratch
✔ Training a 124M decoder model end-to-end
✔ Efficient memory usage with gradient checkpointing
✔ Running on consumer GPU
✔ Packaging into an interactive HF Space

# Logs

[Training Log](./out_124m_mem/train.log)
