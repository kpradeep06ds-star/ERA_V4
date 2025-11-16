# SmolLM2 Training Project

This project implements training for a small language model based on the SmolLM2 architecture. The model is trained on text data with mixed-precision training and gradient scaling for efficient training on GPU.

## Project Structure 

```
project_root/
├── config/
│ └── model_config.py # Model and training configuration
├── model/
│ ├── attention.py # Attention mechanism implementation
│ ├── mlp.py # Multi-layer perceptron implementation
│ ├── transformer.py # Transformer block implementation
│ └── smolLM2.py # Main model architecture
├── training/
│ └── dataset.py # Dataset handling and processing
├── utils/
│ ├── checkpoint.py # Checkpoint saving and loading
│ └── generation.py # Text generation utilities
├── main.py # Training script
└── input.txt # Training data
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

- Initial loss: 10.5829
- Final loss: ~0.0000
- Training speed: ~19,000-21,000 tokens/sec at peak
- Total training time: ~2 minutes 16 seconds for 5000 steps

### Training Progress
- Step 0: Loss = 10.5829
- Step 500: Loss = 0.0004
- Step 1000: Loss = 0.0001
- Step 1500: Loss = 0.0001
- Step 2000: Loss = 0.0001
- Step 2500: Loss = 0.0000
- Step 3000: Loss = 0.0000
- Step 3500: Loss = 0.0000
- Step 4000: Loss = 0.0000
- Step 4500: Loss = 0.0000
- Step 5000: Loss = 0.0000

## Model Configuration
- Vocabulary size: 32,768
- Hidden size: 768
- Number of layers: 12
- Number of attention heads: 12
- Intermediate size: 2,048
- Maximum position embeddings: 2,048
- Context length: 64
- Batch size: 4
- Learning rate: 3e-4

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

[Logs](https://github.com/pradeep6kumar/SMOL135/blob/main/log.txt)


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
- [Hugging Face Model Repository](https://huggingface.co/spaces/pradeep6kumar2024/smollm2-spaces/tree/main)
- [Hugging Face Spaces Demo](https://huggingface.co/spaces/pradeep6kumar2024/smollm2-spaces)

To use the model from Hugging Face:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/SmolLM2-trained")
tokenizer = AutoTokenizer.from_pretrained("your-username/SmolLM2-trained")
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
