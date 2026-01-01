# GPT-2 From Scratch

A culmination of everything I've learned in Andrej Karpathy's makemore series and build nanoGPT videos on YouTube.

A high-performance implementation of GPT-2 (124M parameters) from scratch in PyTorch, trained on the FineWeb-Edu dataset with modern optimizations including RoPE (rotary positional embeddings).


### 1. Prepare Dataset
```bash
python fineweb.py
```
Downloads and tokenizes FineWeb-Edu into sharded format.

### 2. Train Model
```bash
# Single GPU
python train.py

# Multi-GPU with DDP
torchrun --standalone --nproc_per_node=2 train.py
```

### 3. Generate Text
```bash
python inference.py
```
Edit `model_path` to point to your checkpoint.

## Model Configuration

```python
GPTConfig(
    block_size=1024,      # Sequence length
    vocab_size=50304,     # Vocabulary size
    n_layer=12,           # Transformer blocks
    n_head=12,            # Attention heads
    n_embd=768            # Embedding dimension
)
```

**Total Parameters**: ~124M

## Training Details

- **Batch size**: 524,288 tokens (2^19)
- **Micro batch**: 8 sequences × 1024 tokens
- **Learning rate**: 6e-4 with 715-step warmup + cosine decay
- **Max steps**: 19,073 
- **Optimizer**: Fused AdamW with weight decay

## Performance Optimizations

- `torch.compile()`: model kernel optimizations
- Fused AdamW: Optimized kernel on CUDA
- Gradient accumulation: Larger effective batch sizes
- KV-cache: O(n) inference instead of O(n²)
- Multiprocessing: Parallel dataset tokenization

## Checkpoints

Models saved every 1000 steps as `gpt2-checkpoint-{step}.pt` containing:
- Model state dict
- Optimizer state dict
- Training step
- Validation loss

## Hardware

- **Training**: 2xT4 Kaggle GPU

## Key Implementation Details

- **RoPE Embeddings**: Better extrapolation to longer sequences
- **KV_Caching**: Faster model inference

## References

- [Let's Reproduce GPT2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Makemore series](https://www.youtube.com/watch?v=l8pRSuU81PU)

