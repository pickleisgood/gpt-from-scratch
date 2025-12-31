from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import time
from torch.amp import autocast, GradScaler
import math
import inspect
import os

### CHANGES TO RESUME TRAINING:
# load model and optimizer state dicts
# change the range of steps (e.g. for steps in range(2000, 4500))
# change starting shard of dataloader

### Speed up changes
# Used torch.float16 (bfloat16 for A-series GPU) instead of float32
# Changed vocab size from 50265 -> 50304 (nice number)
# torch.compile(model)
# used scaled_dot_product_attention (flash attention)
# fused AdamW optimizer
# set matmul precision to high instead of highest (new GPUs)
# DistributedDataParallel (DDP) for multiple GPUs

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def apply_rope(self, x, cached_cos, cached_sin, idx=None):
        x = x.contiguous()
        B, nh, T, hs = x.shape
        
        # get even and odd indices representing x and y vector components
        xs = x[..., 0::2] #(B, nh, T, hs/2) 
        ys = x[..., 1::2] #(B, nh, T, hs/2)

        # if we are training, index of x is not required
        if idx is None:
            cos = cached_cos[...,:T,:]
            sin = cached_sin[...,:T,:]

        # During inference, since we are using cached_kv, we only pass in one token as x
        # Because of this, the idx of x is necessary for relative distance
        else:
            cos = cached_cos[...,idx:idx+1,:]
            sin = cached_sin[...,idx:idx+1,:]

        # RoPe formula
        out_xs = xs * cos - ys * sin
        out_ys = xs * sin + ys * cos

        out = torch.empty_like(x)
        
        # Write results directly into the even/odd slices
        # This skips the expensive 'stack' + 'flatten' memory reshuffling
        out[..., 0::2] = out_xs
        out[..., 1::2] = out_ys
        
        return out
        
    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_size = C // self.n_head

        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        # manually set RoPe idx if kv_cache is present
        # mus get length of time dimention of element in kv_cache
        idx = kv_cache[0].shape[2] if kv_cache is not None else None
        
        # apply RoPe
        k = self.apply_rope(k, cos, sin, idx)
        q = self.apply_rope(q, cos, sin, idx)

        if kv_cache:
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=2)
            v = torch.cat((v_cache, v), dim=2)

        new_kv_cache = (k.detach(), v.detach())

        # only causal mask if no kv_cache
        y = F.scaled_dot_product_attention(q, k, v, is_causal=kv_cache is None) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y, new_kv_cache

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, kv_cache=None):
        out, new_kv_cache = self.attn(self.ln_1(x), cos, sin, kv_cache)
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # iterate through params
        self.apply(self._init_weights)

        cos, sin = self.compute_rope()
        self.register_buffer('cached_cos', cos)
        self.register_buffer('cached_sin', sin)

    def compute_rope(self, theta=10000):
        head_size = self.config.n_embd // self.config.n_head
        
        # RoPe embedding
        pos = torch.arange(self.config.block_size).float() # get all token positions 

        # RoPe angle rotation formula
        scale = torch.arange(0, head_size, 2).float() / head_size
        inv_freq = 1 / theta ** scale

        # every position rotates a different amount
        freqs = torch.einsum("i, j -> i j", pos, inv_freq) # (T, hs/2)
        freqs = freqs.view(1, 1, self.config.block_size, head_size//2) # (1, 1, T, hs/2)

        return torch.cos(freqs), torch.sin(freqs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
          std = 0.02
          if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5

          torch.nn.init.normal_(module.weight, mean=0.0, std=std)
          if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb 

        # store new kv_cache
        new_kv_cache = []

        # forward the blocks of the transformer
        # Use kv_cache if not None
        for i, block in enumerate(self.transformer.h):
            cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = block(x, self.cached_cos, self.cached_sin, kv_cache=cache)
            new_kv_cache.append(new_cache)
            
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_kv_cache
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
          {'params': decay_params, 'weight_decay': weight_decay},
          {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        if master_process:
          num_decay_params = sum(p.numel() for p in decay_params)
          num_nodecay_params = sum(p.numel() for p in nodecay_params)
          print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
          print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        if master_process:
          print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
if __name__ == "__main__":
  import numpy as np

  def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

  class DataLoaderLite:
      def __init__(self, B, T, process_rank, num_processes, split):
          self.B = B
          self.T = T
          self.process_rank = process_rank
          self.num_processes = num_processes
          assert split in {'train', 'val'}

          # get the shard filenams
          data_root = "edu_fineweb10B"
          shards = os.listdir(data_root)
          shards = [s for s in shards if split in s]
          shards = sorted(shards)
          shards = [os.path.join(data_root, s) for s in shards]
          self.shards = shards
          assert len(shards) > 0, f"no shards found in split: {split}"
          if master_process:
            print(f"found {len(shards)} shards for split {split}")

          self.current_shard = 0
          self.tokens = load_tokens(self.shards[self.current_shard])
          self.current_position = self.B * self.T * self.process_rank
          self.reset()

      def reset(self):
          # state, init at shard zero
          self.current_shard = 0
          self.tokens = load_tokens(self.shards[self.current_shard])
          self.current_position = self.B * self.T * self.process_rank

      def next_batch(self):
          B, T = self.B, self.T
          buf = self.tokens[self.current_position : self.current_position+B*T+1]
          x = buf[:-1].view(B, T) # inputs
          y = buf[1:].view(B, T) # targets
          # advance position
          self.current_position += B * T * self.num_processes
          # if loading next batch is out of bounds reset
          if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
              self.current_shard = (self.current_shard + 1) % len(self.shards)
              self.tokens = load_tokens(self.shards[self.current_shard])
              self.current_position = self.B * self.T * self.process_rank

          return x, y

  # simple launch:
  # python "name".py
  # DDP launch for multiple GPUs:
  # torchrun --standalone --nproc_per_node=8 "name".py

  # run the training loop
  import torch.distributed as dist
  from torch.distributed import init_process_group, destroy_process_group
  from torch.nn.parallel import DistributedDataParallel as DDP

  # Set of DDP (distributed data parallel)
  # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

  ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
  if ddp:
    # use of DDP atm demans CUDA, set device appriopriately according to rank
    assert torch.cuda.is_available(), "for now, need CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    if master_process:
        print(f"using DDP on {ddp_world_size} GPUs", flush=True)

  else:
    #vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #attempt to autudetect device
    device = 'cpu'
    if torch.cuda.is_available():
      device = 'cuda'
    elif torch.backends.mps.is_available():
      device = 'mps'
    print(f'using device: {device}')


  total_batch_size = 524288 # 2**19
  B = 8 # micro batch size
  T = 1024 # sequence length
  assert total_batch_size % (B * T * ddp_world_size) == 0, 'make sure total_batch_size is divisible by b*T*ddp_world_size'
  grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
  if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation step: {grad_accum_steps}")

  train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
  val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

  #torch.set_float32_matmul_precision('high')
  # this does not work for t4 gpu

  torch.manual_seed(1337)
  torch.cuda.manual_seed(1337)

  # create model
  model = GPT(GPTConfig(vocab_size=50304)).to(device)

  # compile for faster training
  model = torch.compile(model)
  if master_process:
      print("model compiled")
  if ddp:
    # ddp averages gradients when backward pass is done
    model = DDP(model, device_ids=[ddp_local_rank])
  raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

  max_lr = 6e-4
  min_lr = max_lr * 0.1
  warmup_steps = 715 # gpt paper value
  max_steps = 19073 # gpt paper value
  def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_steps:
      return max_lr * (it+1)/warmup_steps

    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
      return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

  optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

  scaler = GradScaler()

  for step in range(max_steps):
      t0 = time.time()
      
      if step % 100 == 0:
        model.eval()
        #val_loader.reset()
        with torch.inference_mode():
          val_loss_accum = 0.0
          val_loss_steps = 20
          for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
              logits, loss, _ = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
        if ddp:
          dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
          print(f"validation loss: {val_loss_accum.item():.4f}")

      model.train()
      optimizer.zero_grad(set_to_none=True)
      loss_accum = 0.0
      for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16): #change to bfloat16 if gpu allows
          logits, loss, _ = model(x, y)

        # scale the loss to account for gradient accumulation
        # since we want the MEAN loss not the SUM

        # 1. Scale the loss (Prevents Underflow)
        loss = loss / grad_accum_steps
        # detach tensor from graph
        loss_accum += loss.detach()

        # do not sync GPUs unless final step
        if ddp:
          model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        scaler.scale(loss).backward()

      # set average loss_accum across all GPU ranks
      if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

      # 2. Unscale and Clip (Prevents Overflow/NaNs)
      scaler.unscale_(optimizer)
      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      # 3. Step with Scaler
      lr = get_lr(step)
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr

      scaler.step(optimizer)
      scaler.update()
      torch.cuda.synchronize()
      t1 = time.time()
      dt = (t1 - t0)
      tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
      tokens_per_sec = tokens_processed / dt
      if master_process:
        print(f"step: {step:4d} | loss: {loss_accum.item():.5f} | lr = {lr:.4e} | norm: {norm:.4f} | dt: {(dt*1000):.2f}ms | tok/sec:{tokens_per_sec:.2f}")

      if master_process and (step % 1000 == 0 and step > 0 or step == max_steps - 1):
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'val_loss': val_loss_accum.item()
        }
        torch.save(checkpoint, f"gpt2-checkpoint-{step}.pt")
        print(f"Saved checkpoint at step {step}")

  if ddp:
    destroy_process_group()
    
