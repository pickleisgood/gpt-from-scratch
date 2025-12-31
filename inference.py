import tiktoken
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from train import GPT, GPTConfig, Block, CausalSelfAttention, MLP

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f'using device: {device}')

# 1. Initialize the model properly (Note the parenthesis!)
model = GPT(GPTConfig()) 

# 2. Load the state dictionary
model_path = "YOUR MODEL PATH HERE"
state_dict = torch.load(model_path, map_location=device)

# 3. Fix the keys (Strip unwanted prefixes)
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "_orig_mod.")

# 4. Load the fixed state dict
model.load_state_dict(state_dict)
model.to(device) # Move to GPU after loading
model.eval()
print("Model loaded successfully!")

num_return_sequences = 5
max_length = 300

enc = tiktoken.get_encoding('gpt2')

tokens = enc.encode("The meaning of life is")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(dim=0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

kv_cache = None

while x.shape[1] < max_length:
    with torch.inference_mode():
        if kv_cache is not None:
            tok = x[:, -1].unsqueeze(dim=1)
        else:
            tok = x

        logits, loss, new_kv = model(tok, kv_cache=kv_cache)

        kv_cache = new_kv

        logits = F.softmax(logits, dim=-1)

        probs = logits[:,-1,:]

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, num_samples=1)

        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

