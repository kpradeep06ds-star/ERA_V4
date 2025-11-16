import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        # key, query, value projections
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size
        self.use_flash_attention = config.use_flash_attention
        
        # Add causal mask bias for non-flash attention
        if not self.use_flash_attention:
            self.register_buffer(
                "bias", 
                torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
                .view(1, 1, config.max_position_embeddings, config.max_position_embeddings)
            )
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        
        # calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        if self.use_flash_attention:
            # Use PyTorch's scaled_dot_product_attention with causal mask
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y 