import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import Block

class SmolLM2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # token embedding table
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        # position embedding table
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        
        # final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # Create position indices and clamp them to max_position_embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        pos = pos.expand(b, -1)  # shape (b, t)
        pos = torch.clamp(pos, max=self.config.max_position_embeddings - 1)
        
        # Get embeddings
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (b, t, n_embd)
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        # Get logits and loss
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            
        return logits, loss 