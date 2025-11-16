import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=False)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 