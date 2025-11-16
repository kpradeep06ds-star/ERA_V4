import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, context_length):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", 
                                                      model_max_length=context_length)
        
        # Read and tokenize text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize text with truncation
        tokens = []
        # Process text in chunks to avoid memory issues
        chunk_size = 1000000  # Process 1M characters at a time
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunk_tokens = self.tokenizer.encode(chunk, 
                                               truncation=True,
                                               max_length=context_length)
            tokens.extend(chunk_tokens)
            
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        
        # Create chunks of context_length size
        n = len(self.tokens)
        self.chunks = [(i, min(i + context_length, n)) 
                      for i in range(0, n - context_length, context_length)]
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        # Get chunk start and end indices
        start, end = self.chunks[idx]
        
        # Get input sequence and target
        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1]  # Shift by 1 for next token prediction
        
        # Pad if necessary
        if len(x) < self.context_length:
            padding = self.context_length - len(x)
            x = torch.cat([x, torch.zeros(padding, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(padding, dtype=torch.long)])
            
        # Ensure lengths are correct
        x = x[:self.context_length]
        y = y[:self.context_length]
            
        return x, y 