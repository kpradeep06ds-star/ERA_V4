class ModelConfig:
    def __init__(self):
        # Model architecture
        self.vocab_size = 32768  # Power of 2 for efficiency
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 2048
        self.max_position_embeddings = 2048
        
        # Training
        self.batch_size = 4
        self.context_length = 64  # Reduced to handle memory better
        self.learning_rate = 3e-4
        
        # Optimization
        self.use_flash_attention = True
        self.use_mixed_precision = True 