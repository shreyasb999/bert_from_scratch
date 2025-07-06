class BertConfig:
    """
    Minimal configuration.  Default sizes are trimmed
    (hidden=256, 4 heads, 4 layers) so the model trains fast
    on CPU or small GPU.
    """
    def __init__(
        self,
        vocab_size: int = 30_522,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_position_embeddings: int = 512,
        intermediate_size: int = 512,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.dropout = dropout
