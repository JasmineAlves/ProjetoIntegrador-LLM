GPT_CONFIG_124M = {
 "vocab_size": 50257, # Vocabulary size (Quantas palavras conhece)
 "context_length": 1024, # Context length (Quant. de tokens que o modelo consegue considerar de uma vez, entrada e saída)
 "emb_dim": 768, # Embedding dimension (Quant. de números usados para representar cada token)
 "n_heads": 12, # Number of attention heads (Em cada bloco transformer, analisa-se X aspectos diferentes da frase para multi-head attention)
 "n_layers": 12, # Number of layers (Quant. de blocos transformers que o modelo possui)
 "drop_rate": 0.1, # Dropout rate (Intensidade do dropout para evitar overfitting, 10%)
 "qkv_bias": False # Query-Key-Value bias (Se as camadas usam um vetor de viés em Q K V)
}