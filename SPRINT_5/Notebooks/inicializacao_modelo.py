import torch
from SPRINT_4.Notebooks.GPT_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    # Contexto mais curto para reduzir demanda computacional do modelo de treinamento e ser possivel rodar em um pc comum
    # Mas depois do treinamento vamos mudar para 1024 para os pesos já treinados
    "context_length":  256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()