import torch
import torch.nn as nn
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Localiza a pasta raiz que contém SPRINT 3 e SPRINT 4
raiz = Path(__file__).resolve().parents[2]

# Adiciona a pasta onde está a classe MultiHeadAttention
sys.path.insert(
    0,
    str(raiz / "SPRINT_3" / "Notebooks")
)

from multi_head_attention_class import MultiHeadAttention
from GELU_class import FeedForward
from normalization_class import LayerNorm
from cfg import GPT_CONFIG_124M


class TransformerBlock(nn.Module):
 def __init__(self, cfg):
    super().__init__()
    # Cria multi-head attention que permite o modelo relacionar diferentes tokens do texto
    self.att = MultiHeadAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        context_length=cfg["context_length"],
        num_heads=cfg["n_heads"],
        dropout=cfg["drop_rate"],
        qkv_bias=cfg["qkv_bias"])
    # Cria uma rede que processa e transforma as informações de cada token
    self.ff = FeedForward(cfg)
    # Normalizam os valores para ajudar no treinamento estável
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    # Aplica dropout para reduzir overfitting
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
 
 def forward(self, x):
        # Guarda entrada original
        shortcut = x
        # Normaliza entrada
        x = self.norm1(x)
        # Aplica attention
        x = self.att(x)
        # Aplica dropout
        x = self.drop_shortcut(x)
        # Soma a entra original com o resultado do attention
        x = x + shortcut
        # Repete lógica com a rede FeedForward usando norm2
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        # Retorna resultado final do bloco
        return x

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)