# Inicializar o módulo multi-head com o mesmo número de attention heads e tamanho do emebeddings de entrada e saída do GPT-2 model
import torch
# Importa a classe MultiHeadAttention que já foi criada
from multi_head_attention_class import MultiHeadAttention

# Tamanho do vetor de entrada
# No GPT-2, cada token é representado por 768 números
d_in = 768
# Tamanho do vetor de saída também é 768 no GPT-2
d_out = 768

# Quantidade máxima de tokens que o modelo consegue considerar de uma vez
context_length = 1024

# Quantidade de heads de atenção
num_heads = 12

# Cria o módulo de Multi-Head Attention
mha = MultiHeadAttention(
    d_in,
    d_out,
    context_length,
    dropout=0.0,
    num_heads=num_heads
)

# Mostra o módulo criado
print(mha)