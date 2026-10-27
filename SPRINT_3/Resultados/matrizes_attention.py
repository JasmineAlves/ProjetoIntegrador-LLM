import os
import sys

# Adiciona a pasta raiz do projeto ao caminho de importações do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch
from Notebooks.multi_head_attention_class import MultiHeadAttention

# 8 tokens, cada token possui 8 números
inputs = torch.tensor([
    [0.43, 0.15, 0.89, 0.32, 0.71, 0.24, 0.56, 0.18],
    [0.55, 0.87, 0.66, 0.21, 0.34, 0.92, 0.47, 0.63],
    [0.57, 0.85, 0.64, 0.44, 0.29, 0.76, 0.38, 0.81],
    [0.22, 0.58, 0.33, 0.71, 0.65, 0.19, 0.84, 0.42],
    [0.77, 0.25, 0.10, 0.92, 0.48, 0.36, 0.73, 0.57],
    [0.05, 0.80, 0.55, 0.63, 0.91, 0.27, 0.44, 0.69],
    [0.61, 0.39, 0.78, 0.14, 0.53, 0.88, 0.31, 0.72],
    [0.34, 0.68, 0.23, 0.95, 0.17, 0.59, 0.82, 0.46],
])

# Adiciona o tamanho do batch 1 × 8 × 8
batch = inputs.unsqueeze(0)

# Criação do modelo
torch.manual_seed(123)
mha = MultiHeadAttention(
    d_in=8, d_out=8, context_length=8, dropout=0.0, num_heads=4
)

# Transforma os dados em Query, Key e Value
queries = mha.W_query(batch)
keys = mha.W_key(batch)
values = mha.W_value(batch)

# Cada head terá 8 dimensões de saída ÷ 4 heads = 2 dimensões
# 1 × 8 × 4 × 2
queries = queries.view(1, 8, 4, 2)
keys = keys.view(1, 8, 4, 2)

# Coloca o número de heads antes dos tokens
# 1 × 4 × 8 × 2
queries = queries.transpose(1, 2)
keys = keys.transpose(1, 2)

# Calcula os scores de atenção
# 1 batch × 4 heads × 8 tokens × 8 tokens
attn_scores = queries @ keys.transpose(2, 3)

# Pega a máscara causal para os 8 tokens e aplica -inf
mask = mha.mask.bool()[:8, :8]
attn_scores = attn_scores.masked_fill(mask, -torch.inf)

# Escala os scores dividindo pela raiz da dimensão de cada head (head_dim = 2)
attn_scores = attn_scores / (keys.shape[-1] ** 0.5)

# Softmax transforma os scores em probabilidades (cada linha soma 1)
attn_weights = torch.softmax(attn_scores, dim=-1)

# MATRIZES DE ATENÇÃO ------------------------------------------------------------------------------------------------

attention_matrices = attn_weights[0]  # Pega o primeiro exemplo do batch

# Verificação numérica no terminal para comprovar que os valores variam entre heads
print("--- DIFERENÇA MÉDIA NUMÉRICA ENTRE AS HEADS ---")
print(
    "Head 1 vs Head 2:",
    torch.abs(attention_matrices[0] - attention_matrices[1]).mean().item(),
)
print(
    "Head 1 vs Head 3:",
    torch.abs(attention_matrices[0] - attention_matrices[2]).mean().item(),
)
print(
    "Head 1 vs Head 4:",
    torch.abs(attention_matrices[0] - attention_matrices[3]).mean().item(),
)

tokens = [f"Token {i+1}" for i in range(8)]

for head in range(4):
    matrix = attention_matrices[head].detach().numpy()
    plt.figure(figsize=(8, 7))
    # Fixa vmin=0 e vmax=1 para manter uma escala visual justa entre todas as heads
    im = plt.imshow(
        matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest"
    )
    plt.colorbar(im, label="Peso de atenção")

    # Desenha o valor numérico exato dentro de cada célula do gráfico
    for i in range(8):
        for j in range(8):
            val = matrix[i, j]
            if val > 0.0001:  # Exibe apenas posições permitidas pela máscara
                plt.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=8,
                )

    # eixos
    plt.xticks(range(8), tokens, rotation=45)
    plt.yticks(range(8), tokens)
    plt.xlabel("Token observado")
    plt.ylabel("Token atual")
    plt.title(f"Matriz de Atenção - Head {head + 1}")

    plt.tight_layout()
    plt.show()