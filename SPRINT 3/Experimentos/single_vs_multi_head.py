import os
import sys
# Adiciona a pasta raiz do projeto ao caminho de importações do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from Notebooks.multi_head_attention_class import MultiHeadAttention


# 8 tokens. Cada token possui 8 números
# 8 tokens × 8 dimensões

inputs = torch.tensor([
    [0.43, 0.15, 0.89, 0.32, 0.71, 0.24, 0.56, 0.18],  # Token 1
    [0.55, 0.87, 0.66, 0.21, 0.34, 0.92, 0.47, 0.63],  # Token 2
    [0.57, 0.85, 0.64, 0.44, 0.29, 0.76, 0.38, 0.81],  # Token 3
    [0.22, 0.58, 0.33, 0.71, 0.65, 0.19, 0.84, 0.42],  # Token 4
    [0.77, 0.25, 0.10, 0.92, 0.48, 0.36, 0.73, 0.57],  # Token 5
    [0.05, 0.80, 0.55, 0.63, 0.91, 0.27, 0.44, 0.69],  # Token 6
    [0.61, 0.39, 0.78, 0.14, 0.53, 0.88, 0.31, 0.72],  # Token 7
    [0.34, 0.68, 0.23, 0.95, 0.17, 0.59, 0.82, 0.46]   # Token 8
])

# Sem notação cientifica nos valores dos tensores
torch.set_printoptions(sci_mode=False)

# A MultiHeadAttention 
# batch × tokens × dimensões
# unsqueeze(0) adiciona o tamanho do batch 1 × 8 × 8
batch = inputs.unsqueeze(0)

print("Entrada:")
print(batch.shape)



# SINGLE-HEAD -------------------------------------------------------------------

# 1 head, 8 dimensões por head
torch.manual_seed(123)

single_head = MultiHeadAttention(
    d_in=8,
    d_out=8,
    context_length=8,
    dropout=0.0,
    num_heads=1
)
output_single = single_head(batch)


print("SINGLE-HEAD ATTENTION")
print("Resultado:")
print(output_single)
print("\nFormato:")
print(output_single.shape)



# MULTI-HEAD COM 2 HEADS -------------------------------------------------------------------------

# Cada head recebe 8 ÷ 2 = 4 dimensões
torch.manual_seed(123)
multi_head_2 = MultiHeadAttention(
    d_in=8,
    d_out=8,
    context_length=8,
    dropout=0.0,
    num_heads=2
)
output_multi_2 = multi_head_2(batch)


print("MULTI-HEAD - 2 HEADS")
print("Resultado:")
print(output_multi_2)
print("\nFormato:")
print(output_multi_2.shape)



# MULTI-HEAD COM 4 HEADS ---------------------------------------------------------------------------

# Cada head recebe 8 ÷ 4 = 2 dimensões
torch.manual_seed(123)
multi_head_4 = MultiHeadAttention(
    d_in=8,
    d_out=8,
    context_length=8,
    dropout=0.0,
    num_heads=4
)
output_multi_4 = multi_head_4(batch)


print("MULTI-HEAD - 4 HEADS")
print("Resultado:")
print(output_multi_4)
print("\nFormato:")
print(output_multi_4.shape)



# Comparando os formatos
print("\nCOMPARAÇÃO -----------------------------------------------------")
print("Entrada:       ", batch.shape)
print("1 head:        ", output_single.shape)
print("2 heads:       ", output_multi_2.shape)
print("4 heads:       ", output_multi_4.shape)



# Diferença média entre os resultados
# Mudar a quantidade de heads também muda os valores produzidos pela atenção
difference_1_2 = torch.mean(
    torch.abs(output_single - output_multi_2)
)
difference_2_4 = torch.mean(
    torch.abs(output_multi_2 - output_multi_4)
)

print("\nDiferença média entre 1 e 2 heads:")
print(difference_1_2)
print("\nDiferença média entre 2 e 4 heads:")
print(difference_2_4)