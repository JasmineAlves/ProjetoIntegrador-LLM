import torch
import torch.nn as nn
from normalization_class import LayerNorm

# Implementa uma camada de rede neural com 5 entradas e 6 saídas aplicada em 2 exemplos de entrada
torch.manual_seed(123)
# Cria 2 exemplos de treinamento com 5 dimensões (caracteristicas) cada.
batch_example = torch.randn(2, 5)
# Consiste em uma camada linear seguida por uma função de ativação não-linear (ReLU) que garante que não tenha numeros negativos
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# Examina a média e a variância das saídas
# keepdim=True garante que o tensor de saída contem o mesmo numero de dimensões que o tensor de entrada,
# sem ele o tensor iria retornar um vetor de 2 dimensões ao invés de de uma matriz 2x1
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
# Primeira linha é o valor da média da primeira entrada e a segunda linha da segunda entrada
print("Mean:\n", mean)
# Mesmo vale para a variancia
print("Variance:\n", var)

# Aplicar camada de normalização na camada de saídas
# Subtrai-se a média e divide pela raiz quadrada na variância/desvio padrão
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
# Pode-se observar que contém 0 de média e variância de 1, ainda com números negativos
print("Normalized layer outputs:\n", out_norm)
torch.set_printoptions(sci_mode=False) # Desliga a notação cientifica
print("Mean:\n", mean)
print("Variance:\n", var)

# Aplicar a classe de normalização no lote de entrada
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)