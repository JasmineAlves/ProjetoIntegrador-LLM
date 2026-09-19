# dropout para prevenir overfitting
import torch

torch.manual_seed(123)

# taxa de 50%, ou seja, mascarar metade dos pesos de atenção
dropout = torch.nn.Dropout(0.5)
# cria matriz 6x6 de 1's
example = torch.ones(6, 6)
# metade dos valores será 0
# o resto é o número 2 porque para compensar a redução de elementos, os valores
# que sobraram na matriz são aumentados pelo fator de 1/0.5 = 2
# Isso é crucial para manter o equilíbrio geral dos pesos de atenção
print(dropout(example))