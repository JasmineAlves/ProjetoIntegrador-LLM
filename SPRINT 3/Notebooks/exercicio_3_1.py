# Comparação entre classes
import torch
from self_attention_class import SelfAttention_v1
from self_attention_class import SelfAttention_v2

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your    (x¹)
        [0.55, 0.87, 0.66],  # journey (x²)
        [0.57, 0.85, 0.64],  # starts  (x³)
        [0.22, 0.58, 0.33],  # with    (x⁴)
        [0.77, 0.25, 0.10],  # one     (x⁵)
        [0.05, 0.80, 0.55]   # step    (x⁶)
    ]
)

d_in = inputs.shape[1]
d_out = 2 
torch.manual_seed(123)

# Instância da SelfAttention_v2
attention_v2 = SelfAttention_v2(d_in, d_out)
# Instância da SelfAttention_v1
attention_v1 = SelfAttention_v1(d_in, d_out)

# TRANSFERINDO OS PESOS DA v2 PARA A v1
# O nn.Linear da v2 guarda os pesos em uma orientação diferente da matriz usada diretamente na v1
# Na v2 é d_out x d_in. Enquanto a v1 espera é d_in x d_out
# Por isso usamos .T para TRANSPOR a matriz
attention_v1.W_query.data = attention_v2.W_query.weight.data.T

# Fazemos a mesma coisa para os pesos das Keys.
attention_v1.W_key.data = attention_v2.W_key.weight.data.T
# E também para os pesos dos Values.
attention_v1.W_value.data = attention_v2.W_value.weight.data.T

# Podemos passar a mesma entrada pelas duas versões.
context_v1 = attention_v1(inputs)
context_v2 = attention_v2(inputs)

# Como as duas usam os mesmos pesos e fazem os mesmos cálculos os resultados devem ser iguais 
# printa True
print(torch.allclose(context_v1, context_v2))