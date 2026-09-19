# Mascarar tokens futuros
import torch
from self_attention_class import SelfAttention_v2
# Classe generalizada do causal attention
from causal_attention_class import CausalAttention

# Classe de multi-head attention (várias single-head attention)
from multi_head_attention_class import MultiHeadAttentionWrapper

# Classe de multi-head attention mais eficiente
from multi_head_attention_class import MultiHeadAttention

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

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# Usando a classe de self-attention com pesos treináveis mais eficiente
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# Calcular os pesos de atenção usando softmax
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# Cria mascara onde os valores acima da diagonal são zero, usa função tril
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
# matriz de 1 e 0
print(mask_simple)

# multiplica a matriz de mascara pelos pesos de atenção, onde é 0 continuara sendo 0
masked_simple = attn_weights*mask_simple
print(masked_simple)

# Renormalizar os pesos de atenção para somarem 1 em cada linha
# divide-se cada elemento em cada linha pela soma em cada linha
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
# Pesos acima da diagonal estão zerados
print(masked_simple_norm)

# Pesos de atenção mascarados mais eficientes, trocamos os 0 por valores negativos infinitos
# softmax trata eles como 0
# colocamos 1's na diagonal de cima e "multiplicamos" por -inf
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
# aplicamos softmax e onde era -inf volta ser 0
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# dropout nos pesos de atenção
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
# resultado pode ser diferente dependendo do sistema operacional
print(dropout(attn_weights))

# verificar se o código consegue trabalhar com várias frases ao mesmo tempo
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 



# Utilizando a classe do causal attention
# Pega a quantidade de tokens de cada texto no batch
# batch.shape[1] representa a dimensão dos tokens
# Exemplo: batch.shape = [2, 6, 3]
context_length = batch.shape[1]

# Cria o objeto de atenção causal
# d_in = tamanho do vetor de entrada de cada token
# d_out = tamanho do vetor que queremos na saída
# context_length = quantidade máxima de tokens
# 0.0 = dropout desligado
ca = CausalAttention(d_in, d_out, context_length, 0.0)

# Passamos o batch pela classe atenção causal
# O resultado é armazenado em context_vecs
context_vecs = ca(batch)

# Mostra o tamanho do resultado
# context_vecs.shape = [2, 6, 2]
# 2 textos → 6 tokens → cada token agora possui um vetor de 2 dimensões
print("context_vecs.shape:", context_vecs.shape)



# multi-head attention em sequência
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
 d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# multi-head attention em paralelo, multiplicação de matrizes apenas 1 vez
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape) # (num exemplo, num tokens, num vetor final)
