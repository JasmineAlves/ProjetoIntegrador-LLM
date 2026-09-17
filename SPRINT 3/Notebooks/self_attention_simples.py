import torch


# Temos 6 palavras.
# Cada palavra é representada por um vetor com 3 números.
# Portanto: 6 palavras × 3 números = matriz 6 × 3
# [0.43, 0.15, 0.89] → Your
# [0.55, 0.87, 0.66] → journey
# [0.57, 0.85, 0.64] → starts
# [0.22, 0.58, 0.33] → with
# [0.77, 0.25, 0.10] → one
# [0.05, 0.80, 0.55] → step

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



# inputs[1] significa: pegue a segunda palavra.
query = inputs[1] # journey



# ATTENTION SCORES

# Queremos comparar "journey" com todas as palavras.
# torch.empty(6) cria espaço para guardar esses 6 números.
attn_scores_2 = torch.empty(inputs.shape[0])


# Passamos por Ttodas as palavras. x_i será, a cada volta, o vetor de uma palavra.
# 1 volta → Your
# 2 volta → journey
# 3 volta → starts
# 6 volta → step
for i, x_i in enumerate(inputs):

    # torch.dot() faz o produto escalar entre dois vetores. Multiplica posição por posição
    # Depois SOMA os 3 resultados. Isso gera um único número.
    # Esse número é o attention score.
    attn_scores_2[i] = torch.dot(x_i, query)


# Agora temos 6 attention scores, 1 score para cada palavra comparada com "journey".
print(attn_scores_2)



# NORMALIZAR OS SCORES

# Poderíamos simplesmente dividir cada score pela soma de todos os scores:score / soma dos scores
# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())


# Esta é uma implementação simples do softmax.
# Ela transforma os scores em pesos.
# Os pesos ficam positivos
# Porém, na prática, usamos torch.softmax(),
# que é mais segura numericamente
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())


# Aqui usamos a implementação do próprio PyTorch
# Ela transforma os 6 attention scores em 6 attention weights
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)

# Os 6 pesos devem somar aproximadamente 1
print("Sum:", attn_weights_2.sum())



# VETOR DE CONTEXTO
# Agora vamos juntar as informações das 6 palavras
# Temos 6 palavras, mas cada palavra possui 3 números. O vetor de contexto também terá 3 números

# vetor de zeros
context_vec_2 = torch.zeros(query.shape)

# Passamos novamente pelas 6 palavras
for i, x_i in enumerate(inputs):
    # peso da palavra × vetor da palavra. Depois adicionamos esse resultado ao vetor de contexto.
    # Fazemos isso para TODAS as 6 palavras.
    context_vec_2 += attn_weights_2[i] * x_i


# Depois de combinar as 6 palavras,
# temos UM vetor de contexto para "journey".
# Ele possui 3 números porque cada embeddingpossui 3 números.
print(context_vec_2)