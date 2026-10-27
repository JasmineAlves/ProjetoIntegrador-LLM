import torch

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

# Computar todos os vetores de contexto de todas as palavras e não só journey
attn_scores = torch.empty(6, 6)

# for i, x_i in enumerate(inputs):
    # Adiciona um segundo FOR  para computar o dot para todos os pares de entradas
    # for j, x_j in enumerate(inputs):
      # attn_scores[i, j] = torch.dot(x_i, x_j)
# Cada elemento no tensor representa um attention score entre par de entradas
# print(attn_scores)


# Pode-se utilizar multiplicação de matrizes ao invés de loops FOR porque é mais rápido
attn_scores = inputs @ inputs.T
print(attn_scores)

# Normalização de cada linha para os valores em cada uma somarem 1
# dim=-1 (softmax aplicado em cada linha) é para pegar a última dimensão (colunas), e para cada palavra transforma os 6 scores
# daquela linha em 6 pesos que somam 1
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# verificando a soma das linhas
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# usa os pesos de atenção para computar todos os vetores de contexto atarvés de multiplicação de matriz
all_context_vecs = attn_weights @ inputs # (6 × 6) @ (6 × 3) = (6 x 3) em multiplicação de matrizes
# Cada linha é um vetor de contexto de 3 dimensões (embedding de cada palavra)
print(all_context_vecs)