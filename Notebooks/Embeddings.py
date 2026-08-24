import torch

# Transformação de token IDs em vetores embedding

# Tokens IDs de exemplo
input_ids = torch.tensor([2, 3, 5, 1])
# Tamanho do vocabulário (6 palavras)
vocab_size = 6
# Criar embeddings de tamanho 3
output_dim = 3

# Inicializar os pesos do embedding com valores aleatórios para ser reproduzivel
torch.manual_seed(123)
# Instanciar uma camada embedding
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# Exibe a matriz de pesos subjacente da camada embedding contendo pequenos e aleatórios valores
# que vão ser otimizado durante o treinamento da LLM
# Observa-se que existe uma linha para 6 tokens possiveis no vocabulário e tem 3 colunas para cada dimensão do embedding
print(embedding_layer.weight) 

# Aplicar a um ID(3) do token para obter o vetor de embedding
# Se compararmos com a matriz anterior de embeddings nós vemos que é identico a 4° linha
print(embedding_layer(torch.tensor([3])))

# Aplicar para as 4 entradas tokens IDs
# Criamos os vetores embeddingss a partir dos tokens IDs
print(embedding_layer(input_ids)) # Matriz 4x3