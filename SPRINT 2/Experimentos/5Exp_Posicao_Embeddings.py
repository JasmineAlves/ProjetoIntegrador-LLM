# Experimento 5 - Positional Embeddings
# A posição de um tokeninfluencia sua representação final
# O embedding do token permanece o mesmo, mas o embedding da posição muda
# Consequentemente reprsentação final = token embedding + positional embeedding também vai ser diferente
import torch

# Configurações
torch.manual_seed(123)

vocab_size = 50257
embedding_dim = 8
context_length = 4



# Cria a camada de embedding dos tokens
token_embedding_layer = torch.nn.Embedding(
    vocab_size,
    embedding_dim
)



# Cria a camada de embedding das posições sendo 4 posições [0, 3]. Cada posição possui um vetor de 8 números.
position_embedding_layer = torch.nn.Embedding(
    context_length,
    embedding_dim
)

# Vamos utilizar o mesmo token em duas posições diferentes
# Token ID 100 aparece posição 0 e 3
token_id = torch.tensor([100])



# Embedding do token
token_embedding = token_embedding_layer(token_id)



# Embeedding da posição 0
position_0 = torch.tensor([0])
position_embedding_0 = position_embedding_layer(position_0)

# Embedding da posição 3
position_3 = torch.tensor([3])
position_embedding_3 = position_embedding_layer(position_3)



# Representação final do token na posição 0
final_position_0 = (token_embedding + position_embedding_0)

# Representação final do token na posição 3
final_position_3 = (token_embedding + position_embedding_3)



# Exibição dos resultados
print("EXPERIMENTO 5 - POSITIONAL EMBEDDINGS")


print("\nEmbedding do token 100:")
print(token_embedding)
print("\nEmbedding da posição 0:")
print(position_embedding_0)
print("\nEmbedding da posição 3:")
print(position_embedding_3)
print("\nRepresentação final na posição 0:")
print(final_position_0)
print("\nRepresentação final na posição 3:")
print(final_position_3)



# Verifica se os embeddings das posições são diferentes
print(
    "\nEmbedding da posição 0 é igual ao da posição 3?",
    torch.equal(
        position_embedding_0,
        position_embedding_3
    )
)

# Verifica see as representações finais são diferentes
print(
    "Representação final é igual?",
    torch.equal(
        final_position_0,
        final_position_3
    )
)

# Calcula a diferença entre as repesentações
diferenca = torch.abs(final_position_0 - final_position_3)

print("\nDiferença absoluta entre as representações:")
print(diferenca)


# O Token ID 100 é exatamente o mesmo nos dois casos. O token embedding é igual. Mas a posição 0 
# é diferente da posição 3, então positional embedding deve ser diferente
# e por isso a representação final das duas posições deve ser diferente
# Isso demonstra por que o modelo precisa incorporarinformação de posição.
