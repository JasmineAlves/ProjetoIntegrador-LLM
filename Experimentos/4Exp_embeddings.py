# Experimento 4 - Tamanho dos Embeddings
# Como a dimensão do embedding altera a representação dos tokens e o tamanho das estruturas utilizadas pelo modelo
# O tamanho do vocabulário será mantido em 50257, que é o vocabulário utilizado pelo tokenizer GPT-2
import torch


# Configurações
vocab_size = 50257

# Tamanhos de embedding que vão ser comparados
dimensoes_embedding = [64, 128, 256, 512]

# Exemplo de batch 8 sequências cada uma contendo 4 tokens [8, 4]
input_ids = torch.tensor([
    [100, 200, 300, 400],
    [101, 201, 301, 401],
    [102, 202, 302, 402],
    [103, 203, 303, 403],
    [104, 204, 304, 404],
    [105, 205, 305, 405],
    [106, 206, 306, 406],
    [107, 207, 307, 407]
])

print("EXPERIMENTO 4 - DIMENSÃO DOS EMBEDDINGS")
print("Shape dos Tokens IDs:", input_ids.shape)



# Testa cada dimensão
for output_dim in dimensoes_embedding:

    # Cria uma camada de embedding em que cada um dos 50257 tokens terá um vetor com x (output_dim) números.
    embedding_layer = torch.nn.Embedding(
        vocab_size,
        output_dim
    )

    
    # Converte os Tokens IDs em vetores
    embeddings = embedding_layer(input_ids)

    
    # Quantidade de parâmetros = vocab_size * dimensão
    quantidade_parametros = (vocab_size * output_dim)

    print("\nDimensão do embedding:", output_dim)
    print("Shape dos embeddings:", embeddings.shape)
    print("Quantidade de parâmetros:", quantidade_parametros)


# Os Token IDs possuem apenas duas dimensões [batch, contexto], depois do embedding temos [batch, contexto, embedding_dim]
# Aumentar a dimensão do embedding aumenta diretamente o tamanho da representação vetorial.
# Aumenta a quantidade de parâmetros que a camada de embedding precisa aprender.
