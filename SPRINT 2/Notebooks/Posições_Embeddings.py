import torch
import Entradas_Alvos


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Cada entrada/frase terá 4 tokens
max_length = 4

# Função que cria um DataLoader para organizar os dados em batches para treinar o modelo
dataloader = Entradas_Alvos.create_dataloader_v1(
# raw_text é o texto original que queremos transformar em dados, queremos 8 sequências de uma vez
# sendo que cada sequência possui 4 tokens 8x4=32
# Stride é de quanto em quanto valor caminhar, nesse caso de 4 em 4 ou seja, não temos sobreposição, se fosse =2 
# os ultimos 2 tokens iriam repetir na próxima sequência. 
# O shuffle é false para não embaralhar os dados
 Entradas_Alvos.raw_text, batch_size=8, max_length=max_length,
 stride=max_length, shuffle=False
)
# Cria iterador para podermos pegar os batches um por um
data_iter = iter(dataloader)
# Pega o próximo batch
# Por exemplo INPUT = Eu gosto de    e  TARGET = gosto de Python
inputs, targets = next(data_iter)

# Mostra os IDs dos tokens
print("Token IDs:\n", inputs)

# Formato do tensor = batch_size 8 e max_length 4
print("\nInputs shape:\n", inputs.shape)

# Antes cada token era apenas um número ID, o embedding pega cada ID e transforma e um vetor de 256 números
# ou seja, cada token tem um vetor de tamanho 256
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# ABSOLUTE EMBEDDING POSITION --------------------------------------------------------------------------------------
# Pega a max_length 4
context_length = max_length
# Cria uma tabela de embeddings paa as posições [0 a 3] e cada posição tem um vetor de 256 números
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# Print retorna 4, 256
print(pos_embeddings.shape)

# Lembrando temos os embeddings dos tokens [8, 4, 256] e os da posições [4, 256] o Pytorch entende
# que cada um dos 8 textos deve ser aplicado o [4, 256]

# Soma embedding de "palavra" + embedding da posição 0 = embedding final
# Assim o modelo recebe o significado do token e a informação de onde ele está
input_embeddings = token_embeddings + pos_embeddings
# Resultado [8, 4, 256]
print(input_embeddings.shape)