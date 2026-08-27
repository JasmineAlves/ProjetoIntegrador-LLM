# Resultados dos Experimentos - Sprint 2
# Resultados obtidos nos experimentos da Sprint 2. Serão organizados em tabelas para facilitar
# a análise técnica.
# Experimentos analisados:
# 1. Tokenização
# 2. Tamanho do contexto
# 3. Tamanho do batch
# 4. Dimensão dos embeddings
# 5. Positional embeddings

import sys
import os
import torch
import tiktoken

# Importar códigos da pasta Notebooks
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
NOTEBOOKS_DIR = os.path.join(
    BASE_DIR,
    "Notebooks"
)
sys.path.append(NOTEBOOKS_DIR)
# Importa o código de preparação das sequências
import Entradas_Alvos



# Carrega o texto, ainda é utilizado o the-verdict
texto_path = os.path.join(
    NOTEBOOKS_DIR,
    "the-verdict.txt"
)
with open(
    texto_path,
    "r",
    encoding="utf-8"
) as f:

    raw_text = f.read()



# TOKENIZER --------------------------------------------------------------------------------
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(raw_text)
total_tokens = len(token_ids)



# EXPERIMENTO 1 - TOKENIZAÇÃO
print("\n")
print("EXPERIMENTO 1 - TOKENIZAÇÃO")

textos = [
    "Hello, how are you?",
    "The model learns from data.",
    "Machine learning uses numerical representations.",
    "This is a longer sentence used to observe how the number of tokens changes."
]

print(
    f"{'Texto':<70} {'Caracteres':>12} {'Tokens':>10}"
)

print("-" * 60)

for texto in textos:

    tokens = tokenizer.encode(texto)
    caracteres = len(texto)
    quantidade_tokens = len(tokens)
    print(
        f"{texto:<70} "
        f"{caracteres:>12} "
        f"{quantidade_tokens:>10}"
    )

print("\nThe Verdict:")
print("Caracteres:", len(raw_text))
print("Tokens:", total_tokens)



# EXPERIMENTO 2 - TAMANHO DO CONTEXTO
print("\n")
print("EXPERIMENTO 2 - TAMANHO DO CONTEXTO")

tamanhos_contexto = [
    4,
    8,
    16,
    32,
    64
]

print(
    f"{'Contexto':>12} "
    f"{'Tokens por sequência':>22} "
    f"{'Amostras':>12}"
)

print("-" * 60)

for contexto in tamanhos_contexto:

    dataset = Entradas_Alvos.GPTDatasetV1(
        raw_text,
        tokenizer,
        max_length=contexto,
        stride=contexto
    )

    quantidade_amostras = len(dataset)

    print(
        f"{contexto:>12} "
        f"{contexto:>22} "
        f"{quantidade_amostras:>12}"
    )



# EXPERIMENTO 3 - TAMANHO DO BATCH
print("\n")
print("EXPERIMENTO 3 - TAMANHO DO BATCH")

contexto = 8
tamanhos_batch = [
    1,
    2,
    4,
    8,
    16
]

print(
    f"{'Batch size':>12} "
    f"{'Shape inputs':>20} "
    f"{'Shape targets':>20} "
    f"{'Batches':>12}"
)

print("-" * 60)

for batch_size in tamanhos_batch:

    dataloader = Entradas_Alvos.create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=contexto,
        stride=contexto,
        shuffle=False,
        drop_last=True
    )

    inputs, targets = next(iter(dataloader))

    print(
        f"{batch_size:>12} "
        f"{str(tuple(inputs.shape)):>20} "
        f"{str(tuple(targets.shape)):>20} "
        f"{len(dataloader):>12}"
    )


# EXPERIMENTO 4 - DIMENSÃO DOS EMBEDDINGS
print("\n")
print("EXPERIMENTO 4 - DIMENSÃO DOS EMBEDDINGS")

vocab_size = 50257
dimensoes_embedding = [
    64,
    128,
    256,
    512
]



# Criamos um pequeno batch de Token IDs [8, 4]
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

print(
    f"{'Dimensão':>12} "
    f"{'Shape':>25} "
    f"{'Parâmetros':>18}"
)

print("-" * 60)


for output_dim in dimensoes_embedding:

    embedding_layer = torch.nn.Embedding(
        vocab_size,
        output_dim
    )

    embeddings = embedding_layer(input_ids)

    quantidade_parametros = (
        vocab_size * output_dim
    )

    print(
        f"{output_dim:>12} "
        f"{str(tuple(embeddings.shape)):>25} "
        f"{quantidade_parametros:>18}"
    )



# EXPERIMENTO 5 - POSITIONAL EMBEDDINGS
print("\n")
print("EXPERIMENTO 5 - POSITIONAL EMBEDDINGS")

torch.manual_seed(123)

vocab_size = 50257
embedding_dim = 8
context_length = 4



# Camada responsável pelos embeddings dos tokens
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)



# Camada responsável pelos embeddings das posições
position_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)

# Mesmo Token ID em duas posições diferentes
token_id = torch.tensor([100])
token_embedding = token_embedding_layer(token_id)

# Posição 0
position_0 = torch.tensor([0])
position_embedding_0 = position_embedding_layer(position_0)

# Posição 3
position_3 = torch.tensor([3])
position_embedding_3 = position_embedding_layer(position_3)



# Reprsentação final
final_position_0 = (
    token_embedding +
    position_embedding_0
)

final_position_3 = (
    token_embedding +
    position_embedding_3
)



# Verificações
embeddings_posicoes_diferentes = not torch.equal(
    position_embedding_0,
    position_embedding_3
)

representacoes_diferentes = not torch.equal(
    final_position_0,
    final_position_3
)

print(
    "Mesmo Token ID utilizado:",
    token_id.item()
)
print(
    "Posições comparadas:",
    "0 e 3"
)
print(
    "Embeddings das posições são diferentes:",
    embeddings_posicoes_diferentes
)
print(
    "Representações finais são diferentes:",
    representacoes_diferentes
)


