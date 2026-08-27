# Experimento 1 - Tokenização
# Utilizando o the-verdict.txt
# Observar como diferentes textos são transformados em tokens utilizando o tokenizer BPE que já foi implementado/utilizado
import sys
import os
import tiktoken
# Importar arquivos que estão na pasta Notebooks
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Notebooks")
    )
)
# Carrega o tokenizer BPE 
import BPE



# Textos utilizados no experimento
textos = [
    "Hello, how are you?",
    "The model learns from data.",
    "Machine learning uses numerical representations.",
    "This is a longer sentence used to observe how the number of tokens changes."
]



# Resultads de cada texto
print("EXPERIMENTO 1 - TOKENIZAÇÃO")

for texto in textos:

    # Tokeniza o texto usando BPE
    tokens = BPE.tokenizer.encode(texto)

    # Quantidade de caracteres
    quantidade_caracteres = len(texto)

    # Quantidade de tokens
    quantidade_tokens = len(tokens)

    # Média aproximada de caracteres por token
    media = quantidade_caracteres / quantidade_tokens

    print("\nTexto:")
    print(texto)

    print("Quantidade de caracteres:", quantidade_caracteres)
    print("Quantidade de tokens:", quantidade_tokens)
    print("Média de caracteres por token:", round(media, 2))

    # Mostra os Token IDs produzidos
    print("Token IDs:")
    print(tokens)

    # Mostra os tokens reconstruídos como texto
    print("Texto:")
    print(BPE.tokenizer.decode(tokens))


# Texto para experimento
print("\n")
print("THE-VERDICT.TXT")

with open(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "Notebooks",
        "the-verdict.txt"
    ),
    "r",
    encoding="utf-8"
) as f:

    raw_text = f.read()


tokens_livro = BPE.tokenizer.encode(raw_text)


print("Quantidade de caracteres:", len(raw_text))
print("Quantidade de tokens:", len(tokens_livro))

print(
    "Média de caracteres por token:",
    round(len(raw_text) / len(tokens_livro), 2)
)

# Textos diferentes produzem quantidades diferentes de tokens. A quantidade de tokens não é necessariamente igual à
# quantidade de palavras ou caracteres. Isso acontece porque o BPE pode dividir palavras em unidades menores.
