import re
# Mostra o total de caracteres existente no arquivo
# Objetivo: Tokenizar 20479 em palavras individuais para transformar em embeddings para a LLM
with open("Experimentos/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

# Lista do texto separado assim como espaços em branco
text = "Hello, world. This, is a test."
result = re.sp