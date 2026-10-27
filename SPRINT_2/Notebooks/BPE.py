from importlib.metadata import version
# Biblioteca para byte pair encoding
import tiktoken
# Como esse algoritmo é complexo de montar, usamos uma biblioteca que implementa ele

tokenizer = tiktoken.get_encoding("gpt2")

# Similar as classes implementadas em Tokens_PyTorch.py
# Texto de entrada
text = (
 "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
 "of someunknownPlace."
)

# Codificador
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers) # Tokens IDs
# O token de endoftext é relativamente largo

# Decodificador
strings = tokenizer.decode(integers)
print(strings) # Texto com endoftext