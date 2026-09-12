# Experimento 2 - Tamanho do contexto
# Como o tamanho do contexto influencia a quantidade de sequências de treinamento produzidas.
# Quanto maior o contexto, maior é a quantidade de tokens utilizada em cada sequência.
import sys
import os
import tiktoken
# Permite importar arquivos da pasta Notebooks
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Notebooks")
    )
)
# Importa o algoritmo das entradas alvos
import Entradas_Alvos



# Carrega o texto utilizado
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



# Utilizamos o tokenizer BPE 
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(raw_text)
total_tokens = len(token_ids)

print("EXPERIMENTO 2 - TAMANHO DO CONTEXTO")
print("Total de tokens no texto:", total_tokens)



# Diferentes tamanhos de contexto que vão ser testados
tamanhos_contexto = [4, 8, 16, 32, 64]



# Testa cada tamanho de contexto
for contexto in tamanhos_contexto:

    # Fórmula aproximada para saber quantas janelas podem ser criadas quando stride = contexto (não existe sobreposição).
    # range(0, len(token_ids) - max_length, strides
    quantidade_amostras = (
        (total_tokens - contexto) // contexto
    )
    print("\nContexto:", contexto)
    print("Tokens por sequência:", contexto)
    print("Quantidade de amostras:", quantidade_amostras)



# Agora verifica-se utilizando o Dataset
print("\n")
print("VERIFICAÇÃO COM GPTDatasetV1")

for contexto in tamanhos_contexto:

    dataset = Entradas_Alvos.GPTDatasetV1(
        raw_text,
        tokenizer,
        max_length=contexto,
        stride=contexto
    )

    print(
        "Contexto:",
        contexto,
        "| Amostras:",
        len(dataset)
    )



# Com o mesmo texto: contexto pequeno -> mais sequências, contexto grande  -> menos sequências
# Isso acontece porque cada amostra passa a consumir uma quantidade maior de tokens.
