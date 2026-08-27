# Experimento 3 - Tamanho do Batch
# Observar como diferentes valores de batch_size alteram a organzação dos dados produzidoos pelo DataLoader.
# Vamos manter o tamanho do contexto fixo e modificar apenas o tamanho do lotee
import sys
import os
# Importar arquivos da pasta Notebooks
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Notebooks")
    )
)
import Entradas_Alvos



# Carrega o texto
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



# Configuração do experimento
contexto = 8
tamanhos_batch = [1, 2, 4, 8, 16]

print("EXPERIMENTO 3 - TAMANHO DO BATCH")
print("Tamanho do contexto:", contexto)



# Testa diferentes batch_size
for batch_size in tamanhos_batch:

    dataloader = Entradas_Alvos.create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=contexto,
        stride=contexto,
        shuffle=False,
        drop_last=True
    )

    
    # Pegamos o primeiro batch para observar suas dimensões
    inputs, targets = next(iter(dataloader))

    print("\nBatch size:", batch_size)
    print("Shape dos inputs:", inputs.shape)
    print("Shape dos targets:", targets.shape)

    print(
        "Quantidade de batches:",
        len(dataloader)
    )


# O batch_size determina quantas sequências são processadas  juntas.
# Por exemplo, batch_size = 1 -> uma sequência por batch
# batch_size = 8 -> oito sequências por batch
# batch 1 -> [1, 8]   batch 4 -> [4, 8]   batch 8 -> [8, 8]. O conteúdo é oss Token IDs.