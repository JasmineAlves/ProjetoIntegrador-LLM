import csv
import os
# Este arquivo será usado pelos quatro experimentos para registrar os resultados
ARQUIVO_RESULTADOS = "resultados_experimentos.csv"


def salvar_resultado(experimento, resultados):
    arquivo_existe = os.path.exists(ARQUIVO_RESULTADOS)

    campos = ["experimento"] + list(resultados.keys())
    linha = {"experimento": experimento, **resultados}

    # Se o CSV já existe, mantém as colunas existentes
    if arquivo_existe:
        with open(ARQUIVO_RESULTADOS, "r", newline="", encoding="utf-8") as f:
            leitor = csv.DictReader(f)
            campos_existentes = leitor.fieldnames or []
    else:
        campos_existentes = []

    campos_finais = list(dict.fromkeys(campos_existentes + campos))

    # Reescreve o CSV para acomodar resultados com colunas diferentes
    linhas_antigas = []
    if arquivo_existe:
        with open(ARQUIVO_RESULTADOS, "r", newline="", encoding="utf-8") as f:
            linhas_antigas = list(csv.DictReader(f))

    linhas_antigas.append(linha)

    with open(ARQUIVO_RESULTADOS, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos_finais)
        escritor.writeheader()
        escritor.writerows(linhas_antigas)

    print(f"Resultado salvo em {ARQUIVO_RESULTADOS}")