import time
import torch
import sys
from pathlib import Path
# Localiza a pasta raiz do projeto para permitir importações entre pastas diferentes
raiz = Path(__file__).resolve().parents[2]
# Adiciona o diretório da SPRINT_4 ao caminho de busca do Python
sys.path.insert(
    0,
    str(raiz / "SPRINT_4" / "Notebooks")
)
from cfg import GPT_CONFIG_124M
from GPT_model import GPTModel
from experimentos_reg import salvar_resultado

# Define uma semente fixa para garantir que os números aleatórios sejam reprodutíveis
torch.manual_seed(123)


# Testando diferentes Dimensões de Embedding --------------------------------------------------------------
# Testar o modelo variando o tamanho do vetor que representa cada token 192 384 e 768
for dimensao in [192, 384, 768]:

    # Faz uma cópia das configurações originais
    config = GPT_CONFIG_124M.copy()
    # Altera a dimensão do embedding para o valor da rodada atual
    config["emb_dim"] = dimensao
    # Fixa o número de camadas em 2 para isolar o impacto apenas da dimensão do embedding
    config["n_layers"] = 2

    # A dimensão do embedding precisa ser obrigatoriamente divisível pelo número de cabeças de atenção (n_heads), o vetor é repartido entre elas
    # O comando abaixo trava o código casoessa regra seja violada
    assert dimensao % config["n_heads"] == 0

    # Inicializa o modelo GPT com essa nova configuração
    model = GPTModel(config)
    
    # Coloca o modelo em modo de avaliação
    model.eval()

    # Cria uma frase de entrada 1 única sequência contendo 4 tokens gerados aleatoriamente
    entrada = torch.randint(
        0, config["vocab_size"], (1, 4)
    )

    # Marca o tempo inicial para medir o custo de processamento
    inicio = time.perf_counter()

    with torch.no_grad():
        # Passa a entrada pelo modelo e calcula as pontuações (logits) de saída
        saida = model(entrada)

    # Calcula quanto tempo o processo demoro
    tempo = time.perf_counter() - inicio
    
    # Soma o número total de parâmetros treinaveis
    parametros = sum(
        p.numel() for p in model.parameters()
    )

    # Exibe os resultados obtidoos no teste atual
    print(f"\nDimensão do embedding: {dimensao}")
    print(f"Parâmetros: {parametros:,}")
    print(f"Formato da saída: {tuple(saida.shape)}")
    print(f"Tempo de execução: {tempo:.4f} segundos")

    # Salva os dados
    salvar_resultado(
        "Dimensão dos embeddings",
        {
            "emb_dim": dimensao,
            "blocos": config["n_layers"],
            "parametros": parametros,
            "tempo_segundos": round(tempo, 4),
            "formato_saida": str(tuple(saida.shape))
        }
    )

    del model, entrada, saida