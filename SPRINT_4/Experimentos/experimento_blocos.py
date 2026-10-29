import time
import torch
import sys
from pathlib import Path
# Localiza a pasta raiz do projeto para conseguir importar os módulos de outras pastas
raiz = Path(__file__).resolve().parents[2]
# Adiciona o caminho onde estão os arquivos da SPRINT_4 para o Python conseguir achá-los
sys.path.insert(
    0,
    str(raiz / "SPRINT_4" / "Notebooks")
)
import cfg
import GPT_model
from experimentos_reg import salvar_resultado

torch.manual_seed(123)

# Testando diferentes quantidades de Blocos Transformer --------------------------------------------------------------------------------------------
# Testar o modelo alterando o número de blocos para 2, 4 e 6
for quantidade_blocos in [2, 4, 6]:
    # Cria uma cópia das configurações padrão 
    config = cfg.GPT_CONFIG_124M.copy()
    
    # Altera o número de camadas (blocos transformers) para o tste atual
    config["n_layers"] = quantidade_blocos

    # Inicializa o modelo GPT com a nova configuração de blocos
    model = GPT_model.GPTModel(config)
    
    # Coloca o modelo em modo de avaliação
    model.eval()

    # Cria uma entrada com 1 frase contendo 4 IDs de tokens gerados aleatoriamente
    # Números inteiros entre 0 e o tamanho total do vocabulário
    entrada = torch.randint(
        0, config["vocab_size"], (1, 4)
    )

    # Inicia a contagem do tempo de execução para medir o custo computacional
    inicio = time.perf_counter()

    # Desativa o cálculo de gradientes (economiza memória)
    with torch.no_grad():
        # Passa a entrada pelo modelo e obtém as previsões (logits)
        saida = model(entrada)

    # Calcula quanto tempo o processo levou (em segundos)
    tempo = time.perf_counter() - inicio
    
    # Conta o número total de parâmetros treináveis 
    parametros = sum(
        p.numel() for p in model.parameters()
    )

    # Exibe os resultados detalhados do teste atual
    print(f"\nBlocos: {quantidade_blocos}")
    print(f"Parâmetros: {parametros:,}")
    print(f"Formato da saída: {tuple(saida.shape)}")
    print(f"Tempo de execução: {tempo:.4f} segundos")

    # Salva os dados do experimento
    salvar_resultado(
        "Quantidade de Transformer Blocks",
        {
            "blocos": quantidade_blocos,
            "parametros": parametros,
            "tempo_segundos": round(tempo, 4),
            "formato_saida": str(tuple(saida.shape))
        }
    )

    # Limpa a memória RAM apagando as variáveis do modelo atual antes do próximo loop
    del model, entrada, saida