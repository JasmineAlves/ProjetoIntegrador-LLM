import torch
import sys
from pathlib import Path
# Localiza a pasta raiz do projeto para permitir importações corretas entre as pastas
raiz = Path(__file__).resolve().parents[2]
# Adiciona o diretório da SPRINT_4 ao caminho de busca do Python
sys.path.insert(
    0,
    str(raiz / "SPRINT_4" / "Notebooks")
)
from cfg import GPT_CONFIG_124M
from GPT_model import GPTModel
from experimentos_reg import salvar_resultado


torch.manual_seed(123)
# Configura a base do modelo com apenas 2 camadas para deixar o teste mais rapdio 
config_base = GPT_CONFIG_124M.copy()
config_base["n_layers"] = 2

# Cria uma entrada de 1 frase contendo 4 IDs de tokens gerados aleatoriamente
entrada = torch.randint(
    0, config_base["vocab_size"], (1, 4)
)

# Testando diferentes taxas de dropout ------------------------------------------------------------------------------
# Vamos testar o impacto do dropout variando a taxa 0.0 (desligado) 0.1 (10%) e 0.3 (30%)
for taxa_dropout in [0.0, 0.1, 0.3]:
    # Copia a configuração base 
    config = config_base.copy()
    config["drop_rate"] = taxa_dropout

    # Inicializa o modelo 
    model = GPTModel(config)

    # Modo de treinamento, o Dropout fica ATIVO 
    model.train()
    with torch.no_grad():
        saida_treino = model(entrada)

    #Modo de avaliação, o Dropout é DESATIVADO 
    model.eval()
    with torch.no_grad():
        saida_avaliacao = model(entrada)

    # Calcula a diferença média absoluta entre a saída gerada no modo treino e no modo avaliação
    # Se o dropout for 0.0, as saídas serão idêntica
    # Conforme a taxa aumenta, maior tende a ser a diferença entre os dois modos
    diferenca = torch.mean(
        torch.abs(saida_treino - saida_avaliacao)
    ).item()

    # Exibe resultaddo
    print(f"\nTaxa de Dropout: {taxa_dropout}")
    print(f"Diferença média entre saídas: {diferenca:.6f}")
    print(f"Formato da saída: {tuple(saida_avaliacao.shape)}")

    # Salva os dados
    salvar_resultado(
        "Taxa de Dropout",
        {
            "dropout": taxa_dropout,
            "diferenca_media": round(diferenca, 6),
            "formato_saida": str(tuple(saida_avaliacao.shape))
        }
    )

    # Libera a memória RAM
    del model, saida_treino, saida_avaliacao