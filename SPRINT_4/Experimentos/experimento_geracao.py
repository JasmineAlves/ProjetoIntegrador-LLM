import time
import torch
import tiktoken
import sys
from pathlib import Path
# Localiza a pasta raiz do projeto para permitir importações corretas
raiz = Path(__file__).resolve().parents[2]
# Adiciona o diretório da SPRINT_4 ao caminho de busca do Python
sys.path.insert(
    0,
    str(raiz / "SPRINT_4" / "Notebooks")
)
from cfg import GPT_CONFIG_124M
from GPT_model import GPTModel
from experimentos_reg import salvar_resultado


def generate_text_simple(
    model, idx, max_new_tokens, context_size
):
    # Coloca o modelo em modo de avaliação 
    model.eval()

    # Repete o processo para gerar a quantidade exata de novos tokens 
    for _ in range(max_new_tokens):

        # Seleciona apenas os últimos tokens que cabem no limite do contexto do modelo
        idx_cond = idx[:, -context_size:]

        # Desativa o cálculo de gradientes 
        with torch.no_grad():
            # obter as pontuações (logits)
            logits = model(idx_cond)

        # Filtra apenas os logits correspondentes à última posição da sequência (o próximo token)
        logits = logits[:, -1, :]
        # Converte as pontuações em probabilidades (valores entre 0 e 1)
        probas = torch.softmax(logits, dim=-1)
        # Escolhe de forma gulosa (argmax) o token com a maior probabilidade
        idx_next = torch.argmax(
            probas, dim=-1, keepdim=True
        )
        # Cola o novo token gerado no final da sequência original de entrada
        idx = torch.cat((idx, idx_next), dim=1)

    # Retorna a sequência completa de tokens
    return idx

torch.manual_seed(123)

# Configura um modelo
config = GPT_CONFIG_124M.copy()
config["n_layers"] = 2

# Inicializa o modelo GPT e o coloa em modo de avaliação
model = GPTModel(config)
model.eval()

# Inicializa o tokenizador do gpt para converter texto em números e ao contrario
tokenizador = tiktoken.get_encoding("gpt2")

# Define a frase inicial para ponto de partida para a geração
texto_inicial = "Muitos divos e divas aprendem IA"
ids = tokenizador.encode(texto_inicial)

# Transforma a lista de IDs em um tensor do PyTorch no formato correto (lote de 1 linha)
entrada = torch.tensor([ids], dtype=torch.long)

# Testar diferentes quantidades de tokens gerados -----------------------------------------------------------------------------------
# Testar o tempo que o modelo leva para gerar 5, 10 e 20 tokens
for quantidade_tokens in [5, 10, 20]:
    # Marca o tempo inicial da execução
    inicio = time.perf_counter()
    # Chama a função de geração passando uma cópia do tensor de entrada
    resultado = generate_text_simple(
        model=model,
        idx=entrada.clone(),
        max_new_tokens=quantidade_tokens,
        context_size=config["context_length"]
    )
    # Calcula quanto tempo o processo demorou
    tempo = time.perf_counter() - inicio

    # Convertee de volta a lista de IDs resultante em um texto legivel
    texto_gerado = tokenizador.decode(
        resultado[0].tolist()
    )

    # Exibe resultados
    print(f"\nNovos tokens: {quantidade_tokens}")
    print(f"Tempo: {tempo:.4f} segundos")
    print(f"Texto gerado: {texto_gerado}")

    # Salva os resultados 
    salvar_resultado(
        "Quantidade de tokens gerados",
        {
            "novos_tokens": quantidade_tokens,
            "tempo_segundos": round(tempo, 4),
            "total_tokens": resultado.shape[1],
            "texto_gerado": texto_gerado
        }
    )
    # Libera a memória RAM
    del resultado