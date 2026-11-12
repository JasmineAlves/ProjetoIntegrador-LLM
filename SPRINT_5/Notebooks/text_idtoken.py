import tiktoken  # Transforma texto em numeros e vice-versa
import torch
from SPRINT_4.Notebooks.GPT_model import generate_text_simple
from SPRINT_4.Notebooks.GPT_model import model
from SPRINT_5.Notebooks.inicializacao_modelo import GPT_CONFIG_124M  # Configuraçoes dos parametros

# CODIFICADOR
# Recebe texto comum e um tokenizador
def text_to_token_ids(text, tokenizer):
    # Converte texto em lista de numeros (tokens), allowed_special é para processar marcas como marcador de fim de texto
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # Transforma a lista em um tensor e coloca uma dimensão extra no inicio (unsqueeze) para simular um batch de uma única frase
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # Retorna tensor pronto
    return encoded_tensor

# DECODIFICADOR
# Pega esse tensor e transforma de volta em texto legível
def token_ids_to_text(token_ids, tokenizer):
    # Remove a dimensão extra e transforma o tensor de volta em uma lista simples de números
    flat = token_ids.squeeze(0)
    # Converte a lista do Pytprch em uma lista padrão e usa tokenizador para traduzir esse numeros em texto
    return tokenizer.decode(flat.tolist())

# Frase de exemplo que será utilizada como base para o modelo continuar escrevendo
start_context = "Every effort moves you"
# Tokenizador padrão usado pelo GPT-2
tokenizer = tiktoken.get_encoding("gpt2")

# Chama a função de geração de texto do modelo (treinada para ler sequencia de números e prever qual é o proximo mais provavel)
token_ids = generate_text_simple(
    # Passa pelo modelo treinado
    model = model,
    # Texto inicial convertido em numeros
    idx=text_to_token_ids(start_context, tokenizer),
    # Limite de 10 tokens novos para serem gerados
    max_new_tokens=10,
    # Tamanho maximo de contexto (256)
    context_size=GPT_CONFIG_124M["context_length"]
)

# Pega os numeros gerados pelo modelo e converte em texto usando a função de decodificar
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# OBS: Adiciona dimensão extra para batch, o Pytorch foi feito para processar muito dados ao mesmo tempo (batches) para ganhar
# velocidade na GPU/CPU e por isso o modelo espra recever os dados em 2 dimensões [tamanho_lote, tamanho_texto], se mandarmos apenas
# o texto cru o tokenizador devolve uma lista simples [tamanho_texto] e ai o modelo da erro porque não ve isso como um batch, então
# coloca-se uma dimensão na frente e fica [1, tamanho_texto] e depois tira o 1 quando termina de gerar os numeros porque a função
# decodificação espera uma lista simples (1 dimensão)