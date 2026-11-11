import torch
import tiktoken
from SPRINT_4.Notebooks.GPT_model import model
from SPRINT_5.Notebooks.text_idtoken import token_ids_to_text
from SPRINT_5.Notebooks.text_idtoken import text_to_token_ids
from SPRINT_5.Notebooks.inicializacao_modelo import GPT_CONFIG_124M
from SPRINT_4.Notebooks.GPT_model import generate_text_simple

# Estamos trabalhando com um modelo pequeno então não precisa da GPU
model.to("cpu")
# Modelo em modo avaliação
model.eval()

# Transforma textos em tokens e vice-versa
tokenizer = tiktoken.get_encoding("gpt2")
# Recebe o modelo e a frase inicial
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
# Transforma os numeros gerados em palavras novamente
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# OBS: Como o código atual obriga o modelo a sempre escolher a palavra com a maior pontuação matemática ela se torna totalmente previsível e sem variação
# Isso significa que se usar a mesma frase inicial ("Every effort moves you") dez vezes seguidas vai gerar exatamente a mesma resposta nas dez vezes 
# porque ela sempre tomará a mesma decisão matemática