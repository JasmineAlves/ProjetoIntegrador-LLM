import torch
from SPRINT_5.Notebooks.inicializacao_modelo import GPTModel, GPT_CONFIG_124M
from src.GPT2_config import model_configs, params
from src.carregador_pesos import load_weights_into_gpt
from SPRINT_5.Notebooks.pretrainamento import device
from SPRINT_5.Notebooks.text_idtoken import text_to_token_ids, token_ids_to_text
from SPRINT_2.Notebooks.BPE import tokenizer
from SPRINT_5.Notebooks.geracao_texto import generate

# Modelo gpt2-small
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

# Usar o a configuraçao atualizada para iniciar uma nova instancia GPTModel
gpt = GPTModel(NEW_CONFIG)
# Modo avaliação
gpt.eval()

# Carregar os pesos do modelo OpenAI no nosso GPTModel
load_weights_into_gpt(gpt, params)
gpt.to(device)

# Gerar novo texto usando a função de geração
torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
# Pode-se observar que o modelo funcionou perfeitamente pois gerou um texto coerente
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))