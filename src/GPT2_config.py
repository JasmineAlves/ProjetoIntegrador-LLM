import urllib.request
from src.gpt_download import download_and_load_gpt2

#  Instalar o codigo de download do GPT
""" url = (
 "https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch05/"
 "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename) """

# Inspecionar o conteudo do arquivo para garantir que foi salvo corretamente
settings, params = download_and_load_gpt2(
   model_size="124M", models_dir="gpt2"
)

# Inspecionar conteudo do settings e params
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

# Inspecionar os tensores de peso
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

# OBS: Settings funciona de maneira similar com o GPT_CONFIG_124M criado

# Dicionário que lista as diferenças entre diferentes modelos GPT
# Usa-se ele para atualizar o GPT_CONFIG_124M com os valores corretos
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}