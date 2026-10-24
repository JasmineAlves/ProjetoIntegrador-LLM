# Análise de com o fluxo de dados entra e saí de um modelo GPT
import tiktoken # transforma textos em token
import torch
import dummy_gpt_model
import cfg

# Carrega o tokenizador, divide o texto em tokens e atribui ID para cada token
# Uma palavra pode virar um único token ou ser dividida em partes menores, depende do vocabulário e das regras de tokenização
tokenizer = tiktoken.get_encoding("gpt2")
# Cria lista vazia para guardar as represenntações numericas das frases
batch = []
# Cria 2 variáveis de texto/string
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
# Transforma as frases em tokens e organiza os IDs em um tensor
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
# As duas frases possuem quatro palavras, mas isso não garante que terão a mesma quantidade de tokens 
# O tokenizador trabalha com tokens não necessariamente com palavras inteiras

# stack() junta os 2 tensores em um único tensor com dimensão nova [2,4] = [sequencias no lote, quantos tokens cada sequencia tem]
batch = torch.stack(batch, dim=0)
# Mostra IDs dos tokens no tensor
print(batch)

# Inicia uma nova instancia 124-milhão-parametro e colocamos no lote tokenizado
# Semente aleatória na inicialização usada para tornar certos resultados reproduzíveis
torch.manual_seed(123)
# Chamando o dummy e as configurações do gpt definidas
model = dummy_gpt_model.DummyGPTModel(cfg.GPT_CONFIG_124M)
# Chama internamente o metodo forward do modelo que processa o lote pelas camadas definidas na classe
logits = model(batch)
# São pontuações numéricas que o modelo produz para cada token possível do vocabulário
# Como o vocabulário possui 50.257 tokens o modelo gera 50.257 pontuações para cada posição de entrada
print("Output shape:", logits.shape) # [2, 4, 50257]
print(logits)