import torch
from SPRINT_2.Notebooks.BPE import tokenizer
from SPRINT_2.Notebooks.Entradas_Alvos import create_dataloader_v1
from SPRINT_5.Notebooks.inicializacao_modelo import GPT_CONFIG_124M
from SPRINT_4.Notebooks.GPT_model import model

# Lembrando que esse texto é pequeno para treinar LLM porem ppara fins educAacionais ele é bom porqur
# leva apenas minutos para rodar o código e mais tarde será usado pesos pré treinados do OpenAI
# Carrega o dataset
file_path = "SPRINT_2/Notebooks/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
  text_data = file.read()

# Verifica o numero de caracteres e tokens do dataset
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# Usa 90%  dos dados de treinamento e 10% que sobrou é usado como dados de validação para o modelo avaliar durante treinamento
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
# Carregamento de dados, pega esses textos e os transforma em pequenos pacotes com 2 blocos por vez
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# Garantir que os carregadores de dados foram criados corretamente
# [2, 256] x9, ou seja, são 9 (90%) sets de treino batches com 2 exemplos e 256 tokens
print("Train loader:")
for x, y in train_loader:
 print(x.shape, y.shape)
 # 1 (10%) batch com 2 entradas de exemplo
print("\nValidation loader:")
# Dados de entrada e de saída tem a mesma forma (tamanho batch * num tokens)
for x, y in val_loader:
 print(x.shape, y.shape)

# Calcula o cross entropy loss do batch
def calc_loss_batch(input_batch, target_batch, model, device):
    # Pega pacote de textos (perguntas e gabarito)
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # Modelo lê o texto e chuta os valores
    logits = model(input_batch)
    # Erro comparando os palpites do modelo com as respostas certas
    loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), target_batch.flatten()
    )
    # Valor do erro
    return loss

# Calcula a loss de todos os batches de exemplo dados pelo data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    # Soma total de erros zerada
    total_loss = 0.
    # Verifica se a caixa de pacotes está vazia
    if len(data_loader) == 0:
      return float("nan")
    # Se não especificou um limite de pacotes paar ler
    elif num_batches is None:
       # Avalia todos os pacotes que existem
       num_batches = len(data_loader)
    # Se foi especificado a quantidade de pacotes mas tem menos pacotes na realidade (limite)
    else:
       # Ajusta para o valor do limite para não dar erro
       num_batches = min(num_batches, len(data_loader))
    # Puxa cada pacote por vez (perguntas e gabaritos)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # Verifica se está dnetro do limite de pacotes
        if i < num_batches:
            # Calcula nota de erro do pacote
            loss = calc_loss_batch(
               input_batch, target_batch, model, device
            )
            # Pega o valor de erro do pacote e soma no erro total
            total_loss += loss.item()
        else:
           break
    # Pega todos os erros somados e divide pela quantidade de pacotes (média final de erro do modelo)
    return total_loss / num_batches 

# OBS: PACOTES = BATCHES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Sem gradiente para melhor eficiencia (não estamos treinando ainda)
with torch.no_grad():
 # Calculas notas de erros e garante que os dados são carregados no mesmo dispositivo que o modelo LLM
 train_loss = calc_loss_loader(train_loader, model, device)
 val_loss = calc_loss_loader(val_loader, model, device)
# Loss alta porque o modelo não foi treinado ainda
print("Training loss:", train_loss)
print("Validation loss:", val_loss)