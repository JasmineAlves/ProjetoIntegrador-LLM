import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from SPRINT_5.Notebooks.fluxo_treinamento import train_model_simple, plot_losses
from SPRINT_5.Notebooks.inicializacao_modelo import GPTModel, GPT_CONFIG_124M
from SPRINT_5.Notebooks.pretreinamento import train_loader, val_loader, device
from SPRINT_2.Notebooks.BPE import tokenizer

# EVITAR OVERFITTING
# Código original apresentava descolamento das curvas de treino e validação (memorização)
torch.manual_seed(123)

# Modifica as configurações originais para aumentar o Dropout
# Original drop_rate era 0.1 e testamos com 0.3 (30% dos neurônios desligados no treino)
CONFIG_REGULARIZADA = GPT_CONFIG_124M.copy()
CONFIG_REGULARIZADA["drop_rate"] = 0.3

# Instancia o modelo com a nova configuração
model = GPTModel(CONFIG_REGULARIZADA)
model.to(device)

# Configura o otimizador com Weight Decay maior que pune pesos muito grandes 
# Original era 0.1 vamos testar 0.5
TAXA_APRENDIZADO = 0.0004
DECAIMENTO_PESOS = 0.5

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=TAXA_APRENDIZADO, 
    weight_decay=DECAIMENTO_PESOS
)

# Treinamento
num_epochs = 10 
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Avaliação e gráfico
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# A curva de validação (Val loss) deve demorar mais para descolar da curva de treinamento (Train loss)
# Como o dataset é minúsculo o overfitting ainda vai acontecer mas o gap entre as curvas no gráfico vai ser menor que no teste original