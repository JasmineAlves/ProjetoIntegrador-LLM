import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from SPRINT_5.Notebooks.fluxo_treinamento import train_model_simple, plot_losses
from SPRINT_5.Notebooks.inicializacao_modelo import GPTModel, GPT_CONFIG_124M
from SPRINT_5.Notebooks.pretreinamento import train_loader, val_loader, device
from SPRINT_2.Notebooks.BPE import tokenizer

# TAXA DE APRENDIZADO ALTA
torch.manual_seed(123)

# Instancia o modelo padrão
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

# Configura o otimizador com uma Taxa de Aprendizado muito alta
# Original era 0.0004 e vamos testar com 0.01 (25x maior)
TAXA_APRENDIZADO = 0.01 

# Otimizador AdamW
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=TAXA_APRENDIZADO, 
    weight_decay=0.1
)

# Treinameento, reduz para 5 épocas pois se a taxa for muito alta o modelo "quebra" rápido
num_epochs = 5 
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Avaliação e gráfico
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# A loss de treinamento não deve cair suavemente. Pode explodir (valores muito altos) ou oscilar
# O texto gerado no final de cada época provavelmente será um jargão repetitivo e sem sentido