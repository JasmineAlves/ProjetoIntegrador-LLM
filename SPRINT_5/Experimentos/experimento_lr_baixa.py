import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from SPRINT_5.Notebooks.fluxo_treinamento import train_model_simple, plot_losses
from SPRINT_5.Notebooks.inicializacao_modelo import GPTModel, GPT_CONFIG_124M
from SPRINT_5.Notebooks.pretreinamento import train_loader, val_loader, device
from SPRINT_2.Notebooks.BPE import tokenizer

# TAXA DE APRENDIZADO BAIXA
torch.manual_seed(123)

# Instancia o modelo padrão
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

# Configura o otimizador com uma taxa de aprendizado muito baixa
# Original era 0.0004 mas vamos testar 1e-5 (0.00001 - 40x menor)
TAXA_APRENDIZADO = 1e-5

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=TAXA_APRENDIZADO, 
    weight_decay=0.1
)

# Treinamento, mantem 10 épocas para ver o quão devagar ele aprende
num_epochs = 10 
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Avaliação e gráfico
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# As losses de treinamento e validação devem cair, mas muito lentamente e no final da 10ª época 
# a loss ainda será alta indicando que o modelo precisaria de muito mais épocas para aprender algo útil (Underfitting)