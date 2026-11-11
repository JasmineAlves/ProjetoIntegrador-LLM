import torch
from SPRINT_4.Notebooks.GPT_model import model
from SPRINT_5.Notebooks.inicializacao_modelo import GPTModel, GPT_CONFIG_124M
from SPRINT_5.Notebooks.pretrainamento import device
from SPRINT_5.Notebooks.fluxo_treinamento import optimizer

# SALVAR
# Melhro forma de salvar um modelo para usar/continuar treinando é salvar como um dicionario
# que mapeia cada camada com seus parâmteros
# torch.save(model.state_dict(), "model.pth")

# CARREGAR
# Pode-se carregar os pesos do modelo após salvar em uma nova insttância do GPTmodel
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(torch.load("model.pth", map_location=device))
# Desliga dropout porque durante inferencia não queremos deixar de lado nenhuma informação aprendida pela rede
model.eval()

# Se queremos continuar pre-treinando o modelo depois, salvar o estado do otimizador é recomendado
# AdamW usa historico de dados para ajustar as taxas de aprendizado para cada parametro do modelo dinamicamente
# Sem isso o modelo pode falhar em convergir propriamente, ou seja, pder a habilidade de gerar texto coerente
# Salva modelo e otimizador
torch.save({
 "model_state_dict": model.state_dict(),
 "optimizer_state_dict": optimizer.state_dict(),
 },
 "model_and_optimizer.pth"
)

# Podemos restaurar os estados do modelo e otimizador primeiro carregando e salvando os dados
# e depois usando o load_stat_dict método
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
