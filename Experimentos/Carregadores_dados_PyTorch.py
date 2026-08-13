import torch
import Multicamadas_PyTorch
from torch.utils.data import Dataset # Classe dataset
from torch.utils.data import DataLoader # Classe de dataloader
import torch.nn.functional as F

# Dados para treinar e testar a rede neural
# Aqui criamos alguns exemplos para a rede "aprender"
# Dataset -> dados de treinamento -> modelo aprende  e dados testes -> modelo é validado

# Classe que herda dataset
class ToyDataset(Dataset):
     def __init__(self, X, y):
        # Guarda os dados de entrada dentro do dataset
        self.features = X # Caracteristicas
        self.labels = y # Respostas corretas
    # Meio que é para saber quando alguem pedir o exemplo número N, que deve devolver
    # Mantém x e y juntos
     def __getitem__(self, index):
        # Ex: alguem pediu train_ds[0]
        one_x = self.features[index] # feature[0] -> [-1.2, 3.1]
        one_y = self.labels[index] # labels[0] -> [0]
        return one_x, one_y # ([-1.2, 3.1], 0)
     # Quantos exemplos existem nesse dataset
     def __len__(self):
        return self.labels.shape[0] # shabe de y é (5,) [0, 0, 0, 1, 1]

     
# 5 exemplos de treinamento, cada um com 2 caracteristicas -----------------------------------------------------
X_train = torch.tensor([
 [-1.2, 3.1],
 [-0.9, 2.9],
 [-0.5, 2.6],
 [2.3, -1.1],
 [2.7, -1.5]
])
# Cada exemplo tem uma classe (0 ou 1) que é a resposta correta
y_train = torch.tensor([0, 0, 0, 1, 1])

# Lembrando que o modelo pode errar, por isso calculamos a loss

# Criamos 2 exemplos separados para teste ---------------------------------------------------------------
# Usamos exemplos que ele não viu paar sabermos se o modelo consegue identifcar
# 2 exemplos com 2 caracteristicas cada
X_test = torch.tensor([
 [-0.8, 2.8],
 [2.6, -1.6],
])
# Resposta correta de cada exemplo
y_test = torch.tensor([0, 1])


# Criando dataset de treinamento ----------------------------------------------------
# Guarda os 5 exemplos(x) e suas respostas corretas(y)       
train_ds = ToyDataset(X_train, y_train)

# Criando dataset de teste -----------------------------------------------------
# 2 exemplos(x) e suas respostas corretas(y)
test_ds = ToyDataset(X_test, y_test)


# Semente para números aleatórios
# Faz com que o primeiro embaralhamento (shuffle) seja reproduzível
torch.manual_seed(123)
# Criando dataloader de treinamento ---------------------------------------------------------------
train_loader = DataLoader(
    dataset=train_ds, # usar o dataset criado anteriormente
    batch_size=2, # lote, entre 2 exemplos de cada vez (vai entregar 3 batchs visto que tem 5 exemplos)
    shuffle=True, # embaralhe os exemplos antes de formar o batch, evita que modelo fique dependente da ordem dos dados
    num_workers=0, # quantos processos auxiliares serão usados para carregar os dados (0 = carregar no processo principal)
    drop_last=True # Com 5 não divisivel igualmente por 2 o ultimo batch fica menor e isso pode causar disturbiu na
    # convergencia durante treinamento, o drop_last serve para tirar fora o ultimo batch
)

# Criando dataloader de teste ---------------------------------------------------------------------
test_loader = DataLoader(
    dataset=test_ds, # usar o dataset criado anteriormente
    batch_size=2,
    shuffle=False, # durante o teste não precisa embaralhar porque o objetivo é apenas avaliar o modelo
    num_workers=0
)

# Loop passa por cada batch e enumera (índice + batch -> indice 0 + batch 1),
# cada batch contem caracteristia e respostas corretas (x, y)
for idx, (x, y) in enumerate(train_loader):
 print(f"Batch {idx+1}:", x, y) # Mostra cada batch
# Se fizer 2 FOR iguais, o segundo vai embaralhar novamente os exemplos

# Números aleatórios sejam reproduziveis (pesos)
torch.manual_seed(123)
# Cria rede neural
model = Multicamadas_PyTorch.NeuralNetwork(2, 2) # 2 entradas -> 2 saídas

# Atualizar os pesos da rede depois de calcular os gradientes
optimizer = torch.optim.SGD(
 model.parameters(), lr=0.5 # Pega todos os parametros treinaveis, lr = taxa de aprendizado (learning rate)
 # É meio que o tamanho do passo que o otimizador da ao atualizar os pesos
 # peso novo = peso antigo - taxa * gradiente
)
num_epochs = 3 # quantas vezes passamos pelos dados de treinamento

# Passa 3 vezes 
for epoch in range(num_epochs):

 model.train() # coloca rede em modo treinamento

    # Percorre os batches, cada batch com 2 exemplos (caracteristica e resposta correta)
 for batch_idx, (features, labels) in enumerate(train_loader):
    # Dados entram na rede passando pelas camadas e ReLU, gerando o logits
        logits = model(features)

        # Calcula perda, cross_entropy() compara logits com resposta correta
        # Se o modelo confundiu-se bastante a loss é maior (Ex: classe 0 - 95% e classe 1 - 5% e a resposta correta era classe 1)
        loss = F.cross_entropy(logits, labels)

        # Zera os gradientes do batch anterior para calcular os dos novos
        optimizer.zero_grad()

        # Calcula os gradientes (quanto cada peso contribui para o erro)
        loss.backward()

        # Atualiza os pesos usando os gradientes calculados
        optimizer.step()

        # LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
        f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
        f" | Train Loss: {loss:.2f}")

 # Coloca o modelo em modo avaliação
 model.eval()
 