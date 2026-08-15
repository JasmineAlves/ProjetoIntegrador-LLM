import torch
import Carregadores_dados_PyTorch
import torch.nn.functional as F

# Modificando o loop de treinamento para rodar na GPU
# O códig segue o mesmo de multicamadas, mudando apenas 3 linhas, o reusltado 
# é similiar ao usado com a CPU.
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__() 
     
        self.layers = torch.nn.Sequential(
                   
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(), 
          
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
      
            torch.nn.Linear(20, num_outputs), 
        )
 
    def forward(self, x):
        logits = self.layers(x) 
        return logits 

# Esses tensores vão ser carregados na CPU por padrão
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

# Usamos o to() método, mesmo método que usamos para mudar o tipo de dado do
# tensor. Com ele conseguimos transferir esses tensores na GPU
tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda")
# cuda=0 quer dizer que os tensores residem na 1° GPU
# Se tivesse multiplas GPU poderia dizer para qual o tensor deve ir ex: cuda:1
# Todos os tensores devem estar no mesmo dispositivo, não pode um estar na CPU e
# outro na GPU.
print(tensor_1 + tensor_2)


torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
# Define uma variavel que usa a GPU por padrão
device = torch.device("cuda")
# Podemos usar o seguinte código para que o código seja executavel na CPU se a 
# GPU não estiver disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transfere o modelo para a GPU
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3
for epoch in range(num_epochs):

 model.train()
 for batch_idx, (features, labels) in enumerate(Carregadores_dados_PyTorch.train_loader):
  # Transfere os dados para a GPU
  features, labels = features.to(device), labels.to(device)
  logits = model(features)
  loss = F.cross_entropy(logits, labels) # Loss function

 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

 ### LOGGING
 print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
 f" | Batch {batch_idx:03d}/{len(Carregadores_dados_PyTorch.train_loader):03d}"
 f" | Train/Val Loss: {loss:.2f}")
 model.eval()
 # Insert optional model evaluation code

