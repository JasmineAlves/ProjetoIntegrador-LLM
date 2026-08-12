import torch

# Criação de rede neural e analise da estrutura dela

# Classe que herda módulo nn como modelo base para redes neurais
class NeuralNetwork(torch.nn.Module):
    # Construtor da classe, passamos os valores desejados (número de entradas e saídas)
    # self é a própria rede neural sendo criada
    def __init__(self, num_inputs, num_outputs):
        super().__init__() # Inicialização do módulo nn

        # Criação das camadas da rede neural
        # Sequential coloca as camadas/operações em ordem
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            # Essa camada vai pegar o número de entradas e gerar 30 valores
            # Lidamos com muitos pesos e bias mas a ideia é a mesma (entrada * peso + bias)
            torch.nn.Linear(num_inputs, 30),
            # Função ReLU
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits 


model = NeuralNetwork(50, 3)
print(model)