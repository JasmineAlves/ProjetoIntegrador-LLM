import torch
import torch.nn as nn
from GELU_class import GELU

# Implementa uma rede neural profunda com 5 camadas
# Cada um com camada linear e ativação GELU
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
        nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
        GELU()),
        nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
        GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Computa a saída da camada presente
            layer_output = layer(x)
            # Verifica se o shortcut pode ser aplicado
            if self.use_shortcut and x.shape == layer_output.shape:
              x = x + layer_output
            else:
              x = layer_output
        return x

# Inicializa a rede sem shortcut ---------------------------------------------------------------------------------------------
# Cada camada inicia aceitando um exemplo com 3 entradas e retorna 3 saídas, a ultima camada retorna 1 saída apenas
# [quant entradas, quant de saida nas camadas do meio, ... , ... , ... , quant saída na camada final]
layer_sizes = [3, 3, 3, 3, 3, 1]
# Entrada para exemplo, [[]] indica que os dados estão em um batch com uma amostra com 3 características (1,3)
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
# Inicia o modelo sem shortcut
model_without_shortcut = ExampleDeepNeuralNetwork(
 layer_sizes, use_shortcut=False
)

# Computa os gradientes no modelo backwar pass, mostra como os pesos estão relacionados ao erro
def print_gradients(model, x):
    output = model(x) # forward pass, passa os dados pela rede e obtém a saída
    # Define o resultado esperado como 0
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    # Calcula a perda baseado em quanto perto o alvo e a saída estão
    loss = loss(output, target)
    # Backward pass para calcular os gradientes que indicam como os pesos influenciam o erro
    loss.backward()
    # Percorre os parametros treinaveis do modelo
    for name, param in model.named_parameters():
        # Seleciona só os pesos (sem vieses)
        if 'weight' in name:
          # Mostra o nome de cada conjunto d pesos e a média do valor dos seus gradientes
          print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# Mostra como os gradientes ficam menores quando progredimos da ultima camada para a primeira (vanishing gradient problem)
print_gradients(model_without_shortcut, sample_input)


# Inicializa a rede com shortcut ----------------------------------------------------------------------------------------------
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
 layer_sizes, use_shortcut=True
)
# Valor da ultima camada ainda é maior, porém o valor do gradiente estabiliza ao progredir para primeira camada
print_gradients(model_with_shortcut, sample_input)