import torch
import torch.nn as nn

# TENSOR ---------------------------------------------------------------------------------------------------------
# Tensor é a estrutura básica de dados do PyTorch, parecida com arrays/vetores (tensor de 1 dimensão) do NumPy mas 
# foi desenvolvido para funcionar de forma eficiente com operações de aprendizado de máquina e com GPU.
x = torch.tensor([1, 2, 3])
print(x)

# Uma matriz também é um tensor 
x = torch.tensor([ # Tensor de 2 dimensões (2x2)
    [1, 2],
    [3, 4]
])
print(x)

# Tensor de mais de 2 dimensões
x = torch.tensor([ # Tensor 2 x 2 x 2
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])
print(x)

# OPERAÇÕES BÁSICAS -----------------------------------------------------------------------------------------------
# Pode-se realizar operações diretamente sobre tensores
a = torch.tensor([1, 2, 3], dtype=torch.float32) # Transformar inteiro em float para poder fazer a média, float32 é um tipo muito comum em deep learning
b = torch.tensor([4, 5, 6])
print(a + b) # Soma elemento por elemento
print(a * b) # Multiplicação elemento por elemento
print(torch.sum(a)) # soma
print(torch.mean(a)) # média
print(torch.max(a)) # Maior
# O PyTorch permite realizar operações matemáticas sobre grandes quantidades de dados de maneira eficiente

# MANIPULAÇÃO DE DIMENSÕES -----------------------------------------------------------------------------------------
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(x.shape) # Dimensão do tensor (ixj)

x = x.reshape(3, 2) # Altera dimensões do tensor mas mantendo a ordem 1,2,3,4,5,6
print(x)

x.view(2,3)
print(x) # Altera dimensões do tensor

x = torch.tensor([1, 2, 3])
print(x.shape) # [3]
x = x.unsqueeze(0) # Adiciona uma dimensão 
print(x.shape) # [1,3]

x = x.squeeze(0) # Remove dimesão que possui tamanho 1
print(x.shape) # [3]

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
x = x.T # Transposição ixj para jxi 1,4,2,5,3,6
print(x)

# OPERAÇÕES MATRICIAIS ------------------------------------------------------------------------------------------------

# Multiplicação de matrizes (não é igual elemento por elemento)
a = torch.tensor ([
    [1, 2],
    [3, 4]
])
b = torch.tensor ([
    [5, 6],
    [7, 8]
])
# Pode ser feita pelas duas formas, o resultado é o mesmo
print(a @ b) 
print(torch.matmul(a, b))

# Broadcasting
# Permite fazer operações entre tensores com dimensões compatíveis mesmo não sendo exatamente iguais
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
b = torch.tensor([10, 20, 30])
print(x + b) # Broadcasting meio que duplica a linha do tensor b para ficar os 2 tensores 2x3

# AUTOGRAD ----------------------------------------------------------------------------------------------------------------------
# Sistema responsável por calcular gradientes automaticamente (fundamental para treinar redes neurais)
# Ele basicamente diz quando a saída muda quando altera-se uma determinada variável

# O pytorch acompanha as operações envolvendo x, para depois podermos calcular o gradiente em relação a ele
x = torch.tensor(3.0, requires_grad=True) # (entrada)
y = x**2 # y = 9
# y foi calculado a partir de x, o pytorch sabe que x=3 -> x^2 -> y=9 (ele guarda sse caminho das operações)
y.backward() # Retropropagação, o pytorch faz o caminho inverso, voltando em cada operação para descobrir quando x muda, quando y muda
             # e é isso que a derivada da função permite fazer
print(x.grad) # tensor(6.) Cálculo de como y varia em relação a x. (loss)
# y = x^2 e a derivada dessa função é dy/dx = 2x, se x=3 então o gradiente é 6 (dy/dx = 2 * 3)

# CONSTRUÇÃO DE MODELOS ------------------------------------------------------------------------------------------------------------
# PyTorch fornece o módulo torch.nn que possui várias ferramentas para construir redes neurais

class Modelo(nn.Module): # Cria-se uma classe que herda nn.Module para que possamos tratar ela como um modelo neural
    def __int__(self): # Função executada quando cria-se o modelo e chama ela depois x = Modelo()
        super().__init__()
        self.linear = nn.Linear(3, 2) # Cria uma camada linear que recebe 3 valores e produz 2 valores
        # y = xW + b (PyTorch cria W e b automaticamente)

    def forward(self, x): # Função que define o que acontece com os dados quando eles passam pelo modelo 
        # saida=modelo(x) chamamos o forward() - x -> forward() -> linear -> saída
        return self.linear(x)

# Modelos podem ter várias camadas
class MeuModelo(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x



# OBSERVAÇÕES ------------------------------------------------------------------------------------------------------------------------
# No projeto de LLM praticamente tudo vira tensor, o computador não trabalha diretamente com palavras

# A diferença entre view() e reshape() envolve a organização da memória do tensor

# Redes neurais são cheias de multiplicações matriciais, uma caamada pode ser y = x @ W + b 
# x = entrada, W = pesos, b = bias e y = saída

# Durante o treinamento a rede neural possui vários parâmetros (peso por exemplo) o modelo faz uma previsão (entrada-modelo-previsão)
# Depois comparamos a previsão com a resposta coreta e isso gera uma loss (previsão-comparação-loss) e o objetivo é diminuir essa loss
# e para isso precisamos descobrir como alterar os pesos, e ai entra os gradientes. Descobri quais pesos contribuíram para o erro e 
# quanto devemos alterá-los

# O treinamento simplificado:
# ENTRADA -> MODELO -> PREVISÃO -> CÁLCULO DO LOSS -> BACKWARD() -> GRADIENTES -> ATUALIZAÇÃO PESOS -> REPETE

