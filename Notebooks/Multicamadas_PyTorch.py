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
            # Função ReLU (zera números negativos)
            # Pega cada número que sai da camada linear e faz a regra (número negativo = 0, número positivo = continua igual)
            # Pode pensar nela como uma espécie de filtro, permite que diferentes neuronios da rede fiquem "ativos"
            # detectou -> positivo -> passa, não detectou -> negativo -> vira 0
            torch.nn.ReLU(), # ReLU(X) = max(0, x) entre 0 e x ele escolhe o maior 
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs), # 20 valores geram 3 valores de saída (logits)
        )
    # O que fazer quando colocar uma entrada x dentro da rede
    def forward(self, x):
        logits = self.layers(x) # Passagem dos dados pela rede
        return logits # Retorna os valores de saída produzidos pela rede


model = NeuralNetwork(50, 3)
print(model)

# Quantidade de números que o modelo deve aprender durante o treinamento
# Pega cada parâmetro do modelo e diz se esse parametro deve ser treinado/calcular gradientes (requires_grad=True) e soma tudo
# Primeiro linear -> 50 * 30 + Segundo linear 30 * 20 +  Terceiro linear 20 * 3 = 2213
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

# Cada linear possui peso e bias
# Uma camada linear multiplica as entradas por uma matriz de pesos e adiciona um vetor de bias
# Mesma ideia z = x * w + b só que com diversos números
# Ela precisa dessa matriz porque deve aprender como cada uma dessas 50 entradas influencia cada uma das 30 saídas
# Fazemos 1500 pesos (50*30) porque cada saída tem um peso para cada entrada, além diss cada um dos 30 neurônios
# possui um bias por isso  50 * 30 + 30 = 1530

# O sequential guarda em posições, então [0] é a primeira camada, e com o .weight pegamos os pesos dessa camada
# Imprime a matriz de 1500 pesos
# print(model.layers[0].weight)

# Mostra as dimensões da camada (saídas, entradas)
# Na primeira camda seriam 30 neuronios de saída e cada um precisa de 50 pesos (cada neuronio recebe as 50 entradas e possui 1 bias)
# Poruqe, por exemplo: saida1 = x1 * peso1 + x2 * peso2...x50 * peso50 + bias
print(model.layers[0].weight.shape)

# É importante entender que ao executar o código os núemros nos pesos (matriz) vão ser um 
# pouco diferentes, visto que, modelo de peso é inicializado com pequenos e aleatorios números, que mudam cada vez
# que instancia a rede, isso é importante para quebrar simetria durante treinamento, sem isso os nós iriam realizar
# as mesmas operações durante backpropagation e assim a rede não iria "aprender" mapeamentos complexos das entradas/saídas.

# Mas podemos fazer a inicialização de números aleatória reproduzivel ao mandar gerador de números aleatórios
# torch.manual_seed(123)
# model = NeuralNetwork(50, 3)
# print(model.layers[0].weight)

# Colocando uma entrada na rede neural
torch.manual_seed(123) # semente para números aleatórios para obter o mesmo resultado
# Um exemplo contendo 50 caracteristicas/entradas
X = torch.rand((1, 50)) # números aleatórios entre 1 e 50 (linha, coluna) (shape do tensor)
# Pega o x e passa pela rede neural, executa automaticamente o forward
out = model(X)
# Mostra os 3 valores produzidos pela rede (logits)
# Ao aparecer grad_fn =<AddmmBackward()> sabemos que o tensor que estamos inspecionando 
# foi criado via operação matricial de multiplicacao (mm) e adição (Add)
# Usamos essa informação para computar gradientes durante backpropagation
print(out)

# se queremos apenas usar a rede sem backpropagtion apenas para predição depois do treinamento, construir o grafo para backpropagation é desnecessário
# Quando há um modelo para fazer predições é melhor usar a função no_grad() para que o PyTorch não fique de olho nos gradientes, isso salva memória
# with torch.no_grad():
# out = model(X)
# print(out)

# Pega os logits e transforma em probabilidade, perceba que a soma deles é mais ou menos =1
with torch.no_grad(): # no_grad() porque é apenas uma previsão, não precisa gradientes
 out = torch.softmax(model(X), dim=1) # logits = probabilidade, assim conseguimos saber qual das 3 classes tem maior probabilidade
print(out)