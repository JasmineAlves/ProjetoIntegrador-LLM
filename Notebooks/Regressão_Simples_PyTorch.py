import torch
import torch.nn.functional as F

# Implementação da passagem direta (etapa de previsão)
# Classificador de regressão logística simples (rede neural de camada única)
# Ele retorna uma pontuação de 0 ou 1 (é meio que um rótulo de classe verdadeiro quando computando a perda/loss)
# Basicamente, em palavras mais fáceis, é um modelo que responde se a entrada (x1) pertence a classe 0 ou a classe 1

y = torch.tensor([1.0]) # Rótulo/identificação verdadeiro (resposta correta)
x1 = torch.tensor([1.1]) # Entrada x1 = 1.1
w1 = torch.tensor([2.2]) # Parâmetro de peso, w1 = 2.2
# Pequeno valor adicional que o modelo pode ajustar
b = torch.tensor([0.0])  # Unidade Bias

# Modelo pega a entrada e multiplica pelo peso (1.1 * 2.2 = 2.42)
z = x1 * w1 + b  # Entrada "líquida" ou pré-ativação (1.1 * 2.2 + 0 = 2.42)

# A função sigmoid() transforma um número qualquer em um valor entre 0 e 1 (0.12, 0.8 e etc) (probabilidade entre 0 e 1)
a = torch.sigmoid(z) # Ativação sigmoid(0.918)
# O modelo está dizendo, que acredita que a entrada seja da classe 1 com aproximadamente 91,8% de confiança

# Agora entra a perda que calcula o erro do modelo, o bianry_cross_entropy() calcula o quanto a previsão
# está distante da resposta esperada, basicamente, o quão uim foi a previsão
loss = F.binary_cross_entropy(a, y) 

# Essa sequência é o que pode-se chamar de grafo de computação (imagina como um fluxograma de contas)
# Pensando em uma sequência de computação como um grafo de computação
# A entrada é multiplicada pelo peso e passa por uma função de ativação depois que adiciona o bias. A loss/perda então é 
# calculada ao comparar a saída do modelo e uma saída (y) já rotulada.
# Ao construir esse grafo em segundo plano, podemos utilizar ele para calcular os gradientes de uma função de loss
# em relação ao parâmetros do modelo (nesse caso w1 e b) para treinar o modelo.
# Ou seja, criamos o grafo porque queremos depois descobrir como devemos alterar o w1 para a loss diminuir.
# Ai que entram os gradientes, que vai nos dizer para que direção devo mexer nos parâmetros (w1, b) para diminuir a perda
# Para isso precisamos saber como a loss muda quando w1 ou b mudam.

# PyTorch consegue construir esse grafo de operações e depois calcular automaticamente os gradientes -> Autograd
# Temos duas etapas: o forward pass que o modelo vai pra frente (entrada - pesos - bias - ativação - previsão - loss) e
# backward pass (loss - gradientes - bias - pesos)