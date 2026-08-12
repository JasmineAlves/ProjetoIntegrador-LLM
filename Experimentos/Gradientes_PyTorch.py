import torch
import torch.nn.functional as F
from torch.autograd import grad # Importando a função para calcular gradientes

# Agora o código não só calcula a previsão e a loss, mas também pergunta como deve mudar w1 e b para diminuir a loss

y = torch.tensor([1.0]) # resposta esperada
x1 = torch.tensor([1.1]) # entrada

# Construção de grafos internos por padrão
# Aqui o require_grad permite acompanhar as operações envolvendo w1 e b (para calcularmos o gradinte deles depois)
w1 = torch.tensor([2.2], requires_grad=True) 
b = torch.tensor([0.0], requires_grad=True) 

# Modelo pega a entrada e multiplica pelo peso (1.1 * 2.2 = 2.42)
z = x1 * w1 + b  # Entrada "líquida" ou pré-ativação (1.1 * 2.2 + 0 = 2.42)

# Ativação sigmoid(0.918), 91,8% de confiança
a = torch.sigmoid(z) 

# Cálculo do erro do modelo (saída esperada e a)
loss = F.binary_cross_entropy(a, y)

# Sem o retain_grad o PyTorch iria destruir o grafo depois de calcular os gradientes para liberar memória
# Precisamos dele porque usamos o grafo 2 vezes em w1 e b

# grad() vai calcular o gradiente da loss em relação a w1 (se mexer um pouco em w1, quanto a loss muda)
# Gradiente nos diz a direção e intensidade dessa mudança da loss (aumenta, diminui, quase não muda)
grad_L_w1 = grad(loss, w1, retain_graph=True)
# Mesma ideia
grad_L_b = grad(loss, b, retain_graph=True)
# Gradientes calculados
print(grad_L_w1)
print(grad_L_b)

# Existe uma forma mais automatizada para calcular os gradientes
# loss.backward() o PyTorch calcula o gradiente de todos os nós do grafo e 
# vão ser armazenados pelo .grad -> print(w1.grad)

# Gradiente negativo significa que aumentar w1 ou b -> loss diminui
# Gradiente positivo significa que aumentar w1 ou b -> loss aumenta
# Pode-se usar o números dos gradientes para atualizar os pesos, 
# geralmente a regra simplificada é novo_parametro = parametro_antigo - taxa_aprendizagem * gradiente
# Taxa de aprendizagem é um valor que você escolhe (ou algoritmo define), como exemplo, pode-se usar 0.1