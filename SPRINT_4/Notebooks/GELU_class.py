import torch
import torch.nn as nn
import cfg
import matplotlib.pyplot as plt

# Implementação da função de ativação GELU
class GELU(nn.Module):
    # Inicialização da classe
    def __init__(self):
      super().__init__()

    def forward(self, x):
      # Cálculo da GELU
      return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
            ))

# Pequena rede neural com 2 camadas lineares e uma função de ativação GELU
# Processa os dados dos tokens para ajudar o modelo a aprender padrões mais complexos
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
           # Expande os dados, aumenta a dimensão de 768 para 3072 valores
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # Aplica a GELU, transforma os valores ajudando a rede a aprender relações complexas
            GELU(),
            # Reduz os dados, volta de 3072 para 768 valores
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),

            # A rede expande esse vetor para conseguir processar informações em uma representação maior e depois retorna ao tamanho original
        )
    def forward(self, x):
       return self.layers(x)

# Inicializa um feedforward módulo, com token embedding de 768
ffn = FeedForward(cfg.GPT_CONFIG_124M)
# Coloco com 2 amostras e 3 tokens cada
x = torch.rand(2, 3, 768)
out = ffn(x)
# O tamanho da saída é igual do da entrada em questão de tensor
print(out.shape)


# Comparação gráfica da GELU com a ReLU ----------------------------------------------------------------------------------------
# Cria as duas funções
gelu, relu = GELU(), nn.ReLU()
# Cria os valores de entrada, 100 números com valores entre -3 e 3
x = torch.linspace(-3, 3, 100)
# Calcula as saídas
y_gelu, y_relu = gelu(x), relu(x)
# Plota os gráficos
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()
