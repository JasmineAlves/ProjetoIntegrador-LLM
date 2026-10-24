import torch
import torch.nn as nn

# Classe de camada de normalização
# Opera na ultima dimensão do tensor de entrada x que representa a dimensão embedding
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Constante para evitar divisões por 0 durante normalização
        self.eps = 1e-5
        # scale e shift são 2 parâmetros treináeis (mesma dimensão que a entrada), LLM ajusta durante treinamento
        # se melhora a performance do modelo em sua tarefa de treinamento, permite o modelo aprender escala e mudança
        # que melhor fica para os dados sendo processados.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # Usamos unbiased=False para calcular a variância dividindo pelo número total de valores e manter a compatibilidade com o GPT-2
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift