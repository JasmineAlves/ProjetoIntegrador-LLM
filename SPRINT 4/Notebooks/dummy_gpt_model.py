# Visualizar a estrutura geral do modelo, é meio que um modelo provisório do GPT
import torch
import torch.nn as nn

# Estrutura geral
class DummyGPTModel(nn.Module):
    # Inicialização
    # cfg é o dicionário de configurações do modelo
    def __init__(self, cfg):
        super().__init__()
        # Cria uma camada que transforma cada token em um vetor de números, (cada vetor tem emb_dim números)
        # representa qual token está sendo usado
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Embedding de posições, cria vetores para representar a posição de cada token no texto
        # representa onde esse token está no texto
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Cria camada de dropout, desativa drop_rate% das ativações, evita overfitting
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # Cria os blocos transformers
        # nn.sequential organiza várias camadas em sequencia
        self.trf_blocks = nn.Sequential(
           # Cria lista [], desempacota lista para passar blocos individualmente *
            *[DummyTransformerBlock(cfg)
                # Produz n_layers repetições e em cada uma cria-se um bloco
                for _ in range(cfg["n_layers"])]
         )
        # Cria componente para normalizar os valores antes da saída
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # Cria camada de saída, transforma vetores finais em pontuações para todos os tokens do vocabulario
        # Pontuações são chamadas de logits, indica o quanto o token é favorecido com próximo token, antes do softmax
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    # Define o caminho que os dados percorrem dentro do modelo
    # in_idx = ID dos tokens de entrada
    def forward(self, in_idx):
        # Tamanho da entrada (X sequencia no lote, Y tokens que tem essa sequencia ) (X frases de Y tokens)
        batch_size, seq_len = in_idx.shape
        # Transforma os tokens em vetores (X sequencias, Y tokens, Z números para representar cada token)Z) sendo o Z o emb_dim
        tok_embeds = self.tok_emb(in_idx)
        # Cria os números das posições, transforma cada posição em um vetor de emb_dim números
        pos_embeds = self.pos_emb(
        # Cria uma sequencia de posições (Ex: se seq_len = 4 então [0,1,2,3])
        # device coloca essas posições no mesmo dispositivo de entrada (CPU/GPU)
        torch.arange(seq_len, device=in_idx.device)
        ) # saída será (posições, emb_dim)

        # Soma a representação do token com a representação da sua posição
        # a representação posicional (posições, emb_dim) é aplicada a cada sequência do lote (num_sequencia, posições, emb_dim).
        # x contém informações sobre qual é o token e onde ele aparece
        x = tok_embeds + pos_embeds
        # Aplica dropout durante treinamento
        x = self.drop_emb(x)
        # Realiza o processamento por meio dos blocos Transformer
        # que incluem mecanismos de atenção e redes neurais
        x = self.trf_blocks(x) # Bloco provisório
        # Aplica normalização final
        x = self.final_norm(x) # Bloco provisório
        # A camada linear transforma os vetores de emb_dim números em vocab_size pontuações para cada posição da sequência
        # (X, Y, Z) -> (sequencia, posições, vocab_size)
        logits = self.out_head(x)
        # O modelo produz uma pontuação para cada token do vocabulário em cada uma das X posições
        return logits


# Classes de demonstração sem função nenhuma (provisórias)
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
      super().__init__()

    def forward(self, x):
      return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
       super().__init__()

    def forward(self, x):
       return x