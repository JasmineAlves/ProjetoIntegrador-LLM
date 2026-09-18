import torch
import torch.nn as nn

# Classe compactada para chamar quando precisar do mecanismo de self-attention com pesos treináveis
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()

        # Cria 3 matrizes de pesos que serão usadas para transformar
        # cada embedding em Query, Key e Value
        # Foward propagation, dados entram no modelo e uma saída é calculada
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):

        # Transforma os embeddings de entrada em Keys
        keys = x @ self.W_key

        # Transforma os embeddings de entrada em Queries
        queries = x @ self.W_query

        # Transforma os embeddings de entrada em Values
        values = x @ self.W_value

        # Compara cada Query com todas as Keys
        # O resultado são os "attention scores" eles mostram o quanto cada palavra combina com as outras
        attn_scores = queries @ keys.T  # omega

        # Divide os scores por √d_k para evitar valores muito grandes
        # Depois o softmax transforma os scores em pesos que somam 1
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )

        # Usa os pesos para combinar os Values
        # Palavras com maior peso contribuem mais para o contexto
        context_vec = attn_weights @ values

        # Retorna os vetores de contexto finais
        return context_vec



# Utiliza nn.Linear para criar e inicializar os pesos de forma adequada para o treinamento da rede neural
class SelfAttention_v2(nn.Module):

  # sem bias (y= x @ w), não queremos adicionar o valor extra 
  def __init__(self, d_in, d_out, qkv_bias=False):
    super().__init__()
    # Cria a camada que transforma a entrada em Queries (Q)
    # A Query representa "o que estou procurando?"
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    # Cria a camada que transforma a entrada em Keys (K)
    # A Key representa "qual informação eu tenho?"
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    # Cria a camada que transforma a entrada em Values (V).
    # O Value é a informação que será utilizada no resultado final
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

  def forward(self, x):
    # Transforma a entrada em Keys
    keys = self.W_key(x)
    # Transforma a entrada em Queries
    queries = self.W_query(x)
    # Transforma a entrada em Values
    values = self.W_value(x)
    # Compara Queries com Keys através da multiplicação de matrizes
    # Quanto maior o resultado maior a relação entre elas
    attn_scores = queries @ keys.T

    # Converte os scores em pesos de atenção usando softmax
    # A divisão pela raiz da dimensão das Keys ajuda a deixar os valores mais estáveis
    attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
    )
    # Usa os pesos de atenção para decidir quanto de cada Value deve contribuir para formar o resultado final
    context_vec = attn_weights @ values
    # Retorna o vetor de contexto com as informações mais relevantes
    return context_vec