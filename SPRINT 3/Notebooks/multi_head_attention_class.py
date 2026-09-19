import torch
import torch.nn as nn
from causal_attention_class import CausalAttention

# Essa classe processa sequencialmente no método forward as heads
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False
    ):
        super().__init__()

        # Cria uma lista com várias "cabeças" de atenção. Cada head é uma CausalAttention independente
        # Isso permite que cada uma aprenda a prestar atenção em diferentes tipos de informações
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in,           # tamanho da entrada
                    d_out,          # tamanho da saída de cada head
                    context_length, # quantidade de tokens
                    dropout,        # dropout
                    qkv_bias        # usa ou não bias
                )

                # Repete a criação da CausalAttention de acordo com o número de heads
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        # Passa a mesma entrada "x" por todas as heads. Cada head faz seu próprio cálculo de atenção
        # Depois, torch.cat() junta os resultados
        # dim=-1 significa que vamos juntar os resultados pela última dimensão (dimensão dos vetores)
        return torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

# se num_heads = 2 e d_out = 2 então temos um vetor de contexto de 4 dimensões (2 * 2 = 4)


# Essa classe processa as heads em paralelo
# faz tudo dentro de um único objeto, sendo mais eficiente e organizada
# Junta causal attention e multi-head attention
class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # d_out precisa ser divisível pelo número de heads
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Tamanho de cada head
        self.head_dim = d_out // num_heads
        # Camadas que transformam a entrada em Query, Key e Value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Camada final para transformar o resultado
        self.out_proj = nn.Linear(d_out, d_out)
        # dropout
        self.dropout = nn.Dropout(dropout)

        # Cria a máscara para impedir que um token veja os tokens futuros
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

    def forward(self, x):
        # Pega tamanho do lote, número de tokens e tamanho da entrada
        b, num_tokens, d_in = x.shape
        # Cria Key, Query e Value a partir da entrada
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Divide os vetores entre as várias heads
        keys = keys.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        values = values.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        # Coloca o número de heads antes dos tokens
        # transpose(0) uma forma de fazer várias multiplicações de matrizes ao mesmo tempo (uma para cada head)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calcula o quanto cada token deve prestar atenção aos outros
        attn_scores = queries @ keys.transpose(2, 3)
        # Pega a parte da máscara correspondente ao tamanho da entrada
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Impede que um token veja informações do futuro
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Converte os scores em probabilidades
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )

        # Aplica dropout aos pesos de atenção
        attn_weights = self.dropout(attn_weights)
        # Usa os pesos para combinar os valores (Value)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # Junta novamente as várias cabeças
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        # Faz a transformação final do resultado
        context_vec = self.out_proj(context_vec)
        return context_vec
