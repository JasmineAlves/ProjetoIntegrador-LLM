import torch
import torch.nn as nn

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        # Tamanho dos vetores de saída da atenção
        self.d_out = d_out

        # Essas 3 camadas transformam os dados de entrada em
        # Query = "o que estou procurando?"
        # Key   = "o que cada token oferece?"
        # Value = "qual informação cada token possui?"
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout: durante o treinamento desliga aleatoriamente alguns pesos de atenção para evitar overfitting
        self.dropout = nn.Dropout(dropout)

        # Cria a máscara causal. Ela possui 1 acima da diagonal e 0 no restante.
        # Os 1 indicam posições que devem ser bloqueadas, ou seja, tokens do "futuro"
        # evita erros de dispositivo. Caso o modelo está na GPU e a mascara na CPU não dá erro
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

    def forward(self, x):

        # x possui 3 dimensões:
        # b            = quantidade de frases no batch
        # num_tokens   = quantidade de tokens em cada frase
        # d_in         = tamanho do vetor de cada token
        b, num_tokens, d_in = x.shape

        # Transformamos cada token em uma Key
        keys = self.W_key(x)
        # Transformamos cada token em uma Query
        queries = self.W_query(x)
        # Transformamos cada token em um Value
        values = self.W_value(x)

        # Comparamos cada Query com todas as Keys
        # Isso gera os "scores" de atenção quanto cada token deve prestar atenção nos outros
        # transpose(1, 2) troca as dimensões dos tokens e do vetor para que a multiplicação possa ser feita.
        attn_scores = queries @ keys.transpose(1, 2)

        # Aplicamos a máscara causal onde a máscara possui True, colocamos -infinito.
        # Depois, quando aplicarmos softmax, esses valores receberão peso 0
        # Assim, um token não consegue olhar para tokens futuros
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        # Transformamos os scores em pesos de atenção.
        # Dividimos pela raiz do tamanho das Keys para evitar que os valores fiquem muito grandes
        # softmax transforma os scores em valores entre 0 e 1 indicando quanto de atenção cada token recebe
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )

        # Aplicamos dropout nos pesos de atenção pra evitar overfitting
        attn_weights = self.dropout(attn_weights)

        # Agora usamos os pesos de atenção para combinar os Values
        # Pegue as informações dos outros tokens e dê mais importância para aqueles que receberam maior peso
        context_vec = attn_weights @ values

        # Retornamos o resultado final da atenção
        return context_vec