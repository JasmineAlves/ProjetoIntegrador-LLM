# self-attention com pesos treináveis para bons vetores de contexto
import torch
# Importando a classe da self-attention generalizada
from .self_attention_class import SelfAttention_v1
# Importando a classe da self-attention generalizada mais eficiente
from .self_attention_class import SelfAttention_v2

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your    (x¹)
        [0.55, 0.87, 0.66],  # journey (x²)
        [0.57, 0.85, 0.64],  # starts  (x³)
        [0.22, 0.58, 0.33],  # with    (x⁴)
        [0.77, 0.25, 0.10],  # one     (x⁵)
        [0.05, 0.80, 0.55]   # step    (x⁶)
    ]
)

# segundo elemento de entrada (2° palavra)
x_2 = inputs[1]
# o embedding de entrada tem dimensão 3
d_in = inputs.shape[1]
# o embedding de saída terá dimensão 2
# matrizes Wq Wk e Wv podem transformar um vetor de tamanho 3 em um vetor de tamanho 2 (multiplicação de matrizes)
d_out = 2 

# Inicia as matrizes de peso Wq, Wk e Wv.
# números aleatórios serem reproduzíveis, para números não mudarem a cada execução.
torch.manual_seed(123)
# matriz de (d_in, d_out) ou seja (3, 2) de números aleatórios entre 0 e 1.
# parameter diz: isso aqui é um peso que pode fazer parte de um modelo e ser aprendido durante o treinamento.
# requires_grad = False, durante o treinamento o computador deve aprender os valores de Wq Wk e Wv
# depois do treinamento os valores podem mudar até obter bons valores e para fazer isso precisa-se calcular os gradientes
# por isso normalmente seria True, mas colocamos false pois nesse momento não estamos treinando o modelo.
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# calcular a query, key e value vetores
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
# vetor de 2 dimensões para o vetor query
print(query_2)

# Todas as keys e values das outras palavras para comparação com a 2° palavra
# multiplicação de matrizes
keys = inputs @ W_key
values = inputs @ W_value
# 6 tonkes de entrada de 3 dimensões para 2 dimensões embeddings
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# attention score da palavra 2 comparada com a palavra 2
keys_2 = keys[1]
# compara a query da palavra 2 com a key da palavra 2, gerando um número
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# mesma comparação mas de uma só vez com todas as keys das outras palavras
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# 6 keys e cada key tem 2 números, d_k = 2
d_k = keys.shape[-1]
# divide os scores po raiz de 2, isso deixa os valores menores para evitar que os scores fiquem grandes,
# e ai aplica o softmax para normalizar os scores (somam 1)
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # dim=-1 faz softmax nas 6 posições da linha
# Esses são os attention weights
print(attn_weights_2)

# Score = valor bruto da comparação.
# Weight = importância normalizada de cada palavra.

# Calcular o vetor de contexto
# pesos de atenção servem como um fator que influencia e pesa a importância respectiva de cada vetor value
# Utiliza multiplicação de matrizes
context_vec_2 = attn_weights_2 @ values
# vetor de dimensão 2
print(context_vec_2)

# Usando a classe de self-attention com pesos treináveis para todas as palavras e não só a 2° palavra
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# Usando a classe de self-attention com pesos treináveis mais eficiente
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# As classes geram valores diferentes porque usam pesos iniciais diferentes para as matrizes de peso 
# (nn.Linear é mais sofisticado a inicialização)