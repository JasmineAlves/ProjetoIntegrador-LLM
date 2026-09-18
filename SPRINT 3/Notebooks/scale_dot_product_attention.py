# self-attention com pesos treináveis para bons vetores de contexto
import torch

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