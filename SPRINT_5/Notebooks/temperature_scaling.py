import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Processo de geração do próximo token
vocab = {
 "closer": 0,
 "every": 1,
 "effort": 2,
 "forward": 3,
 "inches": 4,
 "moves": 5,
 "pizza": 6,
 "toward": 7,
 "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

# Assume-se que a LLM de um contexto incial "every effort moves you" e gerou os seguintes logits de next-token
next_token_logits = torch.tensor(
 [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# Anteriormente na generate_text_simples convertia-se as probabilidades via softmax e obtia o token ID correspondente via argmax
# podendo depois mapear o token de volta para texto via vocabulário inverso
# Como o maior logit é o de posição 3 (lembre-se que começa do 0) a palavra gerada seria "forward"
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

# Para aplicar o propcesso de amostragem probabilistica ao trocar a função argmax para multinomial
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])
# Isso ainda da como resultado "forward" porque ainda é a maism propensa e vai ser selecionada
# de qualquer jeito pela multinomial função na maioria das vezes (mas não todas)
# Para observar isso implementa-se a função que repete 1000x a amostra
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
          for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
      print(f"{freq} x {inverse_vocab[i]}")
# Mostra que forward é escolhida a maioria das vezes mas outras tbm são escolhidas algumas vezes
print_sampled_tokens(probas)

# TEMPERATURA SCALING
# Controla processo de distribuição e seleção, divide os logist por um num maior que 0
# Temperaturas maiores que 1 resultam e uma probabilidade de distribuição de token mais uniforme
# Temperaturas menores que 1, resultam e mais distribuições confidentes
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# Gráfico com a probabilidades originais ao lado das probabilidades escaladas com diferentes temperaturas
# [origiinal, mais baixo, mais alto]
temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
   rects = ax.bar(x + i * bar_width, scaled_probas[i],
                  bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

# OBS: Usar temperatura de 1 é o mesmo que  não usar nenhuma temperatura (x/1 = x)
# Os tokens são selecionados com a probabilidade igual da softmax original, por exemplo e temmperature=1
# então o token correspondente a "forward" seria selecionado 60% do tempo
# Temperatura de 0.1 os reusltado serão mais afiados em distribuição, enquanto temperatura de 5 resulta
# em mais distribuição uniformizada onde outros tokens são selecionados mais vezes
# Isso gera mais variedade em geraçaõ de texto mas tbm mais resultados em textos sem sentido