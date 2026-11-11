import torch
from SPRINT_5.Notebooks.temperature_scaling import next_token_logits

# Restringir o tokens de amostra para o top-k mais prováveis tokens e 
# excluir todos os outros tokens do processo de seleção ao mascarar seus scores de probabilidade
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)
# Trocando todos os não selecionados logits com valores infinitos engativos (-inf)
# assim quandoo calcular softmax os scores de probabilidade são 0, e os que sobram somam até 1

# Utiliza a função where para colocar os valores dos tokens que estão abaixo do valor minimo
# do topk para infinito negativo
new_logits = torch.where(
    # Identifica logits menores que o minimo no topk
    condition=next_token_logits < top_logits[-1],
    # Atribui -ing para eles
    input=torch.tensor(float('-inf')),
    # Deixa os logits normais para os outros tokens
    other=next_token_logits
)
print(new_logits)

# Aplica spftmax para transformar isso em probabilidades de próximo token
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)
# Gera 3 não-zerados scores de probabilidade

# Depois pode-se aplicar o temperature scaling e multinominal para amostragem probabilistica
# para selecionar o próximo token entre esses 3 scores de probabilidade não zerados