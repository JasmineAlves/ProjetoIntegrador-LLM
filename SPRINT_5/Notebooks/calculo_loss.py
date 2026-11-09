import torch
from SPRINT_4.Notebooks.GPT_model import model
from SPRINT_5.Notebooks.text_idtoken import token_ids_to_text
from SPRINT_2.Notebooks.BPE import tokenizer


# Considera-se que os 2 exemplos de entrada já tiveram seus tokens IDs mapeados
inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
 [40, 1107, 588]]) # "I really like"]

# Os alvos tem os token IDs próximos
targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
 [1107, 588, 11311]]) # " really like chocolate"]

# Alimenta as entradas pro modelo para calcular os vetores logits, caa exemplo tem 3 tokens
with torch.no_grad(): # Sem gradiente pois não é treino ainda
 logits = model(inputs)
# Probabilidade de cada token no vocabulário
probas = torch.softmax(logits, dim=-1)
# [2, 3, 50257] dimensão do tensor de score de probabilidade
# [quant exemplos (linhas/tamanho do batch), num de tokens em cada linha, dimensão embedding (tamanho vocabulario)]
print(probas.shape)

# Aplica a função argmax na score de probabilidade para obter os token IDs correspondentes
# Pega a palavra com mais probabilidade e prepara ela pra virar texto
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# Gera 2 sets de saídas cada um com 3 token IDs preditados
print("Token IDs:\n", token_ids)

# Converte os tokens IDs de volta para texto (decodificar)
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
 f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# OBS: Os tokens de saida são bem diferentes dos tokens alvos porque o modelo não foi treinado ainda
# Para poder ensinar o modelo precisa medir exatamente o tamanho desse erro usando um cálculo chamado loss (perda) 
# É como dar uma "nota vermelha" que mostra o quão longe ela está do texto correto, diz ao modelo como ele deve ajustar 
# seus mecanismos internos (pesos) para melhorar e gerar um texto bom na próxima ve

# Mostra as probabilidades iniciais de softmax correspondendo aos tokens alvos
# Qual foi a chance (porcentagem) que o modelo deu para a palavra correta do gabarito, como não está treinada essa probabilidade é minuscula
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# Aplica o logaritmo para os scores de probabilidade, a média negativa de probabilidade log é a perda a ser calculada
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# OBS: O logaritmo transforma multiplicação em adição, em vez de multiplicar frações minúsculas o cálculo passa a ser uma soma de números
# evita que os números fiquem pequenos demais para o computador calcular transformando multiplicações frágeis em somas seguras

# Combina os log de probabilidade em um unico score calculando a média
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
# O objetivo é fazer com que esse valor chegue mais proximo possivel de 0 (atualizando os pesos do modelo)

# Cross entropy loss
# O log de probabilidade é negativo porque é simplesmente a média do log d eprobabilidade multiplicado por -1
neg_avg_log_probas = avg_log_probas * -1 # Valor "normal"
print(neg_avg_log_probas)

# Logits tem 3 dimensões [tamanho batch, num tokens, tamanho vocabulario]
print("Logits shape:", logits.shape)
# Targets tem 2 dimensões [tamanho batch, num tokens]
print("Targets shape:", targets.shape)

# Para o cross entropy loss tem que amassar os tensores combinando eles pela dimensão de batch
# Se antes tinha 2 batches com 3 palavras o flatten junta tudo em uma reta de 6 palavras para poder calcular a loss
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

# Isso faz tudo que foi feito acima, função do cross entropy loss
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

# Perplexidade
# O valor que der significa que o modelo está tão confuso que é como se ele estivesse tirando a sorte grande e 
# chutando cegamente entre 48.725 opções de palavras na hora de escolher a próxima palavra
perplexity = torch.exp(loss)