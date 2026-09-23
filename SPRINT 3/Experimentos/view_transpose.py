import torch

# Cria um tensor com 1 exemplo, 2 cabeças, 3 tokens e 4 números por token
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

# Troca as duas últimas dimensões (num_tokens, head_dim) → (head_dim, num_tokens)
# Isso permite fazer a multiplicação entre as matrizes
print(a @ a.transpose(2, 3))

# Pega somente a primeira cabeça
# 0 = primeiro exemplo
# 0 = primeira cabeça
# : = todos os tokens
# : = todos os valores dos tokens
first_head = a[0, 0, :, :]
# Multiplica a primeira cabeça pela sua transposta, isso compara cada token com todos os outros tokens
first_res = first_head @ first_head.T
print("First head:\n", first_res)

# Pega somente a segunda cabeça
second_head = a[0, 1, :, :]
# Faz a mesma comparação para a segunda cabeça
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)