# Amostragem de dados com janela deslizante
import torch
from torch.utils.data import Dataset, DataLoader
import BPE
import tiktoken

# Cria os pares de entradas-alvo ---------------------------------------------------------------------------------------------------------------
# Utiliza o conceito de janela deslizante

# Tokenizar todo o texto usando o BPE tokenizer
with open("Notebooks/the-verdict.txt", "r", encoding="utf-8") as f:
 raw_text = f.read()

enc_text = BPE.tokenizer.encode(raw_text)
print(len(enc_text)) # Número total de tokens no set de treinamento

# Remove os 50 primeiros tokens apenas por demonstração
enc_sample = enc_text[50:]

# Criar pares de entradas-alvo para a tarefe de predição da próxima palavra
# Variavel x contem os tokens de entrada e a y os alvos (entradas transferidas por 1)
context_size = 4 # Quantos tokens estão inclusos na entrada
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")

# Tarefa de prever a próxima palavra
for i in range(1, context_size+1):
 context = enc_sample[:i]
 desired = enc_sample[i]
 print(context, "---->", desired) # Toke ID alvo -----> entrada prevista que a LLM vai receber

# Converter os tokens IDs em texto
# Criamos os pares de entradas-alvo 
for i in range(1, context_size+1):
 context = enc_sample[:i]
 desired = enc_sample[i]
 print(BPE.tokenizer.decode(context), "---->", BPE.tokenizer.decode([desired])) # EX: and ---------> established

# Implementa um carregador de dados no dataset de entrada e retorna as entradas e alvos como tensores (arrays multidimensionais) ----------------------------

# Classe do dataset para entradas e alvos em lote (batch)
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) # Tokeniza o texto todo

        # Janela deslizante para dividir o livro em sequências sobreposta de comprimento máximo (max_length)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # Cada linha é um número de tokens IDs (comprimento max_length) colocado no tensor input_chunk
            self.input_ids.append(torch.tensor(input_chunk))
            # Esse tensor contém os alvos correspondentes
            self.target_ids.append(torch.tensor(target_chunk))

    # Retorna uma unica linha do dataset
    def __len__(self):
      return len(self.input_ids)

    # Retorna número total de linhas no dataset
    def __getitem__(self, idx):
      return self.input_ids[idx], self.target_ids[idx]

# Classe do dataloader que carrega as entradas em batches
def create_dataloader_v1(txt, batch_size=4, max_length=256,
        stride=128, shuffle=True, drop_last=True,
        num_workers=0):
        # Inicia o tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        # Cria o dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # Evita picos de perda durante treinamento
        drop_last=drop_last, # =True, tira o último batch se é mais curto que o batch_size especificado
        num_workers=num_workers # Número de processos do CPU usado para pré processamento
        )
        return dataloader

# Textar o dataloader com um batch de tamanho 1 em uma LLM com contexto de tamanho 4
with open("Notebooks/the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()
dataloader = create_dataloader_v1(
  raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
# Converte o dataloader em um iterador para pegar a próxima entrada através da função built-in next() do python
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch) # Mostra o tensor com tokens IDs de entrada e o tensor com os tokens IDs alvos
# Cada tensor contém 4 tokens IDs uma vez que o max_length=4

# Pegar outro batch do dataset
second_batch = next(data_iter)
print(second_batch)
# Pode-se observar que  segundo batch começa com o segundo Token ID do primeiro batch
# O stride=1 define o número de posições que a entrada passa através dos batches (janela deslizante)

# Dataloader com batch com tamanho maior que 1 de mudança (stride)
# Pode-se ver como stride = colunas, batch_size = linhas
dataloader = create_dataloader_v1(
 raw_text, batch_size=8, max_length=4, stride=4,
 shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)