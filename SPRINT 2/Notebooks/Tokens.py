import re
# Mostra o total de caracteres existente no arquivo
# Objetivo: Tokenizar 20479 em palavras individuais para transformar em embeddings para a LLM
with open("Notebooks/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

# Lista do texto separado assim como espaços em branco, mas não separa caracteres especiais das palavras
# Sempre escreva a palavra da maneira crreta em questão de caixa alta para o código entender da maneira correta a estrutura
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)
# Separa os caracteres especiais
result = re.split(r'([,.]|\s)', text)
print(result)

# Remove os espaços em branco que foram para lista, não há necessidade deles.
result = [item for item in result if item.strip()]
print(result)

# Remover o espaço depende da aplicação que você quer, se remover o espaço de memória aumenta e reduz os custos computacionais
# Se deixar pode ser bom se estamos treinando modelos que são sensiveis a exata estrutura do texto (Python code)
# Para propósitos educacionais tiraremos o espaço

text = "Hello, world. Is this-- a test?" # TEXTO DE ENTRADA
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result) # TEXTO TOKENIZADO

# Aplicando a tokenização no texto inteiro
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) # Número de tokens no texto sem os espaços em branco
print(preprocessed[:30]) # Mostra os primeiros 30 tokens do texto, assim conseguimos ver se o tokenizer ta lidando bem com o texto

# Cria uma lista de todos os tokens únicos e arranja em ordem alfabética
# Assim determinamos o tamanho do vocabulário
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# Cria o vocabulário e mostra as primeiras 51 entradas
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
 print(item)
 if i >= 50:
  break

# Classe de tokenização
# Utiliza o encode método para dividir o texto em tokens e carregar o string-inteiro mapeamento
# para produzir token IDs via vocabulário
# Utiliza decode método para carregar o inverso inteiro-string mapeamento, convertendo IDs para texto novament.
class SimpleTokenizerV1:
    # Guarda o vocabulário para acessar o encode e decode depois
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} # Cria inverso que mapeia IDs para texto

    # Texto de entrada para tokens ID
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # Tokens ID de volta para texto
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove espaços antes da pontuação especificada
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# Instancia um novo objeto tokenizer
tokenizer = SimpleTokenizerV1(vocab)
# tokeniza uma passagem do texto
text = """"It's the last he painted, you know,"
 Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids) # Mostra o ID de cada token
print(tokenizer.decode(ids))

# Colocando um texto que não está no set de treinamento
# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))
# Vai dar erro, visto que a palavra Hello não estava no vocabulário, isso mostra
# porque precisamos considerar diveros e grandes sets de treinamentos para expandir o vocabulário quando trabalhando com LLMs

# Modificando o vocabulário para incluir o tokens especiais de separação de fontes de texto e palavras desconhecidas
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items())) # Novo tamanho do vocabulário

# Mostra o últimos 5 itens do vocabulário
for i, item in enumerate(list(vocab.items())[-5:]):
 # Sendo o último número o unk e o penultimo o endoftext
 print(item)

# Classe de tokenização que lida com palavras desconhecidas unk
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        # Troca palavras desconhecidas para unk tokens
        preprocessed = [item if item in self.str_to_int
        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Substitui espaços antes de pontuações
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# testando o tokenizer com duas frases independentes concatenadas com endoftext
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# Instanciando um novo objeto para tokenizar palavras desconhecidas
tokenizer = SimpleTokenizerV2(vocab)
# Mostra os IDs do tokens, se olharmos o tamanho final da lista de tokens IDs da para ver qual
# é o unk e qual é o endoftext
print(tokenizer.encode(text))

# Detokenização para ver onde está o unk e o endoftext
print(tokenizer.decode(tokenizer.encode(text)))

