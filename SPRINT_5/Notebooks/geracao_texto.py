import torch
from SPRINT_5.Notebooks.text_idtoken import text_to_token_ids
from SPRINT_5.Notebooks.text_idtoken import token_ids_to_text
from SPRINT_2.Notebooks.BPE import tokenizer
from SPRINT_4.Notebooks.GPT_model import model
from SPRINT_5.Notebooks.inicializacao_modelo import GPT_CONFIG_124M
# Gerador de texto modificado com mais diversidade
# idx é o texto inicial convertido em tokens, temperature controla nivel de aleatoriedade na escolha do proximo token
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    # Executa o processo de geração de tokens (gera max_new_tokens tokens)
    for _ in range(max_new_tokens):
        # Seleciona os ultimos tokens do texto gerado respeitando o context_size
        idx_cond = idx[:, -context_size:]
        # Desliga o calculo de gradientes durante geração de texto
        with torch.no_grad():
             # Texto convertido em tokens é enviado ao modelo GPT que calcula os logits 
             # maior logit maior probabilidade daquele token ser o proximo
             logits = model(idx_cond)
        # Seleciona os logits correpsondentes a ultima posição do texto
        logits = logits[:, -1, :]
        # Aplica Top-K se foi definido um valor para ele inicialmente
        if top_k is not None:
            # Sleciona os topk maiores valores de logits
            top_logits, _ = torch.topk(logits, top_k)
            # Encontra o menor valor entre os logits selecionados (limite)
            min_val = top_logits[:, -1]
            # Filtragem, qualquer valor abaixo do limite vira -inf
            logits = torch.where(
                   logits < min_val,
                   torch.tensor(float('-inf')).to(logits.device),
                   logits
            )
        # Se temperatura maior que 0
        if temperature > 0.0:
            # Divide os logist pelo valor da temperatura
            logits = logits / temperature
            # Transforma os logits em probabilidades via softmax (somam 1 no conjunto de opçoes)
            probs = torch.softmax(logits, dim=-1)
            # Escolhe aleatoriamente um proximo token com base nas probabilidades 
            # (não necessariiamente o token com maior prob vai ser escolhido toda vez)
            idx_next = torch.multinomial(probs, num_samples=1)
        # Se temperatura não é maior que 0
        else:
            # Utiliza argmax para escolher diretamente o token que tem maior logit (deixa a geração mais previsivel)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # Verifica se o texto chegou ao fim
        if idx_next == eos_id:
            break
        # Adiciona o novo token ao final do texto
        idx = torch.cat((idx, idx_next), dim=1)
    # Retorna texto gerado
    return idx

# OBS: essa comparação funciona diretamente para o caso de geração com apenas uma sequência
# Em geração com vários textos simultaneamente seria necessário tratar cada sequência separadamente

# Testando a função
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))