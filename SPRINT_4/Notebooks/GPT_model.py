import torch
import torch.nn as nn
from transformer_block_class import TransformerBlock
from normalization_class import LayerNorm
from cfg import GPT_CONFIG_124M
import data_flow
import sys
from pathlib import Path

# Localiza a pasta raiz que contém SPRINT 2 e SPRINT 4
raiz = Path(__file__).resolve().parents[2]
# Adiciona a pasta onde está o BPE
sys.path.insert(
    0,
    str(raiz / "SPRINT_2" / "Notebooks")
)
import BPE

# Implementação da arquitetura do modelo GPT
class GPTModel(nn.Module):
    # Inicializa o token e camadas de embedding posicionais usando o dicionario definido
    # Essas camadas de embeddings convertem os IDs do token de entrada em vetores densos e adicionar informação posicional
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
          *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # Cria a camada final que transforma os vetores em pontuações para todos os tokens possíveis do vocabulário
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    # Define como os dados percorrem o modelo
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
          torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(data_flow.batch)
# Tokens ID do text1 e text2
print("Input batch:\n", data_flow.batch)
# shape [2,4,50257] -> [quant texto de entrada, quant tokens por texto, tamanho do vocabulario]
print("\nOutput shape:", out.shape)
print(out)

# Número total de parametros no tensor do modelo
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

total_params_gpt2 = (
 total_params - sum(p.numel()
 for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
 f"considering weight tying: {total_params_gpt2:,}"
)

# GERAÇÃO DE TEXTO -----------------------------------------------------------------------------------------------------------
# Recebe o modelo, o texto inicial em tokens, quantos tokens novos gerar e tamanho max do contexto
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # Repete processo até gerar a quantidade de tokens solicitada
    for _ in range(max_new_tokens):
        # Seleciona os ultimos tokens de entrada para respeitar limite
        idx_cond = idx[:, -context_size:]
        # Desativa calculo de gradiente (modelo não está sendo mais treinado e sim gerando texto)
        with torch.no_grad():
         # Passa os tokens pelo modelo e obtem pontuações dos possíveis prox tokens
         logits = model(idx_cond)
        # Seleciona as pontuações que correspondem a ultima posição da sequencia
        logits = logits[:, -1, :]
        # Transforma as pontuações em probabilidades (maior prob mairo preferencia)
        probas = torch.softmax(logits, dim=-1)
        # Escolhe token com mais prob e guarda seu ID
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # Adiciona novo token ao final da sequencia original
        idx = torch.cat((idx, idx_next), dim=1)
        # Retorna sequencia completa de tokens
    return idx

# Codificar o contexto de entrada em token IDs
start_context = "Hello, I am"
encoded = BPE.tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

# eval() desabilita componentes que são usado apenas no treinamento
model.eval()
# Codificar o tensor de entrada
out = generate_text_simple(
 model=model,
 idx=encoded_tensor,
 max_new_tokens=6,
 context_size=GPT_CONFIG_124M["context_length"]
)
# Saída é os token IDs
print("Output:", out)
print("Output length:", len(out[0]))

# Decodificar token IDs para texto
decoded_text = BPE.tokenizer.decode(out.squeeze(0).tolist())
# Saí palavras sem sentido pois ainda não treinamos o modelo
print(decoded_text)

# OBS: Se LLM suporta só 5 tokens e o tamamanho do contexto é 10 então só os últimos 5 tokens são usados como contexto