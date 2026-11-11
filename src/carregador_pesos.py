import torch
import numpy as np

# Como antes o modelo inicializava com pesos aleatórios para pre treinamento
# É necessario sobreescrever esses pesos com os pesos carregados do dicionário params
# A função assign() serve para pegar os pesos de um modelo GPT que já foi treinado 
# verificar se eles possuem o mesmo formato dos pesos do nosso modelo e transformá-los em parâmetros que o PyTorch consegue usar
def assign(left, right):
    if left.shape != right.shape:
       raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         "Right: {right.shape}"
        ) 
    return torch.nn.Parameter(torch.tensor(right))

# Coloca os pesos  do token embedding e posicionais especificados nos params
# "Match" os pesos do OpenAI com o nosso GPTModel
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(
        gpt.pos_emb.weight, params['wpe']
    )
    gpt.tok_emb.weight = assign(
        gpt.tok_emb.weight, params['wte']
    )
    # A função np.split divide a atenção e os pesos bias em 3 partes iguais para query, key e value
    for b in range(len(params["blocks"])):
        # Itera em cada bloco transformer no modelo
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"],
            3,
            axis=-1
        )

        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"],
            3,
            axis=-1
        )

        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )

        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )

        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )

    # Weight tying, Compartilha os mesmos pesos em duas partes do modelo, em vez de criar pesos diferentes para cada uma
    # Utiliza os mesmos pesos da camada de entrada (que transforma as palavras em números) na camada de saída (que escolhe a próxima palavra)
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])