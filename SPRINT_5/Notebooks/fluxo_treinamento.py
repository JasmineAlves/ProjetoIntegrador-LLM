import torch
from SPRINT_5.Notebooks.pretreinamento import calc_loss_batch
from SPRINT_5.Notebooks.pretreinamento import calc_loss_loader
from SPRINT_4.Notebooks.GPT_model import generate_text_simple
from SPRINT_5.Notebooks.text_idtoken import text_to_token_ids
from SPRINT_5.Notebooks.text_idtoken import token_ids_to_text
from SPRINT_2.Notebooks.BPE import tokenizer
from SPRINT_4.Notebooks.GPT_model import model
from SPRINT_5.Notebooks.inicializacao_modelo import GPTModel, GPT_CONFIG_124M
from SPRINT_5.Notebooks.pretreinamento import val_loader, train_loader, device

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Treinamento do modelo
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Cria lista vazias para guardar como o modelo se comporta ao longo do tempo
    # Guarda os error do treinamento, erros da validação e quantos tokens o modelo processou
    train_losses, val_losses, track_tokens_seen = [], [], []
    # Para contar quantos tokens foram processados e quantos passos de treinamento já foram executados
    tokens_seen, global_step = 0, -1 # -1 porque no 1° batch sera somado 1

    # Cada epoch sinaliza que o modelo leu o texto inteiro uma vez 
    for epoch in range(num_epochs):
        # Modelo no modo aprendizado
        model.train()
        # Pega os batches de texto de treinamento (perguntas e gabaritos) para serem lidos
        for input_batch, target_batch in train_loader:
            # Limpa gradientes acumulados anteriormente para não somar do batch anterior pro atual
            optimizer.zero_grad()
            # Modelo forma um texto e o sistema calcula o tamanho da loss
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # Modelo olha para o erro e descobre de trás para frente onde foi falhado (como cada parametro contribui para o erro)
            # Assim o sistema descobre em qual direção os parametros precisam ser ajustados
            loss.backward()
            # Otimizador ajusta os pesos para corrigir o erro (utiliza os gradientes calculados antes para atualizar os parâmetros do modelo)
            # UTILIZA O AdamW
            optimizer.step()
            # Conta quantas palavras o modelo já leu e quantos passos ele já passou
            tokens_seen += input_batch.numel()
            global_step += 1

           # Passo de avaliação opcional
           # Define a frequencia de avaliação (nesse caso a avaliação acontece quando o contador de passos é divisivel por 5)
           # Avaliações acontecem em 0,5,10,15,20...
            if global_step % eval_freq == 0:
                # Calcula o erro utilizando os dados de treinamento e o erro utilizando os dados de validação
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # Guarda erros nas listas criadas anteriormente
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                # Guarda quantidade de tokens processados no momento
                track_tokens_seen.append(tokens_seen)
                # Mostra os resultados (época, passo, erro trainamento e erro validação)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                ) 
        
        
        # Gera o texto de exemplo no final de cada época
        # Modelo tenta continuar a frase utilizando o que aprendeu até o momento, ajuda observar se o treinamento está melhorando
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
        
    # Devolve as 3 listas completas
    return train_losses, val_losses, track_tokens_seen

# Função que avalia o aprendizado
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # Modelo em modo avaliação (dropout desligado)
    model.eval() 
    # Calculo de gradientes desligado
    with torch.no_grad():
        # Calcula erro do modelo com os dados de treinamento
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        # Calcula erro do modelo com os dados de validação
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    # Devolve o modelo em modo treinamento (pois ele continua mesmo depois da avaliação)
    model.train()

    return train_loss, val_loss
# OBS: Se a Loss de treinamento diminui mas a Loss de validação começa a aumentar isso pode indicar overfitting

# Faz o modelo "escrever"
def generate_and_print_sample(model, tokenizer, device, start_context):
    # Modelo em modo avaliação (sem dropout e calculo de gradientes)
    # Não queremos atualizar os pesos enquanto estamos apenas gerando texto
    model.eval()
    # Descobre o tamanho max do contexto definido pela matriz de embeddings posicionais do modelo, ou seja,
    # informa quantas posições de tokens o modelo consegue considerar dentro do seu contexto
    context_size = model.pos_emb.weight.shape[0]
    # Transforma texto inicial em tokens numéricos
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    # Desativa calculo de gradientes
    with torch.no_grad():
        # Geração de texto
        # Modelo recebe texto inicial e começa a prever o próximo token e utiliza o token gerado para prever o seguinte
        # Processo se repete 50 até gerar 50 novos tokens ou atingir outra condição definida pela função
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    # Decodificação, transforma os IDs dos tokens gerados em texto
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # Tira pular linha para que o texto fique na mesma linha
    print(decoded_text.replace("\n", " "))
    # Devolve o modelo ao modo treinamento
    model.train()

# Prepara o modelo para começar a aprender
# Torna resultados reproduziveis
# Comentado para por realizar experimentos com diferentes números de epochs
"""torch.manual_seed(123)
# Cria modelo GPT com as configurações selecionadas
model = GPTModel(GPT_CONFIG_124M)
# Modelo fica do dispositivo correto (CPU/GPU)
model.to(device)
# Configura o otimizador (AdamW), ajuda a ajustar os pesos do modelo durante o treinamento
optimizer = torch.optim.AdamW(
    # Acessa parametros treinaveis do modelo
    model.parameters(),
    # Taxa de aprendizado, mecanismo de regularização que penaliza pesos muito grandes (controla complexidade do modelo e reduz overfitting)
    lr=0.0004, weight_decay=0.1
)
# Percorre os dados de treinamento 10x
num_epochs = 10
# Inicia o treinamento
# train_model_simple (modelo GPT, dados treinamento, dados validação, AdamW, CPU/GPU, 10, 5, 5, texto inicial, BPE)
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)"""

# OBS: Taxa muito alta pode fazer o treinamento ficar instável. Taxa muito baixa pode deixar o aprendizado muito lento
# OBS: Loss do treinamento melhor muito (de 9.781 para 0.391) conseguindo gerar no final do treinamento texto gramaticamente correto
# Porém na loss d evalidação nunca fica menor que nem a d etreinamento, ficando entre 6.452 após as 10 epochs

# Gráfico simples para mostrar as losses de treinamento e validação
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

# Coloque este IF antes do código que inicia o treino
if __name__ == "__main__":
    # Prepara o modelo para começar a aprender
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# OBS: É possivel observar no gráfico que tanto as losses de treinamento quanto de validação começam
# a melhorar da epoch 1, mas começam a divergir na epoch 2, loss de validação é muito maior que a de treinamento
# isso indica que o modelo está overfitting os dados de treinamento (modelo memorizou o vocabulario)
# Mas isso é normal de acontecer porque estamos trabalhando com um dataset de treinamento MUITO pequeno e treinando com multiplas epochs,
# o comum é treinar com um dataset grande e 1 epoch só