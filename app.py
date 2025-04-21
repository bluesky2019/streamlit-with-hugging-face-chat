import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configuração da página
st.set_page_config(page_title="Chatbot Hugging Face", page_icon="🤖")

# Título da aplicação
st.title("🤖 Chatbot Simples com Hugging Face")
st.write("Este é um chatbot simples usando o modelo DialoGPT-medium da Microsoft.")

# Inicialização do modelo
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Histórico de conversa
if "history" not in st.session_state:
    st.session_state.history = []

# Função para gerar resposta
def generate_response(user_input):
    # Codifica a entrada do usuário e adiciona o histórico
    chat_history_ids = torch.tensor([])
    for input_text, response_text in st.session_state.history[-4:]:  # Mantém apenas as últimas 4 interações
        new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.nelement() != 0 else new_user_input_ids
        
        bot_response_ids = tokenizer.encode(response_text + tokenizer.eos_token, return_tensors='pt')
        chat_history_ids = torch.cat([chat_history_ids, bot_response_ids], dim=-1)
    
    # Codifica a nova entrada do usuário
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.nelement() != 0 else new_user_input_ids
    
    # Gera resposta
    bot_response_ids = model.generate(
        input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decodifica a resposta
    bot_response = tokenizer.decode(bot_response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_response

# Interface de chat
user_input = st.chat_input("Digite sua mensagem em inglês aqui...")

if user_input:
    # Adiciona mensagem do usuário ao histórico e exibe
    st.session_state.history.append((user_input, ""))
    
    # Gera resposta do bot
    with st.spinner("Pensando..."):
        bot_response = generate_response(user_input)
    
    # Atualiza o histórico com a resposta do bot
    st.session_state.history[-1] = (user_input, bot_response)

# Exibe o histórico de conversa
for user_msg, bot_msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(bot_msg)

# Botão para limpar o histórico
if st.button("Limpar Conversa"):
    st.session_state.history = []
    st.experimental_rerun()