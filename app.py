import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import random

# ⚖️ Carregar variáveis de ambiente
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Sidebar para token opcional
with st.sidebar:
    huggingfacehub_api_key = st.text_input("Hugging Face API Token", type="password")
    st.markdown("[Crie sua chave gratuita](https://huggingface.co/settings/tokens)")
    st.markdown("[Código fonte no GitHub](https://github.com/wandersonepidemiologista/chatjojopy)")

if not huggingfacehub_api_key:
    st.info("Por favor, insira sua chave da Hugging Face na barra lateral.")
    st.stop()

# 🧠 Inicializar modelo Hugging Face com LangChain
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=huggingfacehub_api_key,
    task="text2text-generation"
)

# 🌐 Configuração da interface
st.set_page_config(page_title="ChatJoJoPy", layout="wide")
st.title("🤖 ChatJoJoPy — Emergências em Saúde Pública")

# 📂 Carregar e indexar documentos
@st.cache_resource
def carregar_base():
    documentos = []
    pasta = "documentos"
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta, arquivo)
            loader = PyPDFLoader(caminho)
            documentos.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

try:
    db = carregar_base()
    retriever = db.as_retriever()
    rag = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
except Exception as e:
    st.error(f"Erro ao carregar documentos: {e}")
    st.stop()

# Inicializa histórico de mensagens
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Exibe mensagens anteriores
for i, (role, content) in enumerate(st.session_state.chat_history):
    st.chat_message(role).write(content)

# Input interativo
if prompt := st.chat_input("Digite sua pergunta sobre emergências em saúde pública..."):
    st.chat_message("user").write(prompt)
    try:
        resposta = rag.run(prompt)
    except Exception as e:
        resposta = f"Erro ao gerar resposta: {e}"

    st.chat_message("assistant").write(resposta)
    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("assistant", resposta))