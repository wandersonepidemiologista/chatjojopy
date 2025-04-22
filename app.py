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

# 🌐 Configuração da interface com tema personalizado
st.set_page_config(page_title="ChatJoJoPy", layout="wide")

# Estilo customizado com as cores fornecidas
st.markdown("""
    <style>
    body { background-color: #f8f6ca; color: #025e73; }
    .stApp { background-color: #f8f6ca; }
    .stTextInput > div > div > input { color: #025e73 !important; }
    .stChatMessage.user, .stChatMessage.assistant {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        color: #03658c;
    }
    .stButton button {
        background-color: #02735e !important;
        color: white !important;
    }
    .stButton button:hover {
        background-color: #03658c !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar com a logo e descrição
with st.sidebar:
    st.image("imagens/logo.png", width=200)
    st.title("**Chat JoJoPy**")
    st.subheader("Comunidade de mãos dadas na resposta às emergências em saúde pública")
    st.markdown("""
    **O termo \"JOJOPY\" tem origem no Glossário Indígena Guarani-Português e significa \"deram as mãos\". 
    No contexto guarani, essa expressão vai muito além de um gesto físico, simbolizando união e comunidade, representando a força dos laços coletivos e a construção de uma sociedade solidária. 
    Essa simbologia traduz os objetivos esperados para os participantes do curso: promover a cooperação e o fortalecimento da coletividade.
    """)

# 🧠 Inicializar modelo Hugging Face com LangChain
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    task="text2text-generation"
)

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

# 🔍 Interface com abas
abas = st.tabs(["🧠 Chat", "📄 Documentos", "⚙️ Sobre"])

with abas[0]:
    st.title("🤖 ChatJoJoPy — Emergências em Saúde Pública")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        st.chat_message(role).write(content)

    if prompt := st.chat_input("Digite sua pergunta sobre emergências em saúde pública..."):
        st.chat_message("user").write(prompt)
        try:
            resposta = rag.run(prompt)
        except Exception as e:
            resposta = f"Erro ao gerar resposta: {e}"
        st.chat_message("assistant").write(resposta)
        st.session_state.chat_history.append(("user", prompt))
        st.session_state.chat_history.append(("assistant", resposta))

with abas[1]:
    st.subheader("📄 Documentos carregados")
    st.markdown("Os seguintes arquivos PDF foram incluídos no modelo:")
    for arquivo in os.listdir("documentos"):
        if arquivo.endswith(".pdf"):
            st.markdown(f"- {arquivo}")

with abas[2]:
    st.subheader("⚙️ Sobre o projeto")
    st.markdown("""
    - **Nome:** ChatJoJoPy
    - **Objetivo:** Apoiar a consulta rápida a documentos técnicos sobre emergências em saúde pública.
    - **Desenvolvedor:** Wanderson Oliveira
    - **Código-fonte:** [GitHub](https://github.com/wandersonepidemiologista/chatjojopy)
    """)