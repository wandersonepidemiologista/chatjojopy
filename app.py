import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import random

# 🔐 Carrega token Hugging Face
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 🤖 Configura modelo de linguagem
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=hf_token,
    task="text2text-generation"
)

# 🌐 Configuração da interface
st.set_page_config(page_title="ChatJoJoPy com HF", layout="wide")
st.title("🤖 ChatJoJoPy — Emergências em Saúde Pública")

# 📂 Carrega os documentos
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

db = carregar_base()
retriever = db.as_retriever()
rag = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 💡 Perguntas quebra-gelo
perguntas_exemplo = [
    "O que é uma emergência em saúde pública?",
    "Quais são os critérios para ativar o plano de resposta?",
    "Quem deve ser acionado primeiro em caso de surto?",
    "Quais fases compõem um plano de contingência?",
    "Como a APS atua em situações de desastre?",
    "Quais os principais indicadores de prontidão municipal?",
    "Como se organiza o fluxo de resposta em campo?",
    "O que diferencia uma emergência local de uma nacional?",
    "Como integrar a atenção primária em emergências?",
    "Quais documentos norteiam a resposta a desastres químicos?"
]

sugestao = random.choice(perguntas_exemplo)
st.markdown("### 💡 **Exemplo de pergunta** (quebra-gelo):")
st.info(f"💬 {sugestao}")

if "pergunta" not in st.session_state:
    st.session_state.pergunta = ""

if st.button("👈 Usar essa pergunta"):
    st.session_state.pergunta = sugestao

# 🧠 Campo de entrada com sugestão
pergunta = st.text_input("Digite sua pergunta:", value=st.session_state.pergunta, key="pergunta")

# 📌 Resultado
if pergunta:
    resposta = rag.run(pergunta)
    st.markdown("### 📌 Resposta:")
    st.write(resposta)
