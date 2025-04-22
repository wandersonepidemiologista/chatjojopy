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

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configura o modelo do Hugging Face
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5}, huggingfacehub_api_token=hf_token)

st.set_page_config(page_title="ChatJoJoPy com HF", layout="wide")
st.title("🤖 ChatJoJoPy — Emergências em Saúde Pública")

@st.cache_resource
def carregar_base():
    loader = PyPDFLoader("documentos/plano_municipal.pdf")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

db = carregar_base()
retriever = db.as_retriever()
rag = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

pergunta = st.text_input("💬 Sua pergunta:")
if pergunta:
    resposta = rag.run(pergunta)
    st.markdown("### 📌 Resposta:")
    st.write(resposta)
