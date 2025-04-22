import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# import random # Removido pois não estava sendo utilizado

# --- Configurações Iniciais ---

# ⚖️ Carregar variáveis de ambiente (Requer arquivo .env com HUGGINGFACEHUB_API_TOKEN)
# Certifique-se de que o arquivo .env existe e contém sua API key.
load_dotenv()

# 🌐 Configuração da interface com tema personalizado
st.set_page_config(page_title="ChatJoJoPy", layout="wide")

# Estilo customizado com as cores fornecidas (CORRIGIDO)
st.markdown("""
    <style>
    body { background-color: #ffffff; color: #025e73; }
    .stApp { background-color: #ffffff; }
    .stTextInput > div > div > input { color: #025e73 !important; }
    /* CORREÇÃO: Cor do texto ajustada para contraste com o fundo branco */
    .stChatMessage.user, .stChatMessage.assistant {
        background-color: #ffffff;
        border: 1px solid #e0e0e0; /* Adicionada borda sutil para melhor visualização */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        color: #025e73; /* Cor do texto corrigida */
    }
    .stButton button {
        background-color: #02735e !important;
        color: white !important;
    }
    .stButton button:hover {
        background-color: #015c4b !important; /* Escurecer um pouco no hover */
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
# Certifique-se que a pasta 'imagens' e o arquivo 'logo.png' existem
with st.sidebar:
    try:
        st.image("imagens/logo.png", width=200)
    except FileNotFoundError:
        st.error("Arquivo 'imagens/logo.png' não encontrado.")
    st.title("**Chat JoJoPy**")
    st.subheader("Comunidade de mãos dadas na resposta às emergências em saúde pública")
    st.markdown("""
    O termo **\"JOJOPY\"** tem origem no Glossário Indígena Guarani-Português e significa **\"deram as mãos\"**.
    No contexto guarani, essa expressão vai muito além de um gesto físico, simbolizando união e comunidade, representando a força dos laços coletivos e a construção de uma sociedade solidária.
    Essa simbologia traduz os objetivos esperados para os participantes do curso: **promover a cooperação e o fortalecimento da coletividade**.
    """)

# --- Configuração do Modelo e RAG ---

# 🧠 Inicializar modelo Hugging Face com LangChain
# Nota: 'google/flan-t5-small' é um modelo text-to-text versátil e gratuito.
try:
    llm = HuggingFaceEndpoint(
        repo_id="facebook/bart-large-mnli",
        task="text2text-generation",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.5,
        max_new_tokens=512
    )
except Exception as e:
    st.error(f"Erro ao inicializar o modelo LLM (HuggingFaceEndpoint) com BART: {e}")
    st.error("Verifique sua conexão e a chave HUGGINGFACEHUB_API_TOKEN no arquivo .env.")
    st.stop()

# 📂 Carregar e indexar documentos
# Certifique-se que a pasta 'documentos' existe e contém arquivos PDF.
@st.cache_resource # Usar cache para evitar recarregar/reindexar a cada interação
def carregar_e_indexar_base():
    """Carrega PDFs da pasta 'documentos', divide em chunks e cria um índice FAISS."""
    documentos = []
    pasta_documentos = "documentos"
    if not os.path.isdir(pasta_documentos):
        st.error(f"A pasta '{pasta_documentos}' não foi encontrada.")
        return None # Retorna None se a pasta não existe

    arquivos_pdf = [f for f in os.listdir(pasta_documentos) if f.lower().endswith(".pdf")]
    if not arquivos_pdf:
        st.warning(f"Nenhum arquivo PDF encontrado na pasta '{pasta_documentos}'. O chat funcionará sem base de conhecimento específica.")
        return None # Retorna None se não há PDFs

    for arquivo in arquivos_pdf:
        caminho = os.path.join(pasta_documentos, arquivo)
        try:
            loader = PyPDFLoader(caminho)
            documentos.extend(loader.load())
        except Exception as e:
            st.warning(f"Erro ao carregar o arquivo {arquivo}: {e}")
            continue # Pula para o próximo arquivo em caso de erro

    if not documentos:
        st.error("Nenhum documento PDF pôde ser carregado com sucesso.")
        return None

    try:
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Ajuste overlap se necessário
        chunks = splitter.split_documents(documentos)

        # Usar embeddings do HuggingFace (pode baixar modelo na primeira execução)
        embeddings = HuggingFaceEmbeddings()

        # Criar o índice FAISS
        db = FAISS.from_documents(chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"Erro ao processar documentos e criar índice: {e}")
        return None

# Tentar carregar a base e configurar o RAG
db = carregar_e_indexar_base()
rag = None
if db:
    try:
        from langchain.prompts import PromptTemplate # Importe aqui dentro do bloco

        retriever = db.as_retriever(search_kwargs={'k': 3})
        # Criar um template de prompt customizado
        prompt_template = """Responda à seguinte pergunta com base no contexto fornecido:
        {context}
        Pergunta: {question}
        Resposta:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Configurar a chain RetrievalQA com o tipo 'stuff' e o prompt customizado
        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Erro ao configurar a cadeia RAG: {e}")
else:
    st.warning("Base de documentos não carregada. O chat pode não ter informações específicas dos arquivos.")

# --- Interface Principal com Abas ---
abas = st.tabs(["🧠 Chat", "📄 Documentos", "⚙️ Sobre"])

# Aba de Chat
with abas[0]:
    st.title("🤖 ChatJoJoPy — Emergências em Saúde Pública")
    st.markdown("Faça sua pergunta sobre os documentos carregados ou sobre emergências em saúde pública.")

    st.markdown("---")
    st.markdown("### 💡 Exemplos de perguntas:")
    perguntas_exemplo = [
        "O que é uma emergência em saúde pública?",
        "Quais são os planos nacionais existentes?",
        "Qual a diferença entre preparação e resposta?",
        "O que é um plano de contingência?",
        "Quais documentos orientam os municípios em desastres?",
        "O que são as fases da resposta a emergências?",
        "Como a vigilância atua em situações de emergência?",
        "Quais são os indicadores de prontidão e resposta?"
    ]
    cols = st.columns(2)
    for i, pergunta in enumerate(perguntas_exemplo):
        cols[i % 2].markdown(f"- {pergunta}")
    st.markdown("---")


    # Inicializar histórico do chat na session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Mostrar histórico do chat
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)

    # Input do usuário
    if prompt := st.chat_input("Digite sua pergunta aqui..."):
        # Adicionar mensagem do usuário ao histórico e mostrar na tela
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)

        # Gerar resposta do assistente se o LLM foi inicializado
        if llm:
            try:
                with st.spinner("Pensando..."):
                    # Chamada direta ao LLM
                    resposta = llm.invoke(prompt)

                # Adicionar resposta do assistente ao histórico e mostrar na tela
                st.session_state.chat_history.append(("assistant", resposta))
                with st.chat_message("assistant"):
                    st.write(resposta)

            except Exception as e:
                resposta = f"Erro ao gerar resposta diretamente com o LLM: {e}. Verifique a configuração e a API Key."
                print(f"Erro na execução do LLM direto: {e}")
        else:
            resposta = "O modelo de linguagem não foi inicializado corretamente."
            st.session_state.chat_history.append(("assistant", resposta))
            with st.chat_message("assistant"):
                st.write(resposta)

# Aba de Documentos
with abas[1]:
    st.subheader("📄 Documentos Carregados")
    st.markdown("Os seguintes arquivos PDF foram encontrados na pasta `documentos` e podem ser baixados:")

    pasta_documentos = "documentos"
    try:
        if os.path.isdir(pasta_documentos):
            arquivos_pdf = [f for f in os.listdir(pasta_documentos) if f.lower().endswith(".pdf")]
            if arquivos_pdf:
                for arquivo in arquivos_pdf:
                    caminho = os.path.join(pasta_documentos, arquivo)
                    try:
                        with open(caminho, "rb") as f:
                            bytes_pdf = f.read()
                            st.download_button(
                                label=f"🔗 Baixar: {arquivo}",
                                data=bytes_pdf,
                                file_name=arquivo,
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.error(f"Não foi possível ler o arquivo {arquivo} para download: {e}")
            else:
                st.info("Nenhum arquivo PDF encontrado na pasta 'documentos'.")
        else:
             st.warning(f"A pasta '{pasta_documentos}' não foi encontrada.")

    except Exception as e:
        st.error(f"Erro ao listar documentos: {e}")

# Aba Sobre
with abas[2]:
    st.subheader("⚙️ Sobre o Projeto")
    st.markdown("""
    - **Nome:** Chat JoJoPy
    - **Objetivo:** Apoiar a consulta rápida a documentos técnicos sobre emergências em saúde pública utilizando RAG (Retrieval-Augmented Generation).
    - **Tecnologias:** Streamlit, LangChain, Hugging Face Transformers.
    - **Desenvolvedor:** Wanderson Oliveira - Epidemiologista ([Sobre mim](https://www.wandersonepidemiologista.com/sobre))
    - **Saiba mais sobre o município fictício de Jojopy:** [E-Book Jojopy](https://epidemiologista.quarto.pub/jojopy/)
    """)
    st.markdown("---")
    st.markdown("*(Versão do código revisada e corrigida.)*")