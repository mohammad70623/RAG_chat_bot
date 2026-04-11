import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedBot — AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        border: 1px solid #2b5278;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; color: #e8f4fd; margin: 0; }
    .main-header p  { color: #90b8d4; font-size: 1rem; margin: 0.5rem 0 0; }
    .badge {
        display: inline-block;
        background: #1a4a7a; color: #63b3ed;
        font-size: 0.72rem; font-weight: 600;
        padding: 3px 12px; border-radius: 20px;
        border: 1px solid #2b6cb0;
        margin-top: 0.8rem; letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .chat-label-user {
        text-align: right; font-size: 0.72rem;
        color: #63b3ed; font-weight: 600;
        margin-bottom: 4px; letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .chat-label-bot {
        font-size: 0.72rem; color: #68d391;
        font-weight: 600; margin-bottom: 4px;
        letter-spacing: 0.5px; text-transform: uppercase;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #1a4a7a, #0d3060);
        border: 1px solid #2b5278;
        border-radius: 18px 18px 4px 18px;
        padding: 0.9rem 1.2rem; margin: 0 0 1rem;
        color: #e8f4fd; font-size: 0.95rem;
        line-height: 1.6; max-width: 85%;
        margin-left: auto;
    }
    .chat-bubble-bot {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 18px 18px 18px 4px;
        padding: 0.9rem 1.2rem; margin: 0 0 0.5rem;
        color: #d1e8f5; font-size: 0.95rem;
        line-height: 1.6; max-width: 85%;
    }

    .source-card {
        background: #141921;
        border: 1px solid #2d3748;
        border-left: 3px solid #3182ce;
        border-radius: 8px;
        padding: 0.75rem 1rem; margin: 0.4rem 0;
        font-size: 0.82rem; color: #a0aec0; line-height: 1.5;
    }
    .source-card .source-meta {
        font-size: 0.72rem; color: #4a90d9;
        font-weight: 600; margin-bottom: 4px;
    }

    .stat-box {
        background: #141921; border: 1px solid #2d3748;
        border-radius: 10px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; text-align: center;
    }
    .stat-box .stat-number { font-size: 1.6rem; font-weight: 700; color: #63b3ed; }
    .stat-box .stat-label  { font-size: 0.72rem; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }

    .info-card {
        background: #141921; border: 1px solid #2d3748;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.82rem; color: #a0aec0; line-height: 1.9;
    }

    .chroma-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: #0d2a1a; border: 1px solid #276749;
        border-radius: 8px; padding: 6px 12px;
        font-size: 0.78rem; color: #68d391;
        font-weight: 600; margin-top: 0.5rem;
    }

    .stTextInput > div > div > input {
        background: #1a1f2e !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3182ce !important;
        box-shadow: 0 0 0 2px rgba(49,130,206,0.3) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0, #1a4a7a) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 0.9rem !important; padding: 0.6rem 1.5rem !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
        transform: translateY(-1px) !important;
    }

    .disclaimer {
        background: #1c1a10; border: 1px solid #744210;
        border-left: 3px solid #d69e2e; border-radius: 8px;
        padding: 0.7rem 1rem; font-size: 0.78rem;
        color: #b7791f; margin-top: 1rem; line-height: 1.5;
    }

    .empty-state { text-align: center; padding: 3rem 1rem; }
    .empty-state .icon  { font-size: 3rem; margin-bottom: 1rem; }
    .empty-state .title { font-size: 1.1rem; color: #718096; }
    .empty-state .sub   { font-size: 0.85rem; color: #4a5568; margin-top: 0.5rem; }

    hr { border-color: #2d3748 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
DB_CHROMA_PATH   = "vectorstore/db_chroma"
GROQ_MODEL_NAME  = "llama-3.1-8b-instant"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

SAMPLE_QUESTIONS = [
    "What are symptoms of diabetes?",
    "How is hypertension treated?",
    "What causes chest pain?",
    "What is the dosage of Paracetamol?",
    "What are signs of a heart attack?",
]

# ─── Cached Resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory=DB_CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="medbot_collection"
    )
    return db

@st.cache_resource(show_spinner=False)
def load_llm():
    load_dotenv()
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=GROQ_MODEL_NAME,
        temperature=0.5,
        max_tokens=512,
    )

def build_qa_chain(db, llm):
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

# ─── Session State ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "prefill_query" not in st.session_state: st.session_state.prefill_query = ""
if "input_key"     not in st.session_state: st.session_state.input_key     = 0

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🩺</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:4px;'>MedBot</div>
        <div style='font-size:0.75rem; color:#718096;'>AI Medical Knowledge Base</div>
        <div class='chroma-badge'>🟢 ChromaDB Connected</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Stats
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-number'>{st.session_state.total_queries}</div>
            <div class='stat-label'>Queries</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-number'>{len(st.session_state.messages) // 2}</div>
            <div class='stat-label'>Turns</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Model info
    st.markdown("""<div style='font-size:0.72rem; color:#718096; text-transform:uppercase;
    letter-spacing:0.5px; font-weight:600; margin-bottom:0.5rem;'>Stack Info</div>""",
    unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        🤖 <b style='color:#63b3ed;'>LLM:</b> Llama 3.1 8B Instant<br>
        ⚡ <b style='color:#63b3ed;'>Provider:</b> Groq<br>
        🔎 <b style='color:#63b3ed;'>Embeddings:</b> MiniLM-L6-v2<br>
        🗄️ <b style='color:#68d391;'>Vector DB:</b> ChromaDB (Local)<br>
        📦 <b style='color:#63b3ed;'>Framework:</b> LangChain
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Sample questions
    st.markdown("""<div style='font-size:0.72rem; color:#718096; text-transform:uppercase;
    letter-spacing:0.5px; font-weight:600; margin-bottom:0.5rem;'>Sample Questions</div>""",
    unsafe_allow_html=True)

    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sq_{q}"):
            st.session_state.prefill_query = q
            st.rerun()

    st.markdown("---")

    if st.button("🗑️  Clear Conversation"):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <b>Medical Disclaimer:</b> This tool is for informational purposes only.
        Always consult a qualified medical professional for diagnosis or treatment.
    </div>""", unsafe_allow_html=True)

# ─── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🩺 MedBot — AI Medical Assistant</h1>
    <p>Ask any medical question and get evidence-based answers from your knowledge base</p>
    <span class='badge'>RAG · Groq Llama 3.1 · ChromaDB · HuggingFace</span>
</div>
""", unsafe_allow_html=True)

# Load resources
with st.spinner("⚙️ Connecting to ChromaDB and loading model..."):
    try:
        db       = load_vectorstore()
        llm      = load_llm()
        qa_chain = build_qa_chain(db, llm)
        vec_count = db._collection.count()
    except Exception as e:
        st.error(f"❌ Startup error: {e}")
        st.stop()

# Show vector count
st.markdown(f"""
<div style='display:flex; justify-content:flex-end; margin-bottom:1rem;'>
    <div style='background:#0d2a1a; border:1px solid #276749; border-radius:8px;
                padding:5px 14px; font-size:0.78rem; color:#68d391; font-weight:600;'>
        🗄️ ChromaDB &nbsp;·&nbsp; {vec_count:,} vectors indexed
    </div>
</div>""", unsafe_allow_html=True)

# ─── Chat History ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown("<div class='chat-label-user'>You</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='chat-label-bot'>🩺 MedBot</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-bot'>{msg['content']}</div>", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📄 View Source Documents", expanded=False):
                    for i, doc in enumerate(msg["sources"], 1):
                        src     = doc.metadata.get("source", "Unknown").split("/")[-1]
                        page    = doc.metadata.get("page", "N/A")
                        snippet = doc.page_content[:350]
                        if len(doc.page_content) > 350:
                            snippet += "..."
                        st.markdown(f"""
                        <div class='source-card'>
                            <div class='source-meta'>📄 Source {i} &nbsp;·&nbsp; {src} &nbsp;·&nbsp; Page {page}</div>
                            {snippet}
                        </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='empty-state'>
        <div class='icon'>💬</div>
        <div class='title'>Ask your first medical question</div>
        <div class='sub'>Use the sample questions in the sidebar, or type your own below</div>
    </div>""", unsafe_allow_html=True)

# ─── Input ─────────────────────────────────────────────────────────────────────
st.markdown("---")

col_input, col_btn = st.columns([5, 1])

with col_input:
    default_val = st.session_state.pop("prefill_query", "") or ""
    user_query  = st.text_input(
        label="query",
        value=default_val,
        placeholder="e.g. What are the early symptoms of diabetes?",
        label_visibility="collapsed",
        key=f"query_input_{st.session_state.input_key}",
    )

with col_btn:
    ask_clicked = st.button("Ask →", key="ask_btn")

# ─── Greeting & Conversational Intent Detection ────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "hi there", "hello there", "hey there",
    "whats up", "what's up", "sup",
    "how are you", "how are you doing", "how do you do",
    "how's it going", "hows it going",
}

THANKS = {
    "thanks", "thank you", "thank you so much", "thanks a lot",
    "many thanks", "thx", "ty", "cheers",
}

GOODBYE = {
    "bye", "goodbye", "good bye", "see you", "see ya",
    "take care", "later", "cya",
}

GREETING_REPLY = (
    "Hello! 👋 Welcome to **MedBot**, your AI-powered medical assistant. "
    "I'm here to help you with medical questions based on my knowledge base. "
    "Feel free to ask me about symptoms, treatments, medications, or any health-related topics. "
    "How can I assist you today?"
)

THANKS_REPLY = (
    "You're welcome! 😊 If you have any more medical questions, feel free to ask. "
    "I'm here to help anytime!"
)

GOODBYE_REPLY = (
    "Goodbye! 👋 Take care of your health. "
    "Feel free to come back anytime you have medical questions. Stay well! 🩺"
)

def detect_intent(text: str) -> str:
    """Returns 'greeting', 'thanks', 'goodbye', or 'medical'."""
    normalized = text.lower().strip().rstrip("!?.").strip()
    if normalized in GREETINGS:
        return "greeting"
    if normalized in THANKS:
        return "thanks"
    if normalized in GOODBYE:
        return "goodbye"
    return "medical"


# ─── Run Query ─────────────────────────────────────────────────────────────────
if ask_clicked and user_query.strip():
    query   = user_query.strip()
    intent  = detect_intent(query)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.total_queries += 1

    if intent == "greeting":
        answer  = GREETING_REPLY
        sources = []

    elif intent == "thanks":
        answer  = THANKS_REPLY
        sources = []

    elif intent == "goodbye":
        answer  = GOODBYE_REPLY
        sources = []

    else:
        with st.spinner("🔍 Searching ChromaDB..."):
            try:
                response = qa_chain.invoke({"query": query})
                answer   = response["result"]
                sources  = response.get("source_documents", [])
            except Exception as e:
                answer  = f"⚠️ An error occurred: {e}"
                sources = []

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
    # Clear input by bumping the widget key
    st.session_state.input_key += 1
    st.rerun()
