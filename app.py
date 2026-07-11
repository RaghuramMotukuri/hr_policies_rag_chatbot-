import sys
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None:
    print("[INFO] app.py was executed directly. Rerunning via Streamlit CLI...")
    try:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", __file__, "--server.headless", "true"]
        sys.exit(stcli.main())
    except Exception as e:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__, "--server.headless", "true"])
        sys.exit(0)

# Monkey-patch importlib.metadata.version to avoid NoneType version parses in Python 3.14
try:
    import importlib.metadata
    _orig_version = importlib.metadata.version
    def _patched_version(distribution_name):
        try:
            v = _orig_version(distribution_name)
            if v is None:
                if distribution_name == "torch":
                    return "2.13.0"
                return "0.0.0"
            return v
        except importlib.metadata.PackageNotFoundError:
            if distribution_name == "torch":
                return "2.13.0"
            raise
    importlib.metadata.version = _patched_version
except ImportError:
    pass

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration for a premium, clean layout
st.set_page_config(
    page_title="HR Policies RAG Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
    }
    
    .reportview-container {
        background: #f8fafc;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Modern Glassmorphic Loader style */
    .modern-loader {
        background: rgba(30, 58, 138, 0.03);
        border: 1px solid rgba(30, 58, 138, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
        animation: fadeIn 0.4s ease-out;
    }
    
    .modern-loader-ring {
        width: 44px;
        height: 44px;
        border: 4px solid rgba(59, 130, 246, 0.1);
        border-top: 4px solid #3B82F6;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin-bottom: 16px;
    }
    
    .modern-loader-pulse {
        width: 14px;
        height: 14px;
        background-color: #3B82F6;
        border-radius: 50%;
        animation: pulse-ring 1.2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
        margin-bottom: 16px;
    }

    .modern-loader-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }

    .modern-loader-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #64748B;
        max-width: 320px;
        line-height: 1.4;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes pulse-ring {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to render modern loaders
def get_query_loader_html():
    return """
    <div class="modern-loader">
        <div class="modern-loader-pulse"></div>
        <div class="modern-loader-title">Consulting HR Assistant</div>
        <div class="modern-loader-desc">Searching Firestore database and generating response via Llama 3.3...</div>
    </div>
    """

def get_indexing_loader_html(filename="Documents"):
    return f"""
    <div class="modern-loader">
        <div class="modern-loader-ring"></div>
        <div class="modern-loader-title">Updating Cloud Database</div>
        <div class="modern-loader-desc">Extracting pages, generating MiniLM embeddings, and uploading to Firebase...</div>
    </div>
    """

# Cache resources to prevent reloading models/connections on every rerun
@st.cache_resource
def get_vector_store():
    from src.vectorstore import FirestoreVectorStore
    return FirestoreVectorStore()

@st.cache_resource
def get_rag_search(_vectorstore):
    from src.search import RAGSearch
    return RAGSearch(vectorstore=_vectorstore)

# Initialize resources
try:
    store = get_vector_store()
    rag_search = get_rag_search(store)
    db_initialized = True
except Exception as e:
    st.error(f"Failed to initialize Firestore/Groq connection: {e}")
    db_initialized = False

# Sidebar layout
st.sidebar.title("💼 Control Panel")
st.sidebar.markdown("---")

# 1. File Uploader Section
st.sidebar.subheader("📤 Upload HR Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT policies:",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data/pdf", exist_ok=True)
    os.makedirs("data/text_files", exist_ok=True)
    
    new_files_saved = False
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            target_path = os.path.join("data/pdf", uploaded_file.name)
        else:
            target_path = os.path.join("data/text_files", uploaded_file.name)
            
        if not os.path.exists(target_path):
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Saved: {uploaded_file.name}")
            new_files_saved = True
            
    if new_files_saved:
        st.sidebar.info("New files saved locally. Sync them to Firebase using the button below.")

    if st.sidebar.button("🔄 Sync Uploaded Files to Firebase", use_container_width=True):
        if db_initialized:
            loader_sidebar = st.sidebar.empty()
            loader_sidebar.markdown(get_indexing_loader_html(), unsafe_allow_html=True)
            try:
                from src.data_loader import load_all_documents
                docs = load_all_documents("data")
                if docs:
                    store.build_from_documents(docs)
                    loader_sidebar.empty()
                    st.sidebar.success("Database indexed successfully! All chunks are uploaded to Firestore.")
                else:
                    loader_sidebar.empty()
                    st.sidebar.warning("No documents found in data folder to sync.")
            except Exception as e:
                loader_sidebar.empty()
                st.sidebar.error(f"Sync failed: {e}")
        else:
            st.sidebar.warning("Database not initialized. Please configure credentials and try again.")

st.sidebar.markdown("---")

# 3. Conversation Controls
st.sidebar.subheader("💬 Chat Controls")
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your HR assistant. Ask me anything about company HR policies."}
    ]
    st.rerun()

st.sidebar.markdown("---")

# 3. Connection status indicators
st.sidebar.subheader("📡 Status")
if db_initialized:
    st.sidebar.success("Firebase: Connected")
    if os.getenv("GROQ_API_KEY"):
        st.sidebar.success("Groq LLM: Ready")
    else:
        st.sidebar.warning("Groq LLM: Missing API Key")
else:
    st.sidebar.error("Firebase: Offline")

# Main Application Interface
st.markdown("""
<div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); padding: 2.2rem; border-radius: 16px; margin-bottom: 2rem; color: white; box-shadow: 0 10px 25px rgba(30, 58, 138, 0.12); position: relative; overflow: hidden;">
    <div style="position: absolute; right: -50px; top: -50px; width: 200px; height: 200px; background: rgba(255, 255, 255, 0.05); border-radius: 50%;"></div>
    <h1 style="color: white; margin: 0; font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2.3rem; display: flex; align-items: center; gap: 14px;">
        <span>🤖</span> HR Policies Assistant
    </h1>
    <p style="margin: 0.6rem 0 0 0; opacity: 0.9; font-family: 'Inter', sans-serif; font-size: 1.05rem; font-weight: 300; letter-spacing: 0.2px; line-height: 1.4;">
        A secure semantic search engine connected to Cloud Firestore, providing responses directly from employee handbooks, conduct guidelines, and statements.
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your HR assistant. Ask me anything about company HR policies."}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # If retrieved sources are stored in message, display them
        if "sources" in message and message["sources"]:
            with st.expander("📚 Source References", expanded=False):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** {source['file']} (Page {source['page']})")
                    st.info(source['text'])

# Chat Input
if query := st.chat_input("Enter your HR policy query here..."):
    # Display user message in chat
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Generate RAG response
    with st.chat_message("assistant"):
        if not db_initialized:
            st.error("Cannot perform search. Firebase database is offline.")
        else:
            try:
                # 1. Thinking / Retrieval Step-by-Step Status Indicator
                with st.status("🤔 Analyzing & Retrieving Context...", expanded=True) as status:
                    st.write("🔍 Analyzing query intent...")
                    st.write("📡 Scanning Firestore database for semantic matches...")
                    
                    # Retrieve raw search results
                    search_results = store.query(query, top_k=3)
                    
                    st.write(f"📚 Retrieved {len(search_results)} relevant document chunks:")
                    
                    # Format retrieved chunks for the sources expander
                    sources = []
                    for res in search_results:
                        meta = res["metadata"]
                        source_file = os.path.basename(meta.get("source_file", meta.get("source", "Unknown")))
                        page_num = meta.get("page", 0) + 1 # Convert 0-indexed to 1-indexed for display
                        st.write(f"   - Match: *{source_file}* (Page {page_num})")
                        sources.append({
                            "file": source_file,
                            "page": page_num,
                            "text": res["metadata"].get("text", "")
                        })
                    
                    st.write("🧠 Reading and analyzing text contents...")
                    st.write("✍️ Synthesizing concise semantic response...")
                    
                    # 2. Get LLM response using search.py
                    response = rag_search.search_and_summarize(query, chat_history=st.session_state.messages[:-1], top_k=3)
                    
                    # Mark status as complete and collapse it
                    status.update(label="🤔 Analysis Complete (click to review logs)", state="complete", expanded=False)
                
                # 3. Stream the response sequentially (typewriter effect)
                import time
                def response_streamer(text_content):
                    for chunk in text_content.split(" "):
                        yield chunk + " "
                        time.sleep(0.012)
                
                response_container = st.empty()
                streamed_response = response_container.write_stream(response_streamer(response))
                
                # Display retrieved sources
                if sources:
                    with st.expander("📚 Source References", expanded=False):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}:** {source['file']} (Page {source['page']})")
                            st.info(source['text'])
                
                # Save response to history (including the sources)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Ensure your Groq API key is correct and your Firestore composite index is fully built and Active.")
                
                # If we have partial search results (retrieval succeeded, generation failed)
                if 'search_results' in locals() and search_results:
                    st.info("Direct semantic search matches from database:")
                    for i, res in enumerate(search_results):
                        meta = res["metadata"]
                        source_file = os.path.basename(meta.get("source_file", meta.get("source", "Unknown")))
                        st.markdown(f"**Match {i+1}:** `{source_file}` (Page {meta.get('page', 0) + 1})")
                        st.write(res["metadata"].get("text", ""))