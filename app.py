"""Main entry point. Run with: streamlit run app.py."""
import streamlit as st
from config import APP_TITLE, DOCS_FOLDERS

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")

st.title("🔎 Local RAG Assistant")

st.markdown("""
### Modules

| Module | What it does | Typical use |
|------|------|----------|
| 📚 **Knowledge Q&A** | Answers questions from your local documents | Research, onboarding, internal support |
| 🛡️ **Objection Handling** | Retrieves supporting material and drafts a response | Sales enablement, customer support, training |
| 📝 **Quiz** | Generates quiz questions from indexed content | Learning checks and practice |

---
### Architecture
- **Document loading**: local PDF/PPTX/TXT/Markdown parsing and chunking
- **Vector retrieval**: LM Studio embedding API
- **Answer generation**: LM Studio chat API
- **Privacy**: designed for local-first document workflows

---
### Status
""")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Check LLM service", use_container_width=True):
        with st.spinner("Checking..."):
            from llm_api import chat
            result = chat("Hello. Reply with 'connected'.")
            if result["ok"]:
                st.success("✅ LLM service is ready")
            else:
                st.error(f"❌ {result['error']}")

with col2:
    if st.button("🔍 Check knowledge index", use_container_width=True):
        from rag_engine import get_store
        store = get_store(DOCS_FOLDERS)
        if store.is_loaded:
            if store.is_stale:
                st.warning(f"⚠️ Index needs rebuild: {store.stale_reason}")
            else:
                dimension = store.embedding_dimension or "unknown"
                st.success(f"✅ Index ready: {store.chunk_count} chunks, {dimension} dimensions")
        else:
            st.warning("⚠️ Index not built. Open Knowledge Q&A and rebuild the index.")
