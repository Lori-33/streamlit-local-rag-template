"""Knowledge Q&A with local RAG retrieval and LLM generation."""
import streamlit as st
from rag_engine import get_store
from llm_api import chat as llm_chat
from config import DOCS_FOLDERS

st.set_page_config(page_title="Knowledge Q&A", page_icon="📚", layout="wide")

st.title("📚 Knowledge Q&A")
st.caption("Ask questions against your local documents.")

QA_SYSTEM_PROMPT = """You are a careful knowledge assistant. Answer the user's question using only the provided source material.

Rules:
1. Stay strictly grounded in the supplied context.
2. If the context does not contain the answer, say that the source material does not include enough information.
3. Include source references when possible.
4. Use clear, practical language."""

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 💡 Suggestions")
    st.markdown("""
    **Try asking about:**
    - Setup or configuration
    - Feature behavior
    - Policy or process details
    - Risks, limitations, or examples
    """)

    st.divider()
    st.markdown("### 📂 Index")

    store = get_store(DOCS_FOLDERS)
    if store.is_loaded and not store.is_stale:
        st.success(f"Index ready: {store.chunk_count} chunks")
    elif store.is_loaded:
        st.warning(f"Index needs rebuild: {store.stale_reason}")
    else:
        st.error("Index not built")

    if st.button("🔄 Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding..."):
            from rag_engine import rebuild_store
            try:
                store = rebuild_store(DOCS_FOLDERS)
                st.success(f"Done: {store.chunk_count} chunks")
                st.rerun()
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

# ---- Chat area ----
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []

for msg in st.session_state.qa_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"[{i}] {src}")

if question := st.chat_input("Ask a question about your documents..."):
    store = get_store(DOCS_FOLDERS)
    if not store.is_loaded or store.is_stale:
        st.error("The knowledge index is not ready. Rebuild it from the sidebar first.")
        st.stop()

    st.session_state.qa_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Step 1: Retrieve
        with st.spinner("📂 Searching local documents..."):
            try:
                results = store.search(question, top_k=8)
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.stop()

        if not results:
            no_result = "⚠️ No relevant source material was found. Try rephrasing the question."
            st.warning(no_result)
            st.session_state.qa_messages.append({"role": "assistant", "content": no_result, "sources": []})
            st.stop()

        context = "\n\n---\n\n".join([r["text"] for r in results])
        sources = [r["citation"] for r in results]

        # Step 2: Generate
        with st.spinner("🤔 Drafting answer..."):
            llm_result = llm_chat(
                f"User question: {question}\n\nRelevant source material:\n{context[:5000]}",
                QA_SYSTEM_PROMPT,
                temperature=0.3,
            )

        if llm_result["ok"]:
            answer = llm_result["content"]
            st.markdown(answer)
            with st.expander("📎 Sources"):
                for i, src in enumerate(sources, 1):
                    st.caption(f"[{i}] {src}")
            st.session_state.qa_messages.append({"role": "assistant", "content": answer, "sources": sources})
        else:
            error_msg = f"⚠️ {llm_result['error']}"
            st.error(error_msg)
            st.session_state.qa_messages.append({"role": "assistant", "content": error_msg, "sources": []})

if st.session_state.qa_messages:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.qa_messages = []
        st.rerun()
