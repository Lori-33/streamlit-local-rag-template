"""Objection handling with local RAG retrieval and LLM drafting."""
import streamlit as st
from rag_engine import get_store
from objection_engine import (
    classify_objection,
    expand_queries,
    generate_objection_response,
    retrieve_evidence,
)
from config import DOCS_FOLDERS

st.set_page_config(page_title="Objection Handling", page_icon="🛡️", layout="wide")

st.title("🛡️ Objection Handling")
st.caption("Classify a concern, retrieve supporting evidence, and draft a grounded response.")

# ---- Input area ----
col1, col2 = st.columns([3, 1])

with col1:
    objection = st.text_area(
        "Customer concern",
        placeholder="For example: The price is too high / The setup looks complex / We already use another tool.",
        height=100,
    )

with col2:
    st.markdown("### Common scenarios")
    st.markdown("""
    - Price concern
    - Value uncertainty
    - Safety or risk concern
    - Competitor comparison
    - Adoption barrier
    - Evidence gap
    """)

# ---- Index status ----
store = get_store(DOCS_FOLDERS)
if not store.is_loaded:
    st.warning("⚠️ Index not ready. Rebuild it from Knowledge Q&A first.")
elif store.is_stale:
    st.warning(f"⚠️ Index needs rebuild: {store.stale_reason}")

# ---- Action ----
if st.button("🔍 Retrieve response guidance", type="primary", use_container_width=True, disabled=not objection.strip()):
    if not store.is_loaded or store.is_stale:
        st.error("Index not ready")
        st.stop()

    objection_type = classify_objection(objection)
    queries = expand_queries(objection, objection_type)

    st.markdown("### 🎯 Classification")
    st.info(f"Objection type: {objection_type}")

    with st.expander("🔎 Expanded search queries", expanded=False):
        for query in queries:
            st.caption(query)

    # Step 1: Multi-query retrieval and evidence filtering
    with st.spinner("📂 Retrieving and filtering evidence..."):
        try:
            results = retrieve_evidence(store, objection, objection_type, per_query_top_k=5, final_top_k=10)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    if not results:
        st.warning("⚠️ No related source material was found. Try rephrasing the concern.")
        st.stop()

    sources = [r["citation"] for r in results]

    st.markdown("### 📂 Evidence")
    with st.expander(f"Found {len(results)} relevant chunks", expanded=False):
        for i, r in enumerate(results, 1):
            st.caption(f"[{i}] {r['citation']} (score: {r['score']})")
            if r.get("matched_queries"):
                st.caption("Matched queries: " + " / ".join(r["matched_queries"][:3]))
            st.text(r["text"][:500])

    # Step 2: Generate
    with st.spinner("🤔 Drafting response guidance..."):
        llm_result = generate_objection_response(objection, objection_type, results)

    if not llm_result["ok"]:
        st.error(f"Generation failed: {llm_result['error']}")
        st.stop()

    st.markdown("### 🧭 Response Guidance")
    st.markdown(llm_result["content"])

    with st.expander("📎 Sources", expanded=False):
        for i, src in enumerate(sources, 1):
            st.caption(f"[{i}] {src}")
