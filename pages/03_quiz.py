"""Quiz generation from local RAG context."""
import json
import streamlit as st
from rag_engine import get_store
from llm_api import chat as llm_chat
from config import DOCS_FOLDERS

st.set_page_config(page_title="Quiz", page_icon="📝", layout="wide")

st.title("📝 Quiz")
st.caption("Generate practice questions from your indexed local documents.")

EXAM_GEN_PROMPT = """You are a training designer. Generate high-quality quiz questions from the provided source material.

Rules:
1. Generate {question_count} questions.
2. Mix single-choice, multi-choice, and true/false questions when possible.
3. Stay strictly grounded in the provided material.
4. Cover important concepts, procedures, limitations, examples, and risks.
5. Include the tested knowledge point for each question.
6. Return a JSON array only. Each item must use this shape:
  {{"type": "single/multi/tf", "question": "Question", "options": ["A. xxx", "B. xxx", "C. xxx", "D. xxx"], "answer": "A", "explanation": "Explanation", "knowledge_point": "Knowledge point"}}

Do not include markdown code fences."""

EVAL_PROMPT = """You are a fair quiz reviewer. Grade the following answers.

Questions and correct answers:
{questions_json}

User answers:
{student_answers}

Please provide:
1. Total score ({points_per_question} points per question, {total_points} total)
2. Per-question grading with a short explanation
3. Detailed explanations for incorrect answers
4. Weak knowledge areas
5. Suggestions for further study"""

# ---- session state ----
if "exam_questions" not in st.session_state:
    st.session_state.exam_questions = []
if "exam_answers" not in st.session_state:
    st.session_state.exam_answers = {}
if "exam_submitted" not in st.session_state:
    st.session_state.exam_submitted = False
if "exam_result" not in st.session_state:
    st.session_state.exam_result = None

# ---- Sidebar settings ----
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    exam_topic = st.text_input("Topic scope (optional)", placeholder="Leave empty to use all documents")
    question_count = st.slider("Question count", 3, 10, 5)

    store = get_store(DOCS_FOLDERS)
    if store.is_loaded and not store.is_stale:
        st.success(f"Index ready: {store.chunk_count} chunks")
    elif store.is_loaded:
        st.warning(f"Index needs rebuild: {store.stale_reason}")
    else:
        st.error("Index not built. Rebuild it from Knowledge Q&A first.")

    st.divider()

    if st.button("🎲 Generate quiz", type="primary", use_container_width=True):
        if not store.is_loaded or store.is_stale:
            st.error("Index not ready")
            st.stop()

        st.session_state.exam_questions = []
        st.session_state.exam_answers = {}
        st.session_state.exam_submitted = False
        st.session_state.exam_result = None

        search_query = exam_topic if exam_topic.strip() else "overview setup configuration features limitations examples risks"
        with st.spinner("📂 Searching for quiz material..."):
            try:
                results = store.search(search_query, top_k=8)
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.stop()

        if not results:
            st.error("Could not retrieve enough source material.")
            st.stop()

        context = "\n\n---\n\n".join([r["text"] for r in results])
        st.info(f"Retrieved {len(results)} relevant chunks. Generating questions...")

        with st.spinner(f"🤔 Generating {question_count} questions..."):
            gen_prompt = EXAM_GEN_PROMPT.format(question_count=question_count)
            gen_result = llm_chat(context[:6000], gen_prompt, temperature=0.7)

        if not gen_result["ok"]:
            st.error(f"Generation failed: {gen_result['error']}")
            st.stop()

        raw = gen_result["content"].strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:]) if lines[1:] else raw
            if raw.endswith("```"):
                raw = raw[:-3]
        try:
            questions = json.loads(raw)
            st.session_state.exam_questions = questions
            st.success(f"✅ Generated {len(questions)} questions.")
            st.rerun()
        except json.JSONDecodeError:
            st.error("Could not parse the quiz JSON. Please try again.")
            st.text(raw[:500])

# ---- Main area ----
if not st.session_state.exam_questions:
    st.info("Use the sidebar settings, then click Generate quiz.")
else:
    questions = st.session_state.exam_questions

    if not st.session_state.exam_submitted:
        st.markdown(f"### 📋 {len(questions)} questions")

        for i, q in enumerate(questions):
            st.markdown(f"---")
            type_label = {"single": "Single choice", "multi": "Multiple choice", "tf": "True/False"}.get(q["type"], q["type"])
            st.markdown(f"**Question {i+1}** [{type_label}] {q['question']}")

            key = f"q_{i}"
            if q["type"] == "single":
                st.session_state.exam_answers[key] = st.radio(
                    "Choose an answer", q.get("options", ["A", "B", "C", "D"]),
                    key=key, index=None, label_visibility="collapsed"
                )
            elif q["type"] == "multi":
                st.session_state.exam_answers[key] = st.multiselect(
                    "Choose answers", q.get("options", ["A", "B", "C", "D"]),
                    key=key, label_visibility="collapsed"
                )
            elif q["type"] == "tf":
                st.session_state.exam_answers[key] = st.radio(
                    "True or false", ["True", "False"],
                    key=key, index=None, label_visibility="collapsed"
                )

        st.markdown("---")
        unanswered = sum(1 for v in st.session_state.exam_answers.values()
                        if v is None or v == [])
        if unanswered > 0:
            st.warning(f"{unanswered} questions are unanswered")

        if st.button("📤 Submit", type="primary", use_container_width=True):
            st.session_state.exam_submitted = True
            student_str = ""
            for i, q in enumerate(questions):
                ans = st.session_state.exam_answers.get(f"q_{i}", "Unanswered")
                if isinstance(ans, list):
                    ans = ", ".join(ans)
                student_str += f"Question {i+1}: {ans}\n"

            with st.spinner("📝 Grading..."):
                pts = 100 // len(questions)
                eval_result = llm_chat(
                    EVAL_PROMPT.format(
                        questions_json=str(questions),
                        student_answers=student_str,
                        points_per_question=pts,
                        total_points=100,
                    ),
                    "You are a fair and strict quiz reviewer.",
                    temperature=0.1,
                )

            if eval_result["ok"]:
                st.session_state.exam_result = eval_result["content"]
            else:
                st.session_state.exam_result = f"Grading failed: {eval_result['error']}"
            st.rerun()

    else:
        st.markdown("### 📊 Results")
        if st.session_state.exam_result:
            st.markdown(st.session_state.exam_result)

        st.divider()
        with st.expander("📋 Correct answers", expanded=False):
            for i, q in enumerate(questions):
                st.markdown(f"**Question {i+1}** - Correct answer: **{q.get('answer', 'N/A')}**")
                if q.get("explanation"):
                    st.caption(f"Explanation: {q['explanation']}")

        if st.button("🔄 Start over", use_container_width=True):
            st.session_state.exam_questions = []
            st.session_state.exam_answers = {}
            st.session_state.exam_submitted = False
            st.session_state.exam_result = None
            st.rerun()
