import streamlit as st
from pipeline_v2 import run_pipeline_with_timeout

st.set_page_config(
    page_title="Skin Cancer QA",
    page_icon="",
    layout="wide"
)

st.title("Skin Cancer Question Answering System")
st.markdown(
    "Ask questions about skin cancer — symptoms, treatments, risk factors, and more. "
    "Answers are grounded in a verified medical ontology."
)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sparql_history" not in st.session_state:
    st.session_state.sparql_history = []

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about skin cancer...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                sparql, results, answer = run_pipeline_with_timeout(
                    question,
                    max_retries=2,
                    timeout_seconds=120
                )

            if answer:
                st.write(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                st.session_state.sparql_history.append({
                    "question": question,
                    "sparql": sparql,
                    "results": results,
                    "answer": answer
                })
            else:
                msg = "I could not find an answer in the ontology for that question."
                st.write(msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg
                })

with col2:
    st.subheader("Query Details")

    if st.session_state.sparql_history:
        latest = st.session_state.sparql_history[-1]

        with st.expander("Generated SPARQL Query", expanded=True):
            st.code(latest["sparql"], language="sparql")

        with st.expander("Raw Ontology Results", expanded=True):
            st.text(latest["results"])

        if len(st.session_state.sparql_history) > 1:
            st.subheader("Previous Queries")
            for item in reversed(st.session_state.sparql_history[:-1]):
                with st.expander(f"Q: {item['question'][:40]}..."):
                    st.code(item["sparql"], language="sparql")
                    st.text(item["results"])
    else:
        st.info("Query details will appear here after you ask a question.")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses:
    - **Ontology** — skin cancer knowledge base
    - **Agent 1** — generates SPARQL queries
    - **Agent 2** — validates results
    - **Agent 3** — explains answers
    - **GraphDB** — ontology database
    - **Ollama** — local LLM (llama3.2)
    """)

    st.divider()
    st.header("Example Questions")
    examples = [
        "What are the symptoms of melanoma?",
        "How is basal cell carcinoma treated?",
        "Which cancers have UV exposure as a risk factor?",
        "Where does melanoma commonly appear?",
        "How is skin cancer diagnosed?",
    ]
    for ex in examples:
        st.markdown(f"• {ex}")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.sparql_history = []
        st.rerun()