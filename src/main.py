import html
import uuid

import streamlit as st
from langchain_helper import (
    get_qa_chain,
    get_retriever,
    create_vector_db,
    format_chat_history,
    retrieve_docs,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopEase Support Bot",
    page_icon="🛒",
    layout="centered",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}
.bot-header { text-align: center; padding: 2rem 0 1rem 0; }
.bot-header h1 {
    font-family: 'Sora', sans-serif; font-size: 2rem;
    font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 0.25rem;
}
.bot-header p { color: #8b8fa8; font-size: 0.9rem; }

.chat-row { display: flex; margin: 0.6rem 0; align-items: flex-end; gap: 0.6rem; }
.chat-row.user { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar.bot  { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
.avatar.user { background: linear-gradient(135deg, #0ea5e9, #06b6d4); }

.bubble { max-width: 75%; padding: 0.75rem 1rem; border-radius: 18px; font-size: 0.92rem; line-height: 1.55; }
.bubble.bot  { background: #1e2030; border-bottom-left-radius: 4px; color: #dde1f0; }
.bubble.user { background: linear-gradient(135deg, #1d4ed8, #2563eb); border-bottom-right-radius: 4px; color: #ffffff; }

.source-pill {
    display: inline-block; background: #2a2d3e; border: 1px solid #3a3f5c;
    border-radius: 999px; padding: 0.2rem 0.7rem; font-size: 0.75rem;
    color: #7c85b3; margin-top: 0.4rem; margin-left: 3rem;
}
section[data-testid="stSidebar"] { background: #161824; border-right: 1px solid #2a2d3e; }
.stTextInput > div > div > input {
    background: #1e2030 !important; border: 1px solid #3a3f5c !important;
    border-radius: 12px !important; color: #e8e8e8 !important;
    padding: 0.75rem 1rem !important; font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.5rem 1.5rem !important; font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
hr { border-color: #2a2d3e !important; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Use a dict keyed by a stable UUID per message to avoid positional index bugs
if "sources" not in st.session_state:
    st.session_state.sources = {}


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗄️ Knowledge Base")
    st.markdown("Click below to build the FAQ vector store from the dataset.")
    st.markdown("---")

    if st.button("⚡ Build Knowledge Base"):
        with st.spinner("Embedding FAQs into vector store..."):
            try:
                count = create_vector_db()
                # Reset so chain/retriever are reloaded fresh on next question
                st.session_state.chain = None
                st.session_state.retriever = None
                st.success(f"Done! {count} FAQs indexed.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**Sample Questions**")
    sample_qs = [
        "How do I track my order?",
        "What is your return policy?",
        "Do you offer EMI options?",
        "My payment failed but money was deducted.",
        "How do I cancel my order?",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sample_{q}"):
            st.session_state["prefill"] = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.session_state["user_input"] = ""
        st.rerun()


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bot-header">
    <h1>🛒 ShopEase Support Bot</h1>
    <p>Hi! I'm your ShopEase assistant. Ask me about orders, returns, payments and more.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ── RENDER CHAT HISTORY ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    # Sanitize content before injecting into HTML to prevent XSS
    safe_content = html.escape(msg["content"])

    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-row user">
            <div class="avatar user">🧑</div>
            <div class="bubble user">{safe_content}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row bot">
            <div class="avatar bot">🤖</div>
            <div class="bubble bot">{safe_content}</div>
        </div>""", unsafe_allow_html=True)

        # Show source pill if this message has a matched FAQ source
        msg_id = msg.get("id")
        if msg_id and msg_id in st.session_state.sources:
            src = html.escape(st.session_state.sources[msg_id])
            st.markdown(
                f'<div class="source-pill">📎 Matched FAQ: {src[:80]}{"..." if len(src) > 80 else ""}</div>',
                unsafe_allow_html=True,
            )


# ── INPUT ──────────────────────────────────────────────────────────────────────

# Pop prefill from sidebar sample buttons (if any)
default_val = st.session_state.pop("prefill", "")

# Use a counter to force a fresh widget (new key = cleared input box)
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

question = st.text_input(
    label="Your question",
    label_visibility="collapsed",
    placeholder="Type your question here and press Enter…",
    value=default_val,
    key=f"user_input_{st.session_state.input_counter}",
)


# ── HANDLE QUESTION ────────────────────────────────────────────────────────────
if question:
    # Assign a stable UUID to every new user message (used for source linking)
    user_msg_id = str(uuid.uuid4())
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "id": user_msg_id,
    })

    # Lazy-load chain and retriever (only on first question after build)
    if st.session_state.chain is None:
        try:
            st.session_state.chain = get_qa_chain()
            st.session_state.retriever = get_retriever()
        except FileNotFoundError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Could not load knowledge base: {e}")
            st.stop()

    with st.spinner("Thinking…"):
        try:
            # Step 1: retrieve relevant FAQ docs via the modern .invoke() API
            docs, context = retrieve_docs(st.session_state.retriever, question)

            # Step 2: build chat history string from previous messages (excluding
            # the one we just appended so it isn't double-counted)
            history_str = format_chat_history(st.session_state.messages[:-1])

            # Step 3: call LLMChain with all three variables
            result = st.session_state.chain.invoke({
                "question": question,
                "chat_history": history_str,
                "context": context,
            })

            answer = result["text"]

            # Step 4: assign a UUID to the assistant message and store source
            assistant_msg_id = str(uuid.uuid4())
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "id": assistant_msg_id,
            })

            # Store source keyed by stable UUID — not a positional index
            if docs:
                source_text = docs[0].page_content.split("\n")[0]
                st.session_state.sources[assistant_msg_id] = source_text

        except Exception as e:
            error_id = str(uuid.uuid4())
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}",
                "id": error_id,
            })

    st.session_state.input_counter += 1
    st.rerun()

