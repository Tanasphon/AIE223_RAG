import streamlit as st
from rag_core import RAGSystem
import re

# Function to detect non-English text
def is_non_english(text):
    # Check for any non-ASCII characters (which includes all non-English characters)
    return bool(re.search(r'[^\x00-\x7F]', text))

# Page config
st.set_page_config(
    page_title="Cat Breed Information Assistant",
    page_icon="🐱",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align:center;'>
            <h2 style='color:#4F8BF9;'>🐾 Cat RAG Chatbot</h2>
            <p style='color:#555;'>Your expert assistant for all things about cats and cat breeds!</p>
        </div>
        <hr style='border:1px solid #4F8BF9;'>
    """, unsafe_allow_html=True)
    st.markdown("""
        <b>What can you ask?</b>
        <ul style='margin:0 0 0 18px;padding:0;'>
            <li>Cat breeds and their characteristics</li>
            <li>History and origins of cats</li>
            <li>Breed comparison and recommendations</li>
            <li>Fun facts about cats</li>
        </ul>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='color:#888;font-size:13px;margin-top:20px;'>
            <b>How to use:</b> Type your question in the input box below and press <b>Send</b>. The chatbot will answer and evaluate the quality of its response.
        </div>
        <hr style='border:1px solid #eee;'>
        <div style='text-align:center;color:#aaa;font-size:12px;'>
            &copy; 2024 Cat RAG Chatbot<br>
            Powered by RAG + LLM
        </div>
    """, unsafe_allow_html=True)

# Main UI
st.markdown("# 🐱 Cat RAG Chatbot")
st.markdown("---")
st.markdown("### ✏️ Ask a question about cats")

# Display chat history (custom style)
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(
            f"<span style='color:#4F8BF9;font-weight:bold;'>You:</span> "
            f"<span style='color:#fff;'>{message['content']}</span>",
            unsafe_allow_html=True
        )
    elif not message.get('html'):
        st.markdown(
            f"<span style='color:#FF9800;font-weight:bold;'>Bot:</span> "
            f"<span style='color:#eee;'>{message['content']}</span>",
            unsafe_allow_html=True
        )
        st.markdown('---')

# Fixed ask bar at the bottom (move to after chat history)
st.markdown("""
    <div id="ask-bar-fixed" style="position:fixed;left:0;right:0;bottom:0;z-index:100;background:#181921;padding:20px 24px 16px 24px;box-shadow:0 -2px 12px rgba(0,0,0,0.15);">
        <div style='max-width:900px;margin:auto;'>
""", unsafe_allow_html=True)
user_input = st.text_input("Type your question here...", key="user_input")
st.markdown("</div></div>", unsafe_allow_html=True)

# Input area
if user_input.strip() and (st.session_state.get('last_user_input') != user_input):
    st.session_state['last_user_input'] = user_input

    # Prepare chat_history that does NOT include the current user message
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Language check
    if is_non_english(user_input):
        bot_msg = (
            "Sorry, please type your question in English.<br>"
            "ขออภัย กรุณาพิมพ์คำถามเป็นภาษาอังกฤษ<br>"
            "请用英语输入您的问题"
        )
        st.session_state.messages.append({"role": "assistant", "content": bot_msg, "html": True})
    else:
        rag = st.session_state.rag_system
        with st.spinner("Thinking..."):
            # Send chat_history that does NOT include the current user_input
            answer = rag.query(user_input, chat_history=chat_history)
            context = rag.search_documents(user_input)
            quality = rag.check_answer_quality(answer, user_input, context)
            # Append user and assistant messages after getting the answer
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            # Show the latest answer
            st.markdown("### 🐾 Bot's Answer")
            st.markdown(answer)

# Footer
st.markdown("---")
st.markdown("<center>© 2024 Cat RAG Chatbot</center>", unsafe_allow_html=True)