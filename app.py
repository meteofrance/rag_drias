import streamlit as st
import os

# Add IS_STREAMLIT to the environment
os.environ["IS_STREAMLIT"] = "True"
from main import answer

correct_password = st.secrets["general"]["password"]

# Password protection
password = st.text_input("Password", type="password")
if password == correct_password:
    st.success("Correct password")
elif password == "":
    st.stop()
else:
    st.error("Incorrect password")
    st.stop()

st.title("💬☀️ Chatbot DRIAS")


# Sidebar
st.sidebar.title("Parameters")

n_samples = st.sidebar.slider(
    "Number of retrieved chunks :", min_value=5, max_value=100, value=30
)
use_rag = st.sidebar.checkbox("Use rag", value=True)

generative_model = st.sidebar.selectbox(
    "Choose a generative model:",
    ["Llama-3.2-3B-Instruct", "Chocolatine-3B-Instruct-DPO-v1.0"],
)
reranker_model = st.sidebar.selectbox(
    "Choose a reranker model:", ["", "bge-reranker-v2-m3"]
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Streamed response emulator
    def response_generator():
        response = answer(
            prompt,
            generative_model=generative_model,
            n_samples=n_samples,
            use_rag=use_rag,
            reranker=reranker_model,
        )

        yield response

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
