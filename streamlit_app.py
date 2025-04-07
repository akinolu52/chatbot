import time

import pandas as pd
import streamlit as st
from app import EnhancedChatbot
import uuid
import os

title = "Netwiz AI Chatbot"

# Streamlit UI
st.set_page_config(page_title=title, page_icon="🤖")

# Custom CSS for better styling
st.html("""
<style>
    .stChatMessage {
        # width: 48%;
        background-color: rgba(38, 39, 48, 0.5);
    }
    [class*="st-key-assistant"] div[data-testid="stChatMessageContent"] div[data-testid="stCaptionContainer"] {
        margin-top: -1rem;
        text-align: right;
        margin-right: 1rem;
    }
    [class*="st-key-user"] {
        width: 48%;
        align-self: flex-end;
    }
    [class*="st-key-assistant"] {
        width: 48%;
        align-self: flex-start
    }
</style>
""")

def chat_message(name):
    return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name)

@st.cache_resource
def initialize_chatbot():
    """Initialize chatbot with model loading (not training)"""
    bot = EnhancedChatbot()
    # Disable automatic training in the constructor
    bot.nn.is_training = False
    return bot


def main():
    # st.set_page_config(page_title=title, page_icon="🤖")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
        st.session_state.messages = []

    # Sidebar for controls
    with st.sidebar:
        st.title("Chatbot Controls")

        if st.button("Retrain Model"):
            with st.spinner("Training model..."):
                losses = st.session_state.chatbot.train_model()
                st.line_chart(losses)
                st.success("Model retrained successfully!")

        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.chatbot.context = None
            st.experimental_rerun()

        if st.checkbox("Show Conversation History"):
            st.write(pd.DataFrame(st.session_state.chatbot.conversation_history))

        if st.checkbox("Show Model Metrics") and st.session_state.chatbot.test_results:
            metrics_df = pd.DataFrame(st.session_state.chatbot.test_results)
            st.write("Model Performance Metrics")
            st.dataframe(metrics_df)
            st.line_chart(metrics_df.drop(columns=['timestamp']))

    # Main chat interface
    st.title(f"{title} 🤖")
    # st.write("Ask me anything! I can help with various topics.")
    st.markdown("""
Welcome to your personal AI assistant! Ask me anything about:
- Greetings 
- Questions
- Jokes
- Weather
- And more!
""")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with chat_message(name="user"):
            st.write(prompt)

        start_time = time.time()

        # Get chatbot response
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.generate_response(prompt)

        response_time = time.time() - start_time

        # Display assistant response
        with chat_message(name="assistant"):
            st.write(response)
            st.caption(f"Response time: {response_time:.2f}s")

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
