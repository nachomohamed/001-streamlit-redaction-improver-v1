import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

model = "gpt-4o-mini"

# Page config
st.set_page_config(page_title="LangChain Chatbot", layout="wide")

# API Key input (only once)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.sidebar.warning("Please provide your OpenAI API Key.")
    st.stop()

# Initialize LLM + memory
if "conversation" not in st.session_state:
    llm = ChatOpenAI(
        model_name=model,
        openai_api_key=openai_api_key,
        temperature=0.7
    )
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if user_input := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Get response from LangChain ConversationChain
    response = st.session_state.conversation.predict(input=user_input)

    # Add AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)

# Reset chat button
if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = []
    st.session_state.conversation.memory.clear()
    st.experimental_rerun()
