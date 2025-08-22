# app.py

import streamlit as st
import os
from backend.rag.agent import create_financial_agent
# We now import the function to create a vector store from uploaded files
from backend.rag.setup_rag import create_vector_db_from_files

st.set_page_config(page_title="Financial Analyst Assistant", page_icon="🤖")
st.title("Financial Analyst Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None


st.sidebar.header("Upload Your Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload your 10-K filings or other financial PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)



# The agent is now created only when files are uploaded.
if uploaded_files:
    
    file_names = sorted([file.name for file in uploaded_files])
    session_key = "".join(file_names)
    
    if st.session_state.get("session_key") != session_key:
        with st.spinner("Processing documents and initializing agent... This may take a moment."):
            # Create a vector database in-memory from the uploaded files
            vector_db = create_vector_db_from_files(uploaded_files)
            
            
            # The vector_db is now correctly passed to the agent creation function
            st.session_state.agent_executor = create_financial_agent(vector_db)
            st.session_state.session_key = session_key 
            st.success("AI Agent is ready! Ask questions about your documents.")
else:
    # Clear the agent and session key if no files are uploaded
    st.session_state.agent_executor = None
    st.session_state.session_key = None
    st.info("Please upload your financial documents in the sidebar to begin.")


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Only show the chat input if the agent has been initialized
if st.session_state.agent_executor:
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent_executor.invoke({"input": prompt})
                    response_text = result.get("output", str(result))
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
