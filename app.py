import streamlit as st
from rag.setup_rag import agent_executor

st.set_page_config(page_title="📊 Financial Analyst Agent", layout="centered")

st.title("📊 Financial Analyst Agent")
st.markdown("Ask financial questions based on documents, real-time data, or calculations.")

# Input box for user query
query = st.text_input("Enter your financial question 👇", "")

if query:
    with st.spinner("🔍 Agent is thinking and executing tools..."):
        try:
            # Use the agent_executor to invoke the agent
            # The agent will decide whether to use the RAG tool, web search, or Python REPL
            result = agent_executor.invoke({"input": query})
            
            # The agent's final answer is in the 'output' key of the result dictionary
            answer = result["output"]

            st.success("✅ Agent's Final Answer:")
            st.markdown(f"**{answer}**")

        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")
