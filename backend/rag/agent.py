# backend/rag/agent.py

import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from .setup_rag import create_financial_document_qa_tool
from .tools import get_web_search_tool, get_python_repl_tool

def initialize_tools(vector_db):
    """Initializes all tools, including the RAG tool with the provided vector_db."""
    print("Initializing tools...")
    
    # The RAG tool is now created using the passed-in vector_db
    financial_qa_tool = create_financial_document_qa_tool(vector_db)
    
    web_search_tool = get_web_search_tool()
    python_repl_tool = get_python_repl_tool()

    return [financial_qa_tool, web_search_tool, python_repl_tool]

def create_financial_agent(vector_db):
    """Creates the financial agent, passing the vector_db to the tool initializer."""
    print("Creating financial agent...")

    llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_type="mistral",
    # Move context_length inside the config dictionary
    config={
        'context_length': 4096,
        'max_new_tokens': 2048,
        'temperature': 0.5
    }
)
    
    # This improved prompt is more directive to prevent parsing errors.
    prompt_template = """You are a helpful financial assistant. Your goal is to answer the user's question.

You have access to the following tools:
{tools}

To answer the question, you MUST use the following format. Do NOT deviate from this format.

Question: The user's question you must answer
Thought: You should always think about what to do.
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (this Thought/Action/Action Input/Observation sequence can repeat N times)
Thought: I now have enough information to answer the user's question.
Final Answer: The final answer to the original user question.

IMPORTANT:
- Only use the tools provided.
- If you have the final answer, provide it IMMEDIATELY after "Final Answer:". Do not add any extra conversation.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(prompt_template)
    
    # Pass the session-specific vector_db to the tool initializer
    tools = initialize_tools(vector_db)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor

# The __main__ block has been removed as agent creation is now handled
# by the Streamlit app based on user file uploads.

