# rag/setup_rag.py

from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.tools import Tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool

# New imports for the agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

def get_answer_from_docs(query: str) -> str:
    """
    Answers questions based on the provided financial documents.
    """
    try:
        return rag_chain.invoke(query)['result']
    except Exception as e:
        return f"An error occurred: {str(e)}"

rag_tool = Tool.from_function(
    name="Financial_Document_QA",
    func=get_answer_from_docs,
    description="Useful for answering questions about financial reports, 10-K filings, and company documents. Input should be a question about a company's financials."
)

# 1. Embedding model for document chunks
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load FAISS vectorstore (with security flag)
vectorstore = FAISS.load_local("embeddings/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# 3. HuggingFace local model (lightweight and RAM-safe)
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
# FIX: Removed the 'prompt' argument, as it is no longer supported in this version of the library.
llm = HuggingFacePipeline(pipeline=pipe)

# 4. Create retriever and QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# FIX: Use .invoke() instead of .run() as .run() is being deprecated
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- NEW CODE STARTS HERE ---

# 5. Create the Web Search Tool
web_search_tool = DuckDuckGoSearchRun(
    name="Web_Search",
    description="Useful for finding real-time information, such as current stock prices, news, or general data not available in the provided documents."
)

# 6. Create the Python REPL Tool
python_repl_tool = PythonREPLTool()
python_repl_tool.name = "Python_REPL_Tool"
python_repl_tool.description = "Useful for executing Python code to perform calculations. Input should be a valid Python command or expression."

# 7. Assemble the full toolkit
toolkit = [rag_tool, web_search_tool, python_repl_tool]

# 8. Define the agent prompt
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful and knowledgeable financial assistant. You have access to the following tools:

{tools}

To use a tool, you must follow the correct format:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation cycle can repeat)

Thought: I have a final answer.
Final Answer: the final answer to the original input question.

Begin!
Question: {input}
{agent_scratchpad}
""")

# 9. Create the agent
agent = create_react_agent(llm, toolkit, prompt_template)

# 10. Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=toolkit, 
    verbose=True, 
    handle_parsing_errors=True
)
