# backend/rag/setup_rag.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.tools import tool
import tempfile

# ... (keep your existing get_qa_chain function)

def get_qa_chain(vector_db):
    # This function remains the same
    prompt_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_type="mistral",
    # Consolidate all settings into the config dictionary
    config={
        'context_length': 4096,
        'temperature': 0.5
    }
)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# --- NEW FUNCTION for processing uploaded files ---
def create_vector_db_from_files(uploaded_files, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads PDF documents from uploaded file objects, splits them, creates embeddings,
    and returns an in-memory FAISS vector store.
    """
    documents = []
    for file in uploaded_files:
        # PyPDFLoader needs a file path, so we save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents.extend(loader.load())
        
        # Clean up the temporary file
        os.remove(tmp_file_path)

    print(f"Loaded {len(documents)} document pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(texts, embeddings)
    print("In-memory FAISS index created.")
    
    return db


# --- MODIFIED TOOL CREATION ---
def create_financial_document_qa_tool(vector_db):
    """
    Creates a tool that performs RetrievalQA on the provided vector database.
    This now works with any vector_db, static or in-memory.
    """
    qa_chain = get_qa_chain(vector_db)

    @tool
    def financial_document_qa(query: str) -> str:
        """
        Answers questions about the uploaded financial documents by searching the vector database.
        Input should be a fully formed question.
        """
        response = qa_chain.invoke({"query": query})
        print(f"--- RAG TOOL RAW RESPONSE: {response} ---")
        return response["result"]
    
    return financial_document_qa # Returning the correct function name

