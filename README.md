# Financial Analyst RAG Assistant


An end-to-end web application that serves as an intelligent assistant for financial analysis. The core of this project is a multi-tool ReAct agent built with LangChain, designed to answer complex financial queries by leveraging Retrieval-Augmented Generation (RAG), real-time web search, and Python-based data analysis.

**This project demonstrates a comprehensive skill set in building production-ready AI applications, making it a powerful addition to a professional portfolio.**

**✨ Features**
Multi-Tool LLM Agent: A central, intelligent agent that can reason and select the most appropriate tool to answer a user's query. The agent is powered by a local, quantized Mistral-7B LLM, eliminating reliance on external APIs.

**Retrieval-Augmented Generation (RAG):** The ability to answer questions by retrieving information from a provided corpus of financial documents (PDFs) with a FAISS vector store and HuggingFace embeddings.

**Real-time Web Search**: A dedicated tool for fetching up-to-the-minute information like current stock prices or news headlines using the DuckDuckGo search API.

**Python REPL Tool:** The agent can execute Python code in a secure sandbox to perform calculations and data analysis on retrieved data.

**Interactive Chat UI: **A modern, single-page chat interface built with Streamlit that provides a seamless user experience for uploading documents and interacting with the agent.

**Containerization:** The entire application is configured for deployment with Docker, ensuring a consistent and portable environment.

**🚀 Technology Stack**
Backend & Orchestration: Python, Streamlit, LangChain

LLM: CTransformers with TheBloke/Mistral-7B-Instruct-v0.2-GGUF (local-first approach)

Embeddings: HuggingFaceEmbeddings with sentence-transformers/all-MiniLM-L6-v2

Vector Store: FAISS (local, file-based)

Tools: DuckDuckGo Search, Python REPL, Custom RAG Chain

Containerization: Docker

**⚙️ Setup and Installation**
**Prerequisites
**Python 3.10: It is highly recommended to use Python 3.10 to ensure compatibility with all dependencies.

Docker (Optional): Required for containerized deployment.

Local Installation
Clone the repository:

git clone https://github.com/your-username/financial_analyst_assistant.git
cd financial_analyst_assistant

Create and activate a virtual environment:

python -m venv venv
.\venv\Scripts\activate

Install all required dependencies:

pip install -r requirements.txt

**Create the initial vector database:**
Place any PDF documents you want to use (e.g., 10-K filings) into the docs/ folder. Then, run the data ingestion script:

python -m backend.rag.setup_rag

Note: This step is optional if you plan to upload files directly via the Streamlit UI, but it's useful for pre-populating the RAG system.

**🏃 Usage**
To launch the application, run the Streamlit app from the project's root directory:

streamlit run app.py

The application will open in your web browser. The first time you run it, the Mistral-7B LLM will be downloaded to your local machine (a one-time process). Once the model is loaded, you can upload your own financial documents via the sidebar and begin interacting with the AI assistant.

**🐳 Containerization**
For a production-ready and portable deployment, the project can be containerized using Docker.

Build the Docker image:

docker build -t financial-analyst-assistant .

Run the Docker container:

docker run -p 8501:8501 financial-analyst-assistant

The application will be accessible at http://localhost:8501.
## Example Queries

- “What was Netflix’s total revenue in 2024?”
- “Provide a breakdown of Apple’s net income by product line.”
- “Compare Microsoft and Apple’s R&D expenditure for the last fiscal year.”

---

---
"# FinancialAnalyst-RAG" 
