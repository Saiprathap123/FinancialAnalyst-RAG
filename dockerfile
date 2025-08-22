# Use official (or popular) Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your full project (code, docs, etc.)
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Set Streamlit environment variables (disable telemetry, etc.)
ENV LANGCHAIN_TRACING_V2=false
ENV PYTHONUNBUFFERED=1

# Streamlit entrypoint (you may customize this as needed)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
