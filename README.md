# Adaptive RAG System

An intelligent question-answering system that combines local document retrieval with real-time web search capabilities. The system automatically chooses between local and web sources based on relevance and confidence scores.

## Features

- Local document retrieval using vector similarity search
- Real-time web search integration using Tavily API
- Intelligent source selection based on relevance scoring
- LLM-based response generation with source citations using HuggingFace models
- FastAPI-based REST API for easy integration

## Setup

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. The system is pre-configured with the Tavily API key. No additional configuration is needed.

3. Start the FastAPI server:
```bash
python app.py
```

## API Endpoints

### Query the System
```bash
POST /query
{
    "text": "Your question here"
}
```

### Add Documents
```bash
POST /documents
[
    {
        "content": "Document content",
        "source": "Document source"
    }
]
```

## System Architecture

The system consists of several key components:

1. **Document Processor**: Handles document chunking and vector storage
2. **Web Search**: Manages real-time web searches using Tavily API
3. **Adaptive RAG**: Coordinates between local and web search, generates responses using HuggingFace models
4. **FastAPI Server**: Provides REST API endpoints for system interaction

## Configuration

The system can be configured through the `config.py` file:

- `SIMILARITY_THRESHOLD`: Minimum similarity score for local results
- `MAX_LOCAL_RESULTS`: Maximum number of local results to retrieve
- `MAX_WEB_RESULTS`: Maximum number of web results to retrieve
- `CHUNK_SIZE`: Size of document chunks for processing
- `CHUNK_OVERLAP`: Overlap between document chunks

## Usage Example

```python
import requests

# Add documents
documents = [
    {
        "content": "Your document content here",
        "source": "Document source"
    }
]
requests.post("http://localhost:8000/documents", json=documents)

# Query the system
response = requests.post(
    "http://localhost:8000/query",
    json={"text": "Your question here"}
)
print(response.json())
``` 