from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import os
from document_processor import DocumentProcessor
from adaptive_rag import AdaptiveRAG
from config import settings
import uvicorn

app = FastAPI(title="Adaptive RAG System")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
document_processor = DocumentProcessor()
adaptive_rag = AdaptiveRAG()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main interface."""
    with open("static/index.html") as f:
        return f.read()

@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    try:
        for file in files:
            content = await file.read()
            # Process the document
            document_processor.process_documents([{
                "content": content.decode('utf-8'),
                "source": file.filename
            }])
        return {"message": "Files uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_documents(documents: List[dict]):
    """Add documents directly."""
    try:
        document_processor.process_documents(documents)
        return {"message": "Documents added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(text: dict):
    """Process a query."""
    try:
        result = adaptive_rag.query(text["text"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 