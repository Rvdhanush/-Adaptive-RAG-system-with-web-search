from typing import List, Dict, Any, Optional
import os
import json
import magic
import pytesseract
from PIL import Image
from datetime import datetime
from slugify import slugify
import pandas as pd
from pathlib import Path
import aiofiles
import asyncio
from pydantic import BaseModel

class DocumentMetadata(BaseModel):
    """Model for document metadata."""
    id: str
    title: str
    source: str
    file_type: str
    created_at: datetime
    updated_at: datetime
    size: int
    tags: List[str]
    category: Optional[str]
    author: Optional[str]
    version: int = 1

class DocumentManager:
    def __init__(self, base_dir: str = "./documents"):
        self.base_dir = Path(base_dir)
        self.metadata_dir = self.base_dir / "metadata"
        self.versions_dir = self.base_dir / "versions"
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.base_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)
        
    async def save_document(self, file_content: bytes, filename: str, metadata: Dict[str, Any]) -> DocumentMetadata:
        """Save a document and its metadata."""
        # Generate unique ID and sanitize filename
        doc_id = slugify(filename) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        file_type = magic.from_buffer(file_content, mime=True)
        
        # Create document metadata
        doc_metadata = DocumentMetadata(
            id=doc_id,
            title=filename,
            source=filename,
            file_type=file_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            size=len(file_content),
            tags=metadata.get("tags", []),
            category=metadata.get("category"),
            author=metadata.get("author")
        )
        
        # Save document
        doc_path = self.base_dir / f"{doc_id}{self._get_extension(file_type)}"
        async with aiofiles.open(doc_path, 'wb') as f:
            await f.write(file_content)
            
        # Save metadata
        await self._save_metadata(doc_metadata)
        
        return doc_metadata
        
    async def _save_metadata(self, metadata: DocumentMetadata):
        """Save document metadata to JSON file."""
        metadata_path = self.metadata_dir / f"{metadata.id}.json"
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(metadata.json())
            
    def _get_extension(self, mime_type: str) -> str:
        """Get file extension from MIME type."""
        mime_map = {
            "application/pdf": ".pdf",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "text/plain": ".txt"
        }
        return mime_map.get(mime_type, "")
        
    async def get_document(self, doc_id: str) -> Optional[bytes]:
        """Retrieve a document by ID."""
        metadata = await self.get_metadata(doc_id)
        if not metadata:
            return None
            
        doc_path = self.base_dir / f"{doc_id}{self._get_extension(metadata.file_type)}"
        if not doc_path.exists():
            return None
            
        async with aiofiles.open(doc_path, 'rb') as f:
            return await f.read()
            
    async def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata by ID."""
        metadata_path = self.metadata_dir / f"{doc_id}.json"
        if not metadata_path.exists():
            return None
            
        async with aiofiles.open(metadata_path, 'r') as f:
            content = await f.read()
            return DocumentMetadata.parse_raw(content)
            
    async def update_metadata(self, doc_id: str, updates: Dict[str, Any]) -> Optional[DocumentMetadata]:
        """Update document metadata."""
        metadata = await self.get_metadata(doc_id)
        if not metadata:
            return None
            
        # Update fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
                
        metadata.updated_at = datetime.now()
        await self._save_metadata(metadata)
        return metadata
        
    async def list_documents(self, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[DocumentMetadata]:
        """List all documents with optional filtering."""
        documents = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            async with aiofiles.open(metadata_file, 'r') as f:
                content = await f.read()
                doc = DocumentMetadata.parse_raw(content)
                
                # Apply filters
                if category and doc.category != category:
                    continue
                if tags and not all(tag in doc.tags for tag in tags):
                    continue
                    
                documents.append(doc)
                
        return sorted(documents, key=lambda x: x.updated_at, reverse=True)
        
    async def extract_text(self, doc_id: str) -> Optional[str]:
        """Extract text from a document."""
        metadata = await self.get_metadata(doc_id)
        if not metadata:
            return None
            
        content = await self.get_document(doc_id)
        if not content:
            return None
            
        if metadata.file_type.startswith("image/"):
            # Handle image OCR
            image = Image.open(io.BytesIO(content))
            return pytesseract.image_to_string(image)
        elif metadata.file_type == "application/pdf":
            # Handle PDF
            # Implementation depends on your PDF processing library
            pass
        elif metadata.file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            # Handle Word documents
            # Implementation depends on your Word processing library
            pass
        elif metadata.file_type == "text/plain":
            return content.decode('utf-8')
            
        return None
        
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its metadata."""
        metadata = await self.get_metadata(doc_id)
        if not metadata:
            return False
            
        # Delete document file
        doc_path = self.base_dir / f"{doc_id}{self._get_extension(metadata.file_type)}"
        if doc_path.exists():
            os.remove(doc_path)
            
        # Delete metadata
        metadata_path = self.metadata_dir / f"{doc_id}.json"
        if metadata_path.exists():
            os.remove(metadata_path)
            
        return True 