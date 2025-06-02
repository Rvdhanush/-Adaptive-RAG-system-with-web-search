from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
from config import settings
import os
import re
import nltk
from nltk.tokenize import sent_tokenize

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}  # Force CPU usage for now
        )
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # More semantic text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        self.persist_directory = "./chroma_db"
        self.vector_store = self._load_or_create_vector_store()
        
    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one."""
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return None
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return sent_tokenize(text)
        
    def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Process documents and store them in the vector store."""
        processed_docs = []
        
        for doc in documents:
            # Clean the text
            cleaned_text = self._clean_text(doc["content"])
            
            # Split into sentences first
            sentences = self._split_into_sentences(cleaned_text)
            
            # Process each sentence
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < settings.MIN_CONTENT_LENGTH:
                    continue
                    
                # Create document for each sentence
                chunk = {
                    "content": sentence.strip(),
                    "metadata": {
                        "source": doc.get("source", "unknown"),
                        "sentence_index": i,
                        "total_sentences": len(sentences)
                    }
                }
                processed_docs.append(chunk)
        
        if not processed_docs:
            return
            
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=processed_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Add new documents to existing vector store
            self.vector_store.add_documents(processed_docs)
            
        # Persist the vector store
        self.vector_store.persist()
        
    def _calculate_relevance_score(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate a more sophisticated relevance score."""
        base_score = result["score"]
        
        # Get the content and metadata
        content = result["content"].lower()
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        # Calculate term frequency
        term_frequency = sum(1 for term in query_terms if term in content)
        if term_frequency < settings.MIN_TERM_FREQUENCY:
            return 0.0
        
        # Calculate position score (prefer matches at the start of the content)
        position_score = 1.0
        for term in query_terms:
            if term in content:
                position = content.find(term)
                position_score *= (1 - (position / len(content)))
        
        # Calculate content quality score
        content_length = len(content)
        quality_score = min(content_length / 200, 1.0)  # Prefer longer, more informative content
        
        # Combine scores with weights
        final_score = (
            base_score * 0.4 +  # Base similarity
            (term_frequency / len(query_terms)) * 0.3 +  # Term coverage
            position_score * settings.POSITION_WEIGHT +  # Position importance
            quality_score * 0.1  # Content quality
        )
        
        return min(final_score, 1.0)  # Normalize to [0, 1]
        
    def similarity_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector store."""
        if k is None:
            k = settings.MAX_LOCAL_RESULTS
            
        if not self.vector_store:
            return []
            
        # Get initial results
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k * 3  # Get more results for filtering
        )
        
        # Process and score results
        processed_results = []
        for doc, score in results:
            result_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            # Calculate enhanced relevance score
            enhanced_score = self._calculate_relevance_score(query, result_dict)
            if enhanced_score >= settings.SIMILARITY_THRESHOLD:
                result_dict["score"] = enhanced_score
                processed_results.append(result_dict)
        
        # Sort by enhanced score and take top k
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        return processed_results[:k] 