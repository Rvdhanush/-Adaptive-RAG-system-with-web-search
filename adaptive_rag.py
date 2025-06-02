from transformers import pipeline
from typing import List, Dict, Any
from document_processor import DocumentProcessor
from web_search import WebSearch
from config import settings
import re

class AdaptiveRAG:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.web_search = WebSearch()
        self.llm = pipeline(
            "text2text-generation",
            model=settings.LLM_MODEL,
            max_length=512
        )
        
    def _evaluate_relevance(self, results: List[Dict[str, Any]]) -> bool:
        """Evaluate if the local search results are relevant enough."""
        if not results:
            return False
            
        # Check if we have enough high-quality results
        high_quality_results = [
            result for result in results 
            if result["score"] >= settings.SIMILARITY_THRESHOLD
        ]
        
        return len(high_quality_results) >= 2
        
    def _filter_context(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and organize context for better response generation."""
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Get query terms
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        # Filter results based on content relevance
        filtered_results = []
        for result in sorted_results:
            content = result["content"].lower()
            # Check if content contains enough query terms
            matching_terms = sum(1 for term in query_terms if term in content)
            if matching_terms >= settings.MIN_TERM_FREQUENCY:
                filtered_results.append(result)
        
        return filtered_results[:settings.MAX_LOCAL_RESULTS]
        
    def _generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate a response using the LLM based on the provided context."""
        # Filter and organize context
        filtered_context = self._filter_context(query, context)
        
        if not filtered_context:
            return "I couldn't find enough relevant information in the provided documents to answer your question accurately."
        
        # Format context for the prompt
        formatted_context = "\n\n".join([
            f"Source: {result['metadata'].get('source', 'Unknown')}\n"
            f"Content: {result['content']}\n"
            f"Relevance: {result['score']:.2f}"
            for result in filtered_context
        ])
        
        # Create prompt for the model
        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context from the documents. 
If the context doesn't contain enough information to answer the question accurately, say so.
Do not make up information or use knowledge outside the context.
If the answer is not in the documents, respond with "I cannot find this information in the provided documents."

Context from documents:
{formatted_context}

Question: {query}

Answer:"""
        
        # Generate response
        response = self.llm(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']
        return response.strip()
        
    def query(self, query: str) -> Dict[str, Any]:
        """Process a query using the adaptive RAG system."""
        # Always try local document search first
        local_results = self.document_processor.similarity_search(query)
        
        # If we have local results, use them
        if local_results:
            response = self._generate_response(query, local_results)
            return {
                "response": response,
                "sources": local_results,
                "search_type": "local"
            }
            
        # Only use web search if enabled and no local results found
        if settings.WEB_SEARCH_FALLBACK:
            web_results = self.web_search.search(query)
            response = self._generate_response(query, web_results)
            return {
                "response": response,
                "sources": web_results,
                "search_type": "web"
            }
        
        # If no results found and web search is disabled
        return {
            "response": "I couldn't find any relevant information in the provided documents to answer your question.",
            "sources": [],
            "search_type": "none"
        } 