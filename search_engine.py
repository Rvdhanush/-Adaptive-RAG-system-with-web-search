from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from document_manager import DocumentManager, DocumentMetadata
from document_processor import DocumentProcessor
import pandas as pd
import numpy as np
from collections import defaultdict
from pydantic import BaseModel

class SearchResult(BaseModel):
    """Model for search results."""
    document: DocumentMetadata
    score: float
    matched_terms: List[str]
    snippet: str
    position: int

class SearchEngine:
    def __init__(self, document_manager: DocumentManager, document_processor: DocumentProcessor):
        self.document_manager = document_manager
        self.document_processor = document_processor
        self.search_history = []
        
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        fuzzy_match: bool = False,
        min_score: float = 0.5
    ) -> List[SearchResult]:
        """Perform an advanced search across documents."""
        # Get all documents matching basic filters
        documents = await self.document_manager.list_documents(category, tags)
        
        # Apply date filters
        if date_from:
            documents = [doc for doc in documents if doc.created_at >= date_from]
        if date_to:
            documents = [doc for doc in documents if doc.created_at <= date_to]
            
        # Process query
        query_terms = self._process_query(query, fuzzy_match)
        
        # Search through documents
        results = []
        for doc in documents:
            # Get document content
            content = await self.document_manager.extract_text(doc.id)
            if not content:
                continue
                
            # Calculate relevance score
            score, matched_terms, snippet = self._calculate_relevance(
                content,
                query_terms,
                fuzzy_match
            )
            
            if score >= min_score:
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    matched_terms=matched_terms,
                    snippet=snippet,
                    position=len(results)
                ))
                
        # Sort results by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update search history
        self._update_search_history(query, len(results))
        
        return results
        
    def _process_query(self, query: str, fuzzy_match: bool) -> List[str]:
        """Process the search query."""
        # Split into terms
        terms = re.findall(r'\w+', query.lower())
        
        if fuzzy_match:
            # Add fuzzy variations
            fuzzy_terms = []
            for term in terms:
                fuzzy_terms.extend(self._generate_fuzzy_variations(term))
            terms.extend(fuzzy_terms)
            
        return list(set(terms))  # Remove duplicates
        
    def _generate_fuzzy_variations(self, term: str) -> List[str]:
        """Generate fuzzy variations of a term."""
        variations = []
        # Add common typos
        for i in range(len(term)):
            # Swap adjacent characters
            if i < len(term) - 1:
                swapped = term[:i] + term[i+1] + term[i] + term[i+2:]
                variations.append(swapped)
            # Remove one character
            removed = term[:i] + term[i+1:]
            variations.append(removed)
            # Add one character
            for c in 'abcdefghijklmnopqrstuvwxyz':
                added = term[:i] + c + term[i:]
                variations.append(added)
                
        return variations
        
    def _calculate_relevance(
        self,
        content: str,
        query_terms: List[str],
        fuzzy_match: bool
    ) -> tuple[float, List[str], str]:
        """Calculate relevance score and generate snippet."""
        content_lower = content.lower()
        matched_terms = []
        term_positions = []
        
        # Find term matches
        for term in query_terms:
            if fuzzy_match:
                # Use fuzzy matching
                matches = self._fuzzy_find(term, content_lower)
            else:
                # Use exact matching
                matches = [m.start() for m in re.finditer(r'\b' + re.escape(term) + r'\b', content_lower)]
                
            if matches:
                matched_terms.append(term)
                term_positions.extend(matches)
                
        if not matched_terms:
            return 0.0, [], ""
            
        # Calculate base score
        base_score = len(matched_terms) / len(query_terms)
        
        # Calculate position score
        if term_positions:
            position_score = 1.0 - (min(term_positions) / len(content))
        else:
            position_score = 0.0
            
        # Calculate term density score
        term_density = len(term_positions) / len(content.split())
        density_score = min(term_density * 10, 1.0)
        
        # Combine scores
        final_score = (base_score * 0.4 + position_score * 0.3 + density_score * 0.3)
        
        # Generate snippet
        snippet = self._generate_snippet(content, term_positions)
        
        return final_score, matched_terms, snippet
        
    def _fuzzy_find(self, term: str, content: str) -> List[int]:
        """Find fuzzy matches of a term in content."""
        matches = []
        for i in range(len(content) - len(term) + 1):
            if self._levenshtein_distance(term, content[i:i+len(term)]) <= 2:
                matches.append(i)
        return matches
        
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
        
    def _generate_snippet(self, content: str, positions: List[int]) -> str:
        """Generate a text snippet around matched terms."""
        if not positions:
            return ""
            
        # Find the best window of text
        window_size = 200
        best_window_start = max(0, min(positions) - window_size // 2)
        best_window_end = min(len(content), max(positions) + window_size // 2)
        
        # Extract snippet
        snippet = content[best_window_start:best_window_end]
        
        # Add ellipsis if needed
        if best_window_start > 0:
            snippet = "..." + snippet
        if best_window_end < len(content):
            snippet = snippet + "..."
            
        return snippet
        
    def _update_search_history(self, query: str, result_count: int):
        """Update search history."""
        self.search_history.append({
            "query": query,
            "timestamp": datetime.now(),
            "result_count": result_count
        })
        
        # Keep only last 100 searches
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]
            
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics."""
        if not self.search_history:
            return {
                "total_searches": 0,
                "average_results": 0,
                "popular_terms": [],
                "search_trends": []
            }
            
        # Calculate basic statistics
        total_searches = len(self.search_history)
        average_results = sum(s["result_count"] for s in self.search_history) / total_searches
        
        # Find popular terms
        term_frequency = defaultdict(int)
        for search in self.search_history:
            terms = re.findall(r'\w+', search["query"].lower())
            for term in terms:
                term_frequency[term] += 1
                
        popular_terms = sorted(
            term_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate search trends
        df = pd.DataFrame(self.search_history)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily_searches = df.groupby("date").size().reset_index(name="count")
        
        return {
            "total_searches": total_searches,
            "average_results": average_results,
            "popular_terms": popular_terms,
            "search_trends": daily_searches.to_dict("records")
        } 