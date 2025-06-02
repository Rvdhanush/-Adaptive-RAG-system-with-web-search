from tavily import TavilyClient
from typing import List, Dict, Any
from config import settings

class WebSearch:
    def __init__(self):
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        
    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Perform a web search using Tavily API."""
        if max_results is None:
            max_results = settings.MAX_WEB_RESULTS
            
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            return [
                {
                    "content": result["content"],
                    "metadata": {
                        "source": result["url"],
                        "title": result["title"]
                    },
                    "score": result.get("score", 0.0)
                }
                for result in response["results"]
            ]
        except Exception as e:
            print(f"Error performing web search: {str(e)}")
            return [] 