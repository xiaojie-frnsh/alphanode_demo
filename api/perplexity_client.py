import httpx
import logging
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PerplexityClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
    
    async def get_answer(self, question: str, context: str) -> Optional[str]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": "llama-3.1-sonar-large-128k-online",  # or another model of your choice
                "messages": [
                    {
                        "role": "system",
                        "content": f"Use this context to answer the question: {context}"
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"Perplexity API error: {str(e)}")
            return None