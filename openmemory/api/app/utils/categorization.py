import json
import logging
import os
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
import litellm

load_dotenv()

class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    """
    Get categories for memory using available AI provider via litellm.
    Tries Gemini first, then falls back to OpenAI if available.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    messages = [
        {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
        {"role": "user", "content": memory}
    ]
    
    model = None
    api_key = None

    if gemini_api_key:
        model = "gemini/gemini-1.5-flash"
        api_key = gemini_api_key
        logging.info("Using Gemini for memory categorization via litellm")
    elif openai_api_key:
        model = "gpt-4o-mini"
        api_key = openai_api_key
        logging.info("Using OpenAI for memory categorization via litellm")
    
    if not model:
        logging.warning("No API keys found for Gemini or OpenAI. Returning default category.")
        return ["general"]

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # litellm returns a ModelResponse object, get content from the first choice
        response_content = response.choices[0].message.content
        
        # The response content is a stringified JSON, so we parse it
        parsed_json = json.loads(response_content)
        
        # Now we can use Pydantic to validate and parse the JSON
        parsed_categories = MemoryCategories.parse_obj(parsed_json)
        
        return [cat.strip().lower() for cat in parsed_categories.categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories using litellm: {e}")
        # Return a default category on error
        return ["general"]
