import json
import logging
import os
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

class MemoryCategories(BaseModel):
    categories: List[str]


def _get_gemini_client():
    """Initialize Gemini client if API key is available."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-1.5-flash')
        return None
    except ImportError:
        logging.info("Google Generative AI library not installed. Categorization will use OpenAI if available.")
        return None
    except Exception as e:
        logging.warning(f"Failed to initialize Gemini client: {e}")
        return None


def _get_openai_client():
    """Initialize OpenAI client if API key is available."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        return None
    except ImportError:
        logging.warning("OpenAI library not installed. Install with: pip install openai")
        return None


def _categorize_with_gemini(gemini_client, memory: str) -> List[str]:
    """Categorize memory using Gemini."""
    prompt = f"{MEMORY_CATEGORIZATION_PROMPT}\n\nMemory to categorize: {memory}\n\nPlease respond with a JSON object in this format: {{\"categories\": [\"category1\", \"category2\"]}}"
    
    response = gemini_client.generate_content(prompt)
    response_text = response.text.strip()
    
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            parsed_response = json.loads(json_str)
            if 'categories' in parsed_response:
                return [cat.strip().lower() for cat in parsed_response['categories']]
    except json.JSONDecodeError:
        pass
    
    logging.warning(f"Could not parse JSON response from Gemini: {response_text}")
    return ["general"]


def _categorize_with_openai(openai_client, memory: str) -> List[str]:
    """Categorize memory using OpenAI."""
    messages = [
        {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
        {"role": "user", "content": memory}
    ]

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=MemoryCategories,
        temperature=0
    )

    parsed: MemoryCategories = completion.choices[0].message.parsed
    return [cat.strip().lower() for cat in parsed.categories]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    """
    Get categories for memory using available AI provider.
    Tries Gemini first, then falls back to OpenAI if available.
    """
    try:
        gemini_client = _get_gemini_client()
        if gemini_client:
            logging.info("Using Gemini for memory categorization")
            return _categorize_with_gemini(gemini_client, memory)
        
        openai_client = _get_openai_client()
        if openai_client:
            logging.info("Using OpenAI for memory categorization")
            return _categorize_with_openai(openai_client, memory)
        
        logging.warning("No API keys found for Gemini or OpenAI. Returning default category.")
        return ["general"]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        return ["general"]
