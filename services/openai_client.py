import os
from openai import OpenAI

_client = None

def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Add it to your environment or .env file.")
        _client = OpenAI(api_key=api_key)
    return _client
