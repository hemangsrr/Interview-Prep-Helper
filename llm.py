from __future__ import annotations
from typing import Generator, Optional
from services.openai_client import get_openai_client


class LLM:
    """
    Provider-agnostic LLM wrapper. Swap implementation details here to change providers.
    Usage:
      llm = LLM()
      text = llm.invoke(system, user)
      for chunk in llm.invoke(system, user, stream=True):
          ...
    """

    def __init__(self, model: Optional[str] = None):
        self._client = get_openai_client()
        self._model = model or "gpt-4o"

    @property
    def model(self) -> str:
        return self._model

    def invoke(self, system_prompt: str, user_prompt: str, stream: bool = False, json: bool = False):
        if not stream:
            kwargs = {}
            if json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
            return resp.choices[0].message.content
        else:
            stream_resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            def _gen() -> Generator[str, None, None]:
                for chunk in stream_resp:
                    try:
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, 'content', None)
                        if piece:
                            yield piece
                    except Exception:
                        continue
            return _gen()

    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """Return a single embedding vector for the given text."""
        m = model or "text-embedding-3-small"
        resp = self._client.embeddings.create(model=m, input=text)
        return resp.data[0].embedding
