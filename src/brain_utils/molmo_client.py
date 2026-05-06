"""Thin OpenAI-compatible client for a local Molmo2-ER vLLM server.

The vLLM OpenAI server (`vllm serve allenai/Molmo2-ER ...`) exposes a chat
completions endpoint. We point an `openai.AsyncOpenAI` at it and use vLLM's
`guided_json` extension to enforce a Pydantic schema, so the call sites can
keep parsing into the same Pydantic models they used with Gemini's
`response_schema`.

Env vars:
- MOLMO_BASE_URL  default http://localhost:8000/v1
- MOLMO_MODEL     default allenai/Molmo2-ER
- MOLMO_API_KEY   default EMPTY  (vLLM ignores it but the client requires one)
"""

from __future__ import annotations

import base64
import os
from typing import Optional, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class MolmoClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv(
            "MOLMO_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.getenv("MOLMO_API_KEY", "EMPTY")
        self.model = model or os.getenv("MOLMO_MODEL", "allenai/Molmo2-ER")
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        print(
            f"MolmoClient: base_url={self.base_url} model={self.model}"
        )

    @staticmethod
    def _jpeg_to_data_url(jpeg_bytes: bytes) -> str:
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    async def select(
        self,
        prompt: str,
        jpeg_bytes_list: list[bytes],
        schema: Type[T],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Optional[T]:
        """Call Molmo with a text prompt + N images, enforce `schema` via guided JSON.

        Returns a parsed instance of `schema`, or None on parse / API failure.
        """
        content: list[dict] = [{"type": "text", "text": prompt}]
        for jpg in jpeg_bytes_list:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._jpeg_to_data_url(jpg)},
                }
            )

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"guided_json": schema.model_json_schema()},
            )
        except Exception as e:
            print(f"MolmoClient: request failed: {e}")
            return None

        try:
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"MolmoClient: malformed response: {e}")
            return None

        try:
            return schema.model_validate_json(text)
        except ValidationError as e:
            print(f"MolmoClient: schema validation failed: {e}; raw={text!r}")
            return None
        except Exception as e:
            print(f"MolmoClient: unexpected parse error: {e}; raw={text!r}")
            return None
