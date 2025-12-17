# async generative agent utils
# cqz@cs.stanford.edu

# version 2025.11.13

import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from anthropic import AsyncAnthropic
from cerebras.cloud.sdk import Cerebras
from openai import AsyncOpenAI
from pydantic import BaseModel

from .llm_utils import (
    ant_prep,
    fill_prompt,
    make_output_format,
    modular_instructions,
    parse_json,
    stable_object_hash,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    ProviderType,
    CACHE_DIR,
)


#------------------------------------------------------------------------------
# ASYNC PROVIDER ABSTRACTION
#------------------------------------------------------------------------------

class AsyncLLMProvider(ABC):
    """Abstract base class for async LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[BaseModel],
        return_raw_response: bool,
        return_response_obj: bool,
        tools: list,
        **kwargs,
    ) -> Any:
        """Generate a completion from the LLM."""
        pass

    def _parse_json_response(
        self, text_response: str, response_format: Optional[BaseModel]
    ) -> Any:
        """Parse JSON response and validate against Pydantic model if provided."""
        if response_format is None:
            return text_response

        try:
            cleaned_text = text_response.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            json_data = json.loads(cleaned_text)

            if hasattr(response_format, "model_validate"):
                return response_format.model_validate(json_data)
            return json_data
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Failed to parse response as JSON: {e}")
            return text_response

    def _add_json_format_instruction(
        self, messages: list[dict], response_format: Optional[BaseModel]
    ) -> list[dict]:
        """Add JSON format instruction to messages."""
        if response_format is None or not messages:
            return messages

        messages = messages.copy()
        if messages[-1]["role"] != "user":
            return messages

        if hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            format_instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{json.dumps(schema, indent=2)}"
        else:
            format_instruction = "\n\nPlease respond with valid JSON."

        if isinstance(messages[-1]["content"], str):
            messages[-1]["content"] += format_instruction
        elif isinstance(messages[-1]["content"], list):
            for block in messages[-1]["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] += format_instruction
                    break

        return messages


class AsyncOpenAIProvider(AsyncLLMProvider):
    """Async OpenAI provider implementation."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[BaseModel],
        return_raw_response: bool,
        return_response_obj: bool,
        tools: list,
        **kwargs,
    ) -> Any:
        # Note: OpenAI's async responses API doesn't exist yet, so we use chat.completions
        if response_format is not None:
            messages = self._add_json_format_instruction(messages, response_format)

        response = await self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
        )

        text_response = response.choices[0].message.content or ""

        if response_format is not None and not return_response_obj:
            return self._parse_json_response(text_response, response_format)

        return text_response


class AsyncAnthropicProvider(AsyncLLMProvider):
    """Async Anthropic provider implementation."""

    def __init__(self):
        self.client = AsyncAnthropic()
        self.client.api_key = os.getenv("ANTHROPIC_API_KEY")

    async def generate(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[BaseModel],
        return_raw_response: bool,
        return_response_obj: bool,
        tools: list,
        **kwargs,
    ) -> Any:
        modified_messages, system_content = ant_prep(messages)

        if response_format is not None:
            modified_messages = self._add_json_format_instruction(
                modified_messages, response_format
            )

        api_kwargs = {
            "model": model,
            "temperature": temperature,
            "messages": modified_messages,
            "max_tokens": max_tokens,
        }

        if system_content is not None:
            api_kwargs["system"] = system_content

        response = await self.client.messages.create(**api_kwargs)
        text_response = response.content[0].text

        if response_format is not None and not return_response_obj:
            return self._parse_json_response(text_response, response_format)

        return text_response


class AsyncCerebrasProvider(AsyncLLMProvider):
    """Async Cerebras provider implementation (sync client used in async context)."""

    def __init__(self):
        self.client = None

    def _get_client(self):
        """Lazy initialization of Cerebras client."""
        if self.client is None:
            self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        return self.client

    async def generate(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[BaseModel],
        return_raw_response: bool,
        return_response_obj: bool,
        tools: list,
        **kwargs,
    ) -> Any:
        if response_format is not None:
            messages = self._add_json_format_instruction(messages, response_format)

        # Cerebras SDK doesn't have async client yet, so we use sync in async context
        # This is acceptable for Cerebras since it's very fast
        client = self._get_client()
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, seed=333
        )

        text_response = response.choices[0].message.content

        if response_format is not None and not return_response_obj:
            try:
                cleaned_text = text_response.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

                json_data = json.loads(cleaned_text)

                if hasattr(response_format, "model_validate"):
                    return response_format.model_validate(json_data)
                return json_data
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Failed to parse Cerebras response as JSON: {e}")
                logging.warning(f"Raw response text: {text_response[:500]}")
                return {}

        return text_response


# Async provider registry
_async_providers = {
    ProviderType.OPENAI.value: AsyncOpenAIProvider(),
    ProviderType.ANTHROPIC.value: AsyncAnthropicProvider(),
    ProviderType.CEREBRAS.value: AsyncCerebrasProvider(),
}


def _is_cerebras_503_error(e: Exception) -> bool:
    """Check if an exception is a Cerebras 503 error due to high traffic."""
    exception_str = str(e)
    return (
        "503" in exception_str
        and ("too_many_requests_error" in exception_str.lower()
             or "queue_exceeded" in exception_str.lower())
    )


async def _a_gen(
    messages: str | list[dict],
    provider=DEFAULT_PROVIDER,
    model=DEFAULT_MODEL,
    temperature=1,
    max_tokens=4000,
    response_format=None,
    return_raw_response: bool = False,
    return_response_obj: bool = False,
    tools=[],
) -> str:
    """Internal async generation function with provider routing and error handling."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        if provider not in _async_providers:
            raise ValueError(f"Unknown provider: {provider}")

        llm_provider = _async_providers[provider]
        return await llm_provider.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            return_raw_response=return_raw_response,
            return_response_obj=return_response_obj,
            tools=tools,
        )
    except Exception as e:
        if provider == ProviderType.CEREBRAS.value and _is_cerebras_503_error(e):
            logging.warning(
                f"Cerebras returned 503 error (high traffic). Retrying once with Cerebras before fallback. Original error: {e}"
            )
            await asyncio.sleep(2)

            try:
                llm_provider = _async_providers[provider]
                return await llm_provider.generate(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    return_raw_response=return_raw_response,
                    return_response_obj=return_response_obj,
                    tools=tools,
                )
            except Exception as retry_error:
                if _is_cerebras_503_error(retry_error):
                    logging.warning(
                        f"Cerebras retry failed with 503 error. Falling back to gpt-4.1-mini. Error: {retry_error}"
                    )
                    fallback_provider = ProviderType.OPENAI.value
                    fallback_model = "gpt-4.1-mini"
                    llm_provider = _async_providers[fallback_provider]
                    return await llm_provider.generate(
                        messages=messages,
                        model=fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        return_raw_response=return_raw_response,
                        return_response_obj=return_response_obj,
                        tools=tools,
                    )
                else:
                    print(f"Error generating completion on retry: {retry_error}")
                    raise retry_error

        print(f"Error generating completion: {e}")
        raise e


async def a_gen(
    messages,
    response_format: BaseModel = None,
    use_cache: bool = False,
    **kwargs,
):
    """Async generate text completion from messages with caching support.

    Args:
        messages: String or list of message dicts with 'role' and 'content'
        response_format: Optional Pydantic model for structured output
        use_cache: Whether to use disk-based caching
        **kwargs: Additional arguments (provider, model, temperature, max_tokens, etc.)

    Returns:
        Generated text response or parsed Pydantic model
    """
    if response_format is not None:
        response_format_hsh_tuple = (response_format.__name__,)
    else:
        response_format_hsh_tuple = tuple()

    if kwargs.get("return_response_obj"):
        hsh_path = None
    else:
        hsh_kwargs = {k: v for k, v in kwargs.items()}
        hsh_kwargs["messages"] = messages
        hsh_kwargs["response_format"] = response_format_hsh_tuple
        hsh = stable_object_hash(hsh_kwargs)
        hsh_base_path = os.path.join(CACHE_DIR, hsh[0], hsh[1])
        try:
            os.makedirs(hsh_base_path, exist_ok=True)
        except OSError as e:
            logging.warning(
                f"Cannot create cache directory {hsh_base_path}: {e}. Disabling cache for this request."
            )
            hsh_path = None
        else:
            hsh_path = os.path.join(hsh_base_path, f"{hsh}.json")

    if use_cache and not kwargs.get("return_response_obj") and hsh_path:
        if os.path.exists(hsh_path):
            with open(hsh_path, "r") as f:
                out = f.read()
                if len(out) > 1:
                    try:
                        return json.loads(out)
                    except Exception as exc:
                        logging.warning(
                            f"Error loading from cache: {traceback.format_exception_only(exc)}. Will recrunch."
                        )

    prompt_preview = " ".join((messages if isinstance(messages, str) else next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")).split()[:5]) if messages else "[empty]"
    start_time = time.time()
    output = await _a_gen(
        messages=messages,
        response_format=response_format,
        **kwargs,
    )
    logging.info(f"[LLM] provider={kwargs.get('provider', DEFAULT_PROVIDER)}, model={kwargs.get('model', DEFAULT_MODEL)}, time={time.time() - start_time:.2f}s, prompt=\"{prompt_preview}...\"")

    if response_format is not None and not kwargs.get("return_response_obj"):
        if hasattr(output, "dict"):
            output = output.dict()
        elif isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                pass

    if hsh_path is not None:
        try:
            with open(hsh_path, "w") as f:
                f.write(json.dumps(output, indent=2))
        except OSError as e:
            logging.warning(f"Cannot write to cache file {hsh_path}: {e}")

    return output


#------------------------------------------------------------------------------
# ASYNC MODULAR GENERATION
#------------------------------------------------------------------------------

async def a_mod_gen(
    modules: List[Dict],
    provider=DEFAULT_PROVIDER,
    model=DEFAULT_MODEL,
    placeholders: Dict = {},
    target_keys = None,
    max_attempts=3,
    debug=False,
    **kwargs
) -> Dict[str, Any] | tuple[Dict[str, Any], str, str]:
    """Async generate structured output from modular instructions.

    Args:
        modules: List of instruction modules with 'instruction' and optional 'name', 'image'
        provider: LLM provider ('oai', 'ant', or 'cer')
        model: Model name to use
        placeholders: Dict of values to fill in prompt template
        target_keys: Keys to extract from response (defaults to module names)
        max_attempts: Number of retries on failed parsing
        debug: If True, returns (parsed, raw_response, filled_prompt)
        **kwargs: Additional arguments passed to a_gen()

    Returns:
        If debug=False: Dict of parsed responses
        If debug=True: Tuple of (parsed_dict, raw_response, filled_prompt)
    """
    for module in modules:
        if 'instruction' not in module:
            raise ValueError("Each module must have an 'instruction' field")

    async def attempt() -> tuple[Dict[str, Any], str, str]:
        prompt, image_urls = modular_instructions(modules)
        filled = fill_prompt(prompt, placeholders)

        if image_urls:
            content = [{"type": "text", "text": filled}]
            for image_url in image_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            messages = [{"role": "user", "content": content}]
            raw_response = await a_gen(messages, provider=provider, model=model, **kwargs)
        else:
            raw_response = await a_gen(filled, provider=provider, model=model, **kwargs)

        if not raw_response:
            print("Error: response was empty")
            return ({}, "", filled)

        keys = ([module["name"].lower() for module in modules if "name" in module]
                if target_keys is None else target_keys)
        parsed = parse_json(raw_response, keys)
        return (parsed or {}, raw_response, filled)

    for i in range(max_attempts):
        parsed, raw_response, filled = await attempt()
        if parsed and parsed != {}:
            break
        print(f"[GEN] Retrying... ({i+1} / {max_attempts})")

    return (parsed, raw_response, filled) if debug else parsed


#------------------------------------------------------------------------------
# ASYNC UTILITIES
#------------------------------------------------------------------------------

async def a_get_embedding(text: str) -> np.ndarray:
    """Async get text embedding vector using OpenAI.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as numpy array
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise e
