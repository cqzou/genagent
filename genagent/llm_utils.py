# generative agent utils
# cqz@cs.stanford.edu

# version 2025.11.13

import hashlib
import json
import logging
import os
import re
import time
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from anthropic import Anthropic, AsyncAnthropic
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI, RateLimitError, LengthFinishReasonError
from pydantic import BaseModel

#------------------------------------------------------------------------------
# INITIALIZATION AND CONFIGURATION
#------------------------------------------------------------------------------

load_dotenv(override=True)


class ProviderType(str, Enum):
    """Enum for LLM provider types."""
    OPENAI = "oai"
    ANTHROPIC = "ant"
    CEREBRAS = "cer"


# Default configuration
DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', ProviderType.ANTHROPIC.value)
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'claude-sonnet-4-5-20250929')

# Cache directory configuration
if os.environ.get("DYNO"):
    CACHE_DIR = "/tmp/genagent_cache"
else:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    CACHE_DIR = os.path.join(dir_path, "..", ".genagent_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_openai_client():
    """Get an OpenAI client for direct API access (e.g., embeddings)."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_async_openai_client():
    """Get an async OpenAI client for async API access."""
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#------------------------------------------------------------------------------
# MESSAGE PROCESSING
#------------------------------------------------------------------------------

def ant_prep(messages):
    """Prepare messages for Anthropic API, which doesn't support system messages.
    Uses the first system message as system param, converts other system messages to user.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Tuple of (modified_messages, system_content)
    """
    modified_messages = []
    system_content = None

    for msg in messages:
        if msg["role"] == "system":
            if system_content is None:
                system_content = msg["content"]
            else:
                modified_messages.append({"role": "user", "content": msg["content"]})
        else:
            if isinstance(msg.get("content"), list):
                ant_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        ant_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        ant_content.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url
                            }
                        })
                modified_messages.append({"role": msg["role"], "content": ant_content})
            else:
                modified_messages.append(msg)

    return modified_messages, system_content


def create_image_message(text: str, image_url: str) -> dict:
    """Create a user message with text and image content.

    Args:
        text: Text content of the message
        image_url: URL of the image

    Returns:
        Formatted message dict with image content
    """
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }


#------------------------------------------------------------------------------
# PROVIDER ABSTRACTION
#------------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
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


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
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
        reasoning_effort = kwargs.get("reasoning_effort", "medium")
        previous_response_id = kwargs.get("previous_response_id")

        for _ in range(10):
            try:
                if response_format is not None:
                    response = self.client.responses.parse(
                        model=model,
                        temperature=temperature,
                        input=messages,
                        text_format=response_format,
                        previous_response_id=previous_response_id,
                        **(
                            {"reasoning_effort": reasoning_effort}
                            if ("o4" in model or "o3" in model)
                            else {}
                        ),
                        tools=tools,
                    )
                    return response if return_response_obj else response.output_parsed
                elif return_raw_response:
                    return self.client.chat.completions.with_raw_response.create(
                        model=model,
                        temperature=temperature,
                        messages=messages,
                        seed=229,
                    )
                else:
                    response = self.client.responses.create(
                        service_tier="priority",
                        model=model,
                        temperature=temperature,
                        input=messages,
                        previous_response_id=previous_response_id,
                        tools=tools,
                    )
                    return response if return_response_obj else response.output_text
            except LengthFinishReasonError as e:
                if temperature > 0:
                    time.sleep(5)
                else:
                    logging.warning(
                        f"Raising LengthFinishReasonError as temperature == 0."
                    )
                    raise e
            except RateLimitError as e:
                traceback.print_exc()
                time.sleep(5)

        raise ValueError("Ratelimits failed after 10 attempts")


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""

    STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

    def __init__(self):
        self.client = Anthropic()
        self.client.api_key = os.getenv("ANTHROPIC_API_KEY")

    def generate(
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

        api_kwargs = {
            "model": model,
            "temperature": temperature,
            "messages": modified_messages,
            "max_tokens": max_tokens,
        }

        if system_content is not None:
            api_kwargs["system"] = system_content

        # Use structured outputs with Pydantic models
        if response_format is not None:
            response = self.client.beta.messages.parse(
                betas=[self.STRUCTURED_OUTPUTS_BETA],
                output_format=response_format,
                **api_kwargs,
            )
            if return_response_obj:
                return response
            # parsed_output is the validated Pydantic model instance
            return response.parsed_output

        response = self.client.messages.create(**api_kwargs)
        return response.content[0].text


class CerebrasProvider(LLMProvider):
    """Cerebras provider implementation."""

    def __init__(self):
        self.client = None

    def _get_client(self):
        """Lazy initialization of Cerebras client."""
        if self.client is None:
            self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        return self.client

    def generate(
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


# Provider registry
_providers = {
    ProviderType.OPENAI.value: OpenAIProvider(),
    ProviderType.ANTHROPIC.value: AnthropicProvider(),
    ProviderType.CEREBRAS.value: CerebrasProvider(),
}


def _is_cerebras_503_error(e: Exception) -> bool:
    """Check if an exception is a Cerebras 503 error due to high traffic."""
    exception_str = str(e)
    return (
        "503" in exception_str
        and ("too_many_requests_error" in exception_str.lower()
             or "queue_exceeded" in exception_str.lower())
    )


def _gen(
    messages: str | list[dict],
    provider=DEFAULT_PROVIDER,
    model=DEFAULT_MODEL,
    temperature=1,
    max_tokens=4000,
    response_format=None,
    return_raw_response: bool = False,
    reasoning_effort: str = "medium",
    previous_response_id: str | None = None,
    return_response_obj: bool = False,
    tools=[],
) -> str:
    """Internal generation function with provider routing and error handling."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        if provider not in _providers:
            raise ValueError(f"Unknown provider: {provider}")

        llm_provider = _providers[provider]
        return llm_provider.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            return_raw_response=return_raw_response,
            return_response_obj=return_response_obj,
            tools=tools,
            reasoning_effort=reasoning_effort,
            previous_response_id=previous_response_id,
        )
    except Exception as e:
        if provider == ProviderType.CEREBRAS.value and _is_cerebras_503_error(e):
            logging.warning(
                f"Cerebras returned 503 error (high traffic). Retrying once with Cerebras before fallback. Original error: {e}"
            )
            time.sleep(2)

            try:
                llm_provider = _providers[provider]
                return llm_provider.generate(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    return_raw_response=return_raw_response,
                    return_response_obj=return_response_obj,
                    tools=tools,
                    reasoning_effort=reasoning_effort,
                    previous_response_id=previous_response_id,
                )
            except Exception as retry_error:
                if _is_cerebras_503_error(retry_error):
                    logging.warning(
                        f"Cerebras retry failed with 503 error. Falling back to gpt-4.1-mini. Error: {retry_error}"
                    )
                    fallback_provider = ProviderType.OPENAI.value
                    fallback_model = "gpt-4.1-mini"
                    llm_provider = _providers[fallback_provider]
                    return llm_provider.generate(
                        messages=messages,
                        model=fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        return_raw_response=return_raw_response,
                        return_response_obj=return_response_obj,
                        tools=tools,
                        reasoning_effort=reasoning_effort,
                        previous_response_id=previous_response_id,
                    )
                else:
                    print(f"Error generating completion on retry: {retry_error}")
                    raise retry_error

        print(f"Error generating completion: {e}")
        raise e


def stable_object_hash(obj) -> str:
    """Generate stable hash for caching."""
    obj_str = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(obj_str.encode("utf-8")).hexdigest()


def gen(
    messages,
    response_format: BaseModel = None,
    use_cache: bool = False,
    **kwargs,
):
    """Generate text completion from messages with caching support.

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
    output = _gen(
        messages=messages,
        response_format=response_format,
        **kwargs,
    )
    logging.info(f"[LLM] provider={kwargs.get('provider', DEFAULT_PROVIDER)}, model={kwargs.get('model', DEFAULT_MODEL)}, time={time.time() - start_time:.2f}s, prompt=\"{prompt_preview}...\"")

    if response_format is not None and not kwargs.get("return_response_obj"):
        # Handle Pydantic models (v2 uses model_dump, v1 uses dict)
        if hasattr(output, "model_dump"):
            output = output.model_dump()
        elif hasattr(output, "dict"):
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
# MODULAR GENERATION (Legacy mod_gen support)
#------------------------------------------------------------------------------

def fill_prompt(prompt: str, placeholders: Dict) -> str:
    """Fill placeholders in a prompt template.

    Args:
        prompt: Template string with placeholders like !<NAME>!
        placeholders: Dict mapping placeholder names to values

    Returns:
        Filled prompt string
    """
    for placeholder, value in placeholders.items():
        placeholder_tag = f"!<{placeholder.upper()}>!"
        if placeholder_tag in prompt:
            prompt = prompt.replace(placeholder_tag, str(value))

    unfilled = re.findall(r'!<[^>]+>!', prompt)
    if unfilled:
        raise ValueError(f"Placeholders not filled: {', '.join(unfilled)}")

    return prompt


def make_output_format(modules: List[Dict]) -> str:
    """Generate JSON output format string from modules."""
    output_format = "Response format:\n{\n"
    for module in modules:
        if 'name' in module and module['name']:
            output_format += f'    "{module["name"].lower()}": "...",\n'
    output_format = output_format.rstrip(',\n') + "\n}"
    return output_format


def modular_instructions(modules: List[Dict]) -> tuple[str, List[str]]:
    """Generate a prompt from instruction modules.

    Args:
        modules: List of dicts with:
            - 'instruction': The text instruction (required)
            - 'name': Output key name (optional)
            - 'image': URL of an image to include (optional)

    Returns:
        Tuple of (prompt, image_urls)
    """
    prompt = ""
    step_count = 0
    image_urls = []

    for module in modules:
        if 'image' in module:
            image_urls.append(module['image'])

        if 'name' in module:
            step_count += 1
            prompt += f"Step {step_count} ({module['name']}): {module['instruction']}\n"
        else:
            prompt += f"{module['instruction']}\n"

    prompt += "\n"
    prompt += make_output_format(modules)
    return prompt, image_urls


def parse_json(response: str, target_keys: List[str] = None) -> Dict[str, Any] | None:
    """Parse JSON from response text, handling nested structures."""
    json_start = response.find('{')
    if json_start == -1:
        json_start = response.find('[')
    json_end = response.rfind('}') + 1
    if json_start == -1:
        json_end = response.rfind(']') + 1

    cleaned_response = response[json_start:json_end].replace('\\"', '"')
    try:
        parsed = json.loads(cleaned_response)
        if target_keys:
            if isinstance(parsed, list):
                return {target_keys[0]: parsed}
            parsed = {key: parsed.get(key, "") for key in target_keys}
        return parsed

    except json.JSONDecodeError:
        return None


def mod_gen(
    modules: List[Dict],
    provider=DEFAULT_PROVIDER,
    model=DEFAULT_MODEL,
    placeholders: Dict = {},
    target_keys = None,
    max_attempts=3,
    debug=False,
    response_model: Optional[type[BaseModel]] = None,
    **kwargs
) -> Dict[str, Any] | BaseModel | tuple[Dict[str, Any] | BaseModel, str, str]:
    """Generate structured output from modular instructions.

    Args:
        modules: List of instruction modules with 'instruction' and optional 'name', 'image'
        provider: LLM provider ('oai', 'ant', or 'cer')
        model: Model name to use
        placeholders: Dict of values to fill in prompt template
        target_keys: Keys to extract from response (defaults to module names)
        max_attempts: Number of retries on failed parsing (only used without response_model)
        debug: If True, returns (parsed, raw_response, filled_prompt)
        response_model: Optional Pydantic model for structured outputs. When provided,
            uses native structured outputs API instead of text-based JSON parsing.
        **kwargs: Additional arguments passed to gen()

    Returns:
        If debug=False: Dict of parsed responses or Pydantic model instance
        If debug=True: Tuple of (parsed_dict/model, raw_response, filled_prompt)
    """
    for module in modules:
        if 'instruction' not in module:
            raise ValueError("Each module must have an 'instruction' field")

    # Build prompt from modules (without JSON format instruction if using response_model)
    prompt = ""
    step_count = 0
    image_urls = []

    for module in modules:
        if 'image' in module:
            image_urls.append(module['image'])

        if 'name' in module:
            step_count += 1
            prompt += f"Step {step_count} ({module['name']}): {module['instruction']}\n"
        else:
            prompt += f"{module['instruction']}\n"

    # Only add JSON format instruction if not using response_model
    if response_model is None:
        prompt += "\n" + make_output_format(modules)

    filled = fill_prompt(prompt, placeholders)

    # Build messages with optional images
    if image_urls:
        content = [{"type": "text", "text": filled}]
        for image_url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": filled}]

    # Use structured outputs with Pydantic model
    if response_model is not None:
        result = gen(
            messages,
            provider=provider,
            model=model,
            response_format=response_model,
            **kwargs
        )
        if debug:
            return (result, str(result), filled)
        return result

    # Legacy text-based JSON parsing with retries
    def attempt() -> tuple[Dict[str, Any], str, str]:
        raw_response = gen(messages, provider=provider, model=model, **kwargs)

        if not raw_response:
            logging.error("Error: response was empty")
            return ({}, "", filled)

        keys = ([module["name"].lower() for module in modules if "name" in module]
                if target_keys is None else target_keys)
        parsed = parse_json(raw_response, keys)
        return (parsed or {}, raw_response, filled)

    for i in range(max_attempts):
        parsed, raw_response, filled = attempt()
        if parsed and parsed != {}:
            break
        logging.warning(f"[GEN] Retrying... ({i+1} / {max_attempts})")

    return (parsed, raw_response, filled) if debug else parsed


#------------------------------------------------------------------------------
# UTILITIES
#------------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    """Get text embedding vector using OpenAI."""
    client = get_openai_client()
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise e


def get_image(prompt: str) -> str:
    """Generate image from text prompt using DALL-E."""
    client = get_openai_client()
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="hd",
        n=1,
    )
    print(response.data[0].revised_prompt)

    if not response.data[0].url:
        raise ValueError("Image generation failed: No URL returned")

    return response.data[0].url
