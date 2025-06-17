# async generative agent utils
# cqz@cs.stanford.edu

# version 2025.06.16

import numpy as np
from typing import Any, Dict, List

from .llm_utils import (
    ant_prep,
    fill_prompt,
    make_output_format,
    modular_instructions,
    parse_json,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    aoai,
    aant
)

async def a_gen(messages: str | list[dict], provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL, temperature=1, max_tokens=4000) -> str:
    """Async version of gen. Generate text completion from messages.
    
    Args:
        messages: String or list of message dicts with 'role' and 'content'
        provider: LLM provider ('oai' or 'ant')
        model: Model name to use
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text response
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        if provider == 'oai':
            response = await aoai.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages # type: ignore
            )
            return response.choices[0].message.content or ""

        elif provider == 'ant':  # Anthropic // requires max_tokens
            # Process messages for Anthropic
            modified_messages, system_content = ant_prep(messages)
            
            # Create API call with or without system parameter
            kwargs = {
                "model": model,
                "temperature": temperature,
                "messages": modified_messages,
                "max_tokens": max_tokens
            }
            
            # Add system parameter only if we have system content
            if system_content is not None:
                kwargs["system"] = system_content
            
            response = await aant.messages.create(**kwargs)
            return response.content[0].text

        else:
            raise ValueError(f"Unknown provider: {provider}")

    except Exception as e:
        print(f"Error generating completion: {e}")
        raise e


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
    """Async version of mod_gen. Generate structured output from modular instructions with retries.

    Args:
        modules: List of instruction modules, see above for format
            Each module can have:
            - 'instruction': The text instruction (required)
            - 'name': Output key name (optional)
            - 'image': URL of an image to include (optional)
        provider: LLM provider ('oai' or 'ant')
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
    # Validate required fields
    for module in modules:
        if 'instruction' not in module:
            raise ValueError("Each module must have an 'instruction' field")

    async def attempt() -> tuple[Dict[str, Any], str, str]:
        prompt, image_urls = modular_instructions(modules)
        filled = fill_prompt(prompt, placeholders)
        
        # If we have images, create a message with images
        if image_urls:
            # Build content array with text first
            content = [{"type": "text", "text": filled}]
            
            # Add each image URL
            for image_url in image_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            # Create message with images
            messages = [{"role": "user", "content": content}]
            raw_response = await a_gen(messages, provider=provider, model=model, **kwargs)
        else:
            # No images, use simple text generation
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


async def a_get_embedding(text: str) -> np.ndarray:
    """Async version of get_embedding. Get text embedding vector.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as numpy array
    """
    try:
        response = await aoai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise e 