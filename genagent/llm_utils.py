# generative agent utils
# cqz@cs.stanford.edu

# version 2025.06.16

import json
import os
import re
from typing import Any, Dict, List

import numpy as np
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

#------------------------------------------------------------------------------
# INITIALIZATION AND CONFIGURATION
#------------------------------------------------------------------------------

load_dotenv(override=True)

# Initialize OpenAI clients
oai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
aoai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Anthropic clients
ant = Anthropic()
ant.api_key = os.getenv('ANTHROPIC_API_KEY')
aant = AsyncAnthropic()
aant.api_key = os.getenv('ANTHROPIC_API_KEY')

# Default configuration
DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'ant')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'claude-4-sonnet-20250514')

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
    
    # Process messages, keeping their original order
    for msg in messages:
        if msg["role"] == "system":
            if system_content is None:
                # Use the first system message as the system parameter
                system_content = msg["content"]
            else:
                # Convert additional system messages to user messages
                modified_messages.append({"role": "user", "content": msg["content"]})
        else:
            # Handle messages with complex content (text + images)
            if isinstance(msg.get("content"), list):
                # Convert OpenAI format to Anthropic format
                ant_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        ant_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        # Convert to Anthropic format
                        ant_content.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url
                            }
                        })
                modified_messages.append({"role": msg["role"], "content": ant_content})
            else:
                # Keep non-system messages as they are
                modified_messages.append(msg)
    
    return modified_messages, system_content


def create_image_message(text: str, image_url: str) -> dict:
    """Create a user message with text and image content.
    
    Args:
        text: Text content of the message
        image_url: URL of the image
        
    Returns:
        Formatted message dict with image content for OpenAI/Anthropic APIs
    """
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }

#------------------------------------------------------------------------------
# TEXT GENERATION
#------------------------------------------------------------------------------

def gen(messages: str | list[dict], provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL, temperature=1, max_tokens=4000) -> str:
    """Generate text completion from messages.
    
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
            response = oai.chat.completions.create(
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
            
            response = ant.messages.create(**kwargs)
            return response.content[0].text

        else:
            raise ValueError(f"Unknown provider: {provider}")

    except Exception as e:
        print(f"Error generating completion: {e}")
        raise e

#------------------------------------------------------------------------------
# MODULAR GENERATION
#------------------------------------------------------------------------------

def fill_prompt(prompt: str, placeholders: Dict) -> str:
    """Fill placeholders in a prompt template.
    
    Args:
        prompt: Template string with placeholders like !<NAME>!
        placeholders: Dict mapping placeholder names to values
        
    Returns:
        Filled prompt string
        
    Raises:
        ValueError: If any placeholders remain unfilled
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
    """Generate JSON output format string from modules.
    
    Args:
        modules: List of module dicts with optional 'name' field
        
    Returns:
        JSON format string for response
    """
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
        # Collect image URLs if present
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
    """Parse JSON from response text, handling nested structures.
    
    Args:
        response: Response text containing JSON
        target_keys: Optional list of keys to extract
        
    Returns:
        Parsed dict or None if parsing fails
    """
    # Start from the end and go backwards
    for i in range(len(response) - 1, -1, -1):
        if response[i] in ("}", "]"):
            stack = []
            j = i
            while j >= 0:
                if response[j] in ("}", "]"):
                    stack.append(response[j])
                elif response[j] == "{" and stack and stack[-1] == "}":
                    stack.pop()
                elif response[j] == "[" and stack and stack[-1] == "]":
                    stack.pop()
                if not stack:
                    candidate = response[j : i + 1]
                    # Clean escaped quotes
                    cleaned_candidate = candidate.replace('\\"', '"')
                    try:
                        parsed = json.loads(cleaned_candidate)
                        # Handle target_keys if provided
                        if target_keys:
                            if isinstance(parsed, list):
                                # If it's a list, return it under the first target key
                                return {target_keys[0]: parsed}
                            parsed = {key: parsed.get(key, "") for key in target_keys}
                        return parsed
                    except json.JSONDecodeError:
                        break
                j -= 1
    return None


def mod_gen(
    modules: List[Dict],
    provider=DEFAULT_PROVIDER,
    model=DEFAULT_MODEL,
    placeholders: Dict = {},
    target_keys = None,
    max_attempts=3,
    debug=False,
    **kwargs
) -> Dict[str, Any] | tuple[Dict[str, Any], str, str]:
    """Generate structured output from modular instructions. Supports retries.

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
        **kwargs: Additional arguments passed to gen()

    Returns:
        If debug=False: Dict of parsed responses
        If debug=True: Tuple of (parsed_dict, raw_response, filled_prompt)
    """
    # Validate required fields
    for module in modules:
        if 'instruction' not in module:
            raise ValueError("Each module must have an 'instruction' field")

    def attempt() -> tuple[Dict[str, Any], str, str]:
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
            raw_response = gen(messages, provider=provider, model=model, **kwargs)
        else:
            # No images, use simple text generation
            raw_response = gen(filled, provider=provider, model=model, **kwargs)

        if not raw_response:
            print("Error: response was empty")
            return ({}, "", filled)

        keys = ([module["name"].lower() for module in modules if "name" in module] 
                if target_keys is None else target_keys)
        parsed = parse_json(raw_response, keys)
        return (parsed or {}, raw_response, filled)

    for i in range(max_attempts):
        parsed, raw_response, filled = attempt()
        if parsed and parsed != {}:
            break
        print(f"[GEN] Retrying... ({i+1} / {max_attempts})")

    return (parsed, raw_response, filled) if debug else parsed

#------------------------------------------------------------------------------
# UTILITIES
#------------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    """Get text embedding vector.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as numpy array
    """
    try:
        response = oai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise e


def get_image(prompt: str) -> str:
    """Generate image from text prompt.
    
    Args:
        prompt: Text description of desired image
        
    Returns:
        URL of generated image
        
    Raises:
        ValueError: If image generation fails
    """
    response = oai.images.generate(
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