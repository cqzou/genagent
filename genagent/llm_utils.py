# generative agent utils
# cqz@cs.stanford.edu

# version 2025.06.03

import json
import os
import re
from typing import Any, Dict, List

import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

#------------------------------------------------------------------------------
# INITIALIZATION AND CONFIGURATION
#------------------------------------------------------------------------------

load_dotenv(override=True)
oai = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

ant = Anthropic()
ant.api_key = os.getenv('ANTHROPIC_API_KEY')

DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'ant')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'claude-3-7-sonnet-latest')

#------------------------------------------------------------------------------
# BASIC TEXT GENERATION
#------------------------------------------------------------------------------

def ant_prep(messages):
  '''
  Prepare messages for Anthropic API, which doesn't support system messages.
  Uses the first system message as system param, converts other system messages to user.
  '''
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
      # Keep non-system messages as they are
      modified_messages.append(msg)
  
  return modified_messages, system_content


def gen(messages: str | list[dict], provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL, temperature=1, max_tokens=4000) -> str:
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
      # print(f"Generating with {model}")
      # claude-3-7-sonnet-20250219
      # claude-3-5-sonnet-20241022
      
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


def simple_gen(prompt, provider='oai', model='o1-mini', temperature=1, max_tokens=4000) -> str:
  return gen(prompt, provider, model, temperature, max_tokens)


def fill_prompt(prompt, placeholders):
  for placeholder, value in placeholders.items():
    placeholder_tag = f"!<{placeholder.upper()}>!"
    if placeholder_tag in prompt:
      prompt = prompt.replace(placeholder_tag, str(value))
  
  unfilled = re.findall(r'!<[^>]+>!', prompt)
  if unfilled: 
    raise ValueError(f"Placeholders not filled: {', '.join(unfilled)}")
    
  return prompt


def make_output_format(modules):
  output_format = "Response format:\n{\n"
  for module in modules:
    if 'name' in module and module['name']:
      output_format += f'    "{module["name"].lower()}": "...",\n'
  output_format = output_format.rstrip(',\n') + "\n}"
  return output_format


def modular_instructions(modules):
  """
  Generate a prompt from instruction modules.
  
  Modules are a list of dicts with an 'instruction' key (req) and a 'name' key (opt).
  
  If a module has a 'name', it will be used as a requested output key (used for fields
  that you want to extract from the response).
  
  If a module does not have a 'name', it will be added to the prompt only (used for 
  showing data or instructions that don't need to be extracted).
  """
  prompt = ""
  step_count = 0
  for module in modules:
    if 'name' in module:
      step_count += 1
      prompt += f"Step {step_count} ({module['name']}): {module['instruction']}\n"
    else:
      prompt += f"{module['instruction']}\n"
  prompt += "\n"
  prompt += make_output_format(modules)
  return prompt


def parse_json(response, target_keys=None):
  json_start = response.find('{')
  if json_start == -1:  # If no object found, check for array
    json_start = response.find('[')
  json_end = response.rfind('}') + 1
  if json_start == -1:  # If still no start found
    json_end = response.rfind(']') + 1
  
  cleaned_response = response[json_start:json_end].replace('\\"', '"')
  try:
    parsed = json.loads(cleaned_response)
    if target_keys:
      if isinstance(parsed, list):
        # If it's a list, return it under the first target key
        return {target_keys[0]: parsed}
      parsed = {key: parsed.get(key, "") for key in target_keys}
    return parsed

  except json.JSONDecodeError:
    # print(f"[GEN] JSONDecodeError: {response}")
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

  def attempt() -> tuple[Dict[str, Any], str, str]:
    prompt = modular_instructions(modules)
    filled = fill_prompt(prompt, placeholders)
    raw_response = simple_gen(filled, provider=provider, model=model, **kwargs)

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


# random utils

def get_embedding(text: str) -> np.ndarray:
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