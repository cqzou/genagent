# generative agent utils
# cqz@cs.stanford.edu

# version 2025.02.24

# TODO:
# [] clean up memory and chatting
# [] support claude 3.7 reasoning
# [] fix linting fhdsklfa;djslgahdsl
# [] cleaner model toggle with settings per project
# [] fix module making - more intuitive in/out

import json
import os
import re
import uuid
from typing import Any, Dict, List

import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

#------------------------------------------------------------------------------
# INITIALIZATION AND CONFIGURATION
#------------------------------------------------------------------------------

load_dotenv()
oai = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

ant = Anthropic()
ant.api_key = os.getenv('ANTHROPIC_API_KEY')

DEFAULT_PROVIDER = 'ant'
DEFAULT_MODEL = 'claude-3-7-sonnet-latest'

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
      # print(module)
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
    provider='oai',
    model='gpt-4o',
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


# agent utils v1

class MemoryNode:
  def __init__(self, content: str):
    self.id = uuid.uuid4()
    self.content = content
    self.embedding = get_embedding(content)
    self.timestamp = 0
    self.importance = 0 # TODO importance function
    
  def to_dict(self):
    """Convert MemoryNode to a dictionary for serialization"""
    return {
      'id': str(self.id),
      'content': self.content,
      'embedding': self.embedding.tolist(),
      'timestamp': self.timestamp,
      'importance': self.importance
    }
  
  @classmethod
  def from_dict(cls, data):
    """Create a MemoryNode from a dictionary"""
    node = cls(data['content'])
    node.id = uuid.UUID(data['id'])
    node.embedding = np.array(data['embedding'])
    node.timestamp = data['timestamp']
    node.importance = data['importance']
    return node


class Agent:
  
  class MemoryStream:
    def __init__(self):
      self.memories: List[MemoryNode] = []
      self.memory_index: Dict[uuid.UUID, MemoryNode] = {}
      self.embedding_matrix: np.ndarray = np.empty((0, 1536))  # ada-002 embeddings are 1536-dim
      
    def add_memory(self, content: str) -> MemoryNode:
      node = MemoryNode(content)
      self.memories.append(node)
      self.memory_index[node.id] = node
      
      if self.embedding_matrix.size == 0:
        self.embedding_matrix = node.embedding.reshape(1, -1)
      else:
        self.embedding_matrix = np.vstack([self.embedding_matrix, node.embedding])
      
      return node

    def retrieve_memories(self, 
        query: str,
        top_k: int = 5,
        weights: dict = {
          "relevance": 1.0,
          "recency": 1.0, 
          "importance": 1.0
        }
       ) -> List[MemoryNode]:
      
      if not self.memories:
        return []

      query_embedding = get_embedding(query)
      
      # Calculate relevance scores using dot product
      relevance_scores = np.dot(self.embedding_matrix, query_embedding)
      
      # Calculate recency scores
      max_timestamp = max((m.timestamp for m in self.memories), default=1) or 1
      recency_scores = np.array([m.timestamp / max_timestamp for m in self.memories])
        
      # Get importance scores
      importance_scores = np.array([m.importance for m in self.memories])
      
      # Calculate total scores with weights
      total_scores = (
        weights["relevance"] * relevance_scores +
        weights["recency"] * recency_scores +
        weights["importance"] * importance_scores
      )
      
      # Get top memories
      top_indices = np.argsort(total_scores)[-top_k:][::-1]
      return [self.memories[i] for i in top_indices]

    def to_text(self, memories: List[MemoryNode], separator: str = "\n\n") -> str:
      """Convert a list of memory nodes to a single text block"""
      return separator.join([memory.content for memory in memories])

    def to_dict(self):
      """Convert MemoryStream to a dictionary for serialization"""
      return {
        'memories': [memory.to_dict() for memory in self.memories]
      }
    
    @classmethod
    def from_dict(cls, data):
      """Create a MemoryStream from a dictionary"""
      stream = cls()
      for memory_data in data['memories']:
        node = MemoryNode.from_dict(memory_data)
        stream.memories.append(node)
        stream.memory_index[node.id] = node
      
      if stream.memories:
        stream.embedding_matrix = np.vstack([m.embedding.reshape(1, -1) for m in stream.memories])
      
      return stream

  
  def __init__(self, name: str):
    self.name = name
    self.memory_stream = self.MemoryStream()
    
  def add_memory(self, content: str) -> MemoryNode:
    return self.memory_stream.add_memory(content)
    
  def retrieve_memories(self, 
      query: str,
      top_k: int = 5,
      weights: dict = {
        "relevance": 1.0,
        "recency": 1.0, 
        "importance": 1.0
      }
     ) -> List[MemoryNode]:
    return self.memory_stream.retrieve_memories(query, top_k, weights)
  
  def retrieve_memories_as_text(self,
      query: str,
      top_k: int = 5,
      weights: dict = {
        "relevance": 1.0,
        "recency": 1.0, 
        "importance": 1.0
      },
      separator: str = "\n\n"
     ) -> str:
    """Retrieve memories and return them as a formatted text block"""
    memories = self.retrieve_memories(query, top_k, weights)
    return self.memory_stream.to_text(memories, separator)

  def simple_ask(self, system_prompt: str, query: str, provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL, temperature=1) -> str:
    prompt = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": query}
    ]
    return gen(prompt, provider=provider, model=model, temperature=temperature)

  def to_dict(self):
    """Convert Agent to a dictionary for serialization"""
    return {
      'name': self.name,
      'memory_stream': self.memory_stream.to_dict()
    }
  
  @classmethod
  def from_dict(cls, data):
    """Create an Agent from a dictionary"""
    agent = cls(data['name'])
    agent.memory_stream = cls.MemoryStream.from_dict(data['memory_stream'])
    return agent
  
  def save(self, filepath):
    """Save agent to a JSON file"""
    with open(filepath, 'w') as f:
      json.dump(self.to_dict(), f)
  
  @classmethod
  def load(cls, filepath):
    """Load agent from a JSON file"""
    with open(filepath, 'r') as f:
      data = json.load(f)
    return cls.from_dict(data)


def create_simple_agent(name: str, memory_content: str) -> Agent:
  """
  Create a simple agent with an optional initial memory.
  
  Args:
      name: The name of the agent
      memory_content: Optional content to store as the agent's initial memory
      
  Returns:
      An Agent instance
  """
  agent = Agent(name)
  if memory_content:
    agent.add_memory(memory_content)
  return agent


class ChatSession:
  """
  Simple chat session that manages messages (list of dicts) between user and agent.
  """
  
  def __init__(self, agent: Agent, system_prompt: str = ""):
    self.agent = agent
    self.messages = []
    
    # Initialize with system message if provided
    if system_prompt:
      self.messages.append({"role": "system", "content": system_prompt})
  
  def add_user_message(self, content: str):
    """Add a user message to the conversation"""
    message = {"role": "user", "content": content}
    self.messages.append(message)
    return message
  
  def add_agent_message(self, content: str):
    """Add an agent message to the conversation"""
    message = {"role": "assistant", "content": content}
    self.messages.append(message)
    return message
  
  def get_response(self, provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL, temperature=1) -> str:
    """Generate a response from the agent based on conversation history"""
    response = gen(self.messages, provider=provider, model=model, temperature=temperature)
    self.add_agent_message(response)
    return response
  
  def chat(self, user_message: str, provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL, temperature=1) -> str:
    """Add user message and get agent response"""
    self.add_user_message(user_message)
    return self.get_response(provider=provider, model=model, temperature=temperature)
  
  def to_dict(self):
    """Convert ChatSession to a dictionary for serialization"""
    return {
      'agent': self.agent.to_dict(),
      'messages': self.messages
    }
  
  @classmethod
  def from_dict(cls, data):
    """Create a ChatSession from a dictionary"""
    agent = Agent.from_dict(data['agent'])
    session = cls(agent)
    session.messages = data['messages']
    return session
  
  def save(self, filepath):
    """Save chat session to a JSON file"""
    with open(filepath, 'w') as f:
      json.dump(self.to_dict(), f)
  
  @classmethod
  def load(cls, filepath):
    """Load chat session from a JSON file"""
    with open(filepath, 'r') as f:
      data = json.load(f)
    return cls.from_dict(data)


def create_simple_chat(agent_name: str, system_prompt: str = "") -> ChatSession:
  """
  Create a simple chat session with a new agent.
  
  Args:
      agent_name: The name of the agent
      system_prompt: Optional system prompt to initialize the chat
      
  Returns:
      A ChatSession instance with a new agent
  """
  agent = Agent(agent_name)
  return ChatSession(agent, system_prompt)

