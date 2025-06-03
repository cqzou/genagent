from .llm_utils import gen, get_embedding, DEFAULT_PROVIDER, DEFAULT_MODEL
import json
import uuid
import numpy as np
from typing import List, Dict

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
