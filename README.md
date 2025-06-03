# GenAgent Utilities

A Python package providing utilities for generative agent tasks, including:
- Interacting with LLM providers (OpenAI, Anthropic).
- Helper functions for prompt engineering and structured output.
- Basic agent memory and chat session management.

Version: 0.2.0 (See `genagent/__init__.py` for the current version)

## Installation

```bash
pip install genagent
```

## Configuration

GenAgent now supports environment variable configuration for default provider and model:

```bash
# In your .env file or environment
DEFAULT_PROVIDER=ant  # or 'oai' for OpenAI
DEFAULT_MODEL=claude-3-7-sonnet-latest  # or any model you prefer
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

```python
from genagent import gen, Agent, ChatSession, mod_gen, modular_instructions

# --- Basic text generation ---
response = gen("Tell me a concise joke.")
print(f"Joke: {response}")

# --- Agent memory example ---
my_agent = Agent(name="MyTestAgent")
my_agent.add_memory("The capital of France is Paris.")
my_agent.add_memory("The Eiffel Tower is in Paris.")
retrieved_text = my_agent.retrieve_memories_as_text(query="What is a famous landmark in the capital of France?")
print(f"Retrieved memories based on query: {retrieved_text}")

# --- Chat session example ---
# Assuming 'my_agent' is already created and has some memories
chat_session = ChatSession(agent=my_agent, system_prompt="You are a helpful assistant. Use your memory when relevant.")
user_query = "What can you tell me about Paris based on your memory?"
assistant_response = chat_session.chat(user_query)
print(f"\nChat Session:")
print(f"  User: {user_query}")
print(f"  Assistant: {assistant_response}")

# --- Structured output with mod_gen ---
# 'mod_gen' helps generate structured output based on a list of instruction modules.
# Each module in the list is a dictionary that can have:
#   - 'instruction': (Required) The text prompt or instruction for the LLM.
#   - 'name': (Optional) If provided, the LLM's output for this instruction will be captured under this key in the result.
#             Modules without a 'name' contribute to the context but aren't extracted as separate fields.

# Example modules for extracting information:
instruction_modules = [
    {
        "instruction": "You are analyzing a piece of text. The text is: 'GenAgent is a Python library for building applications with Large Language Models. It simplifies interactions with models like GPT and Claude, and provides tools for memory management.'"
    },
    {
        "name": "summary",
        "instruction": "Provide a brief one-sentence summary of the text."
    },
    {
        "name": "main_features",
        "instruction": "List up to three main features or capabilities mentioned."
    },
    {
        "name": "target_audience",
        "instruction": "Based on the description, who is the target audience?"
    }
]

print("\nAttempting structured output with mod_gen...")
# mod_gen now uses environment defaults if not specified
structured_output = mod_gen(
    modules=instruction_modules
    # provider and model will use environment defaults
)

if structured_output:
    print("\nStructured Output from mod_gen:")
    for key, value in structured_output.items():
        print(f"- {key.capitalize()}: {value}")
else:
    print("\nMod_gen did not return structured output. Check for errors or try debugging.")

## What's New in 0.2.0

- **Modular architecture**: Agent utilities moved to separate `agent_utils` module
- **Environment configuration**: Default provider and model can be set via environment variables
- **Improved defaults**: `mod_gen` now uses configured defaults instead of hardcoded values
- **Better code organization**: Cleaner separation of concerns between LLM utilities and agent functionality

## Development

To install for development (from the root of the `llm-utils` repository):

```bash
pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
