import os
import asyncio
import numpy as np
from dotenv import load_dotenv

from genagent.async_utils import a_gen, a_mod_gen, a_get_embedding

# Load environment variables
load_dotenv(override=True)

# Test data
SIMPLE_PROMPT = "What is 2+2?"
COMPLEX_PROMPT = "Explain quantum computing in simple terms."

TEST_MODULES = [
    {
        "instruction": "Analyze the following text and extract key points: !<TEXT>!",
        "name": "analysis"
    },
    {
        "instruction": "Based on the analysis, provide recommendations.",
        "name": "recommendations"
    }
]

async def test_a_gen_openai():
    """Test async generation with OpenAI"""
    response = await a_gen(SIMPLE_PROMPT, provider='oai', model='gpt-4.1')
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ test_a_gen_openai passed")

async def test_a_gen_anthropic():
    """Test async generation with Anthropic"""
    response = await a_gen(SIMPLE_PROMPT, provider='ant', model='claude-4-sonnet-20250514')
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ test_a_gen_anthropic passed")

async def test_a_gen_with_messages():
    """Test generation with a list of messages"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": COMPLEX_PROMPT}
    ]
    response = await a_gen(messages, provider='oai', model='gpt-4.1')
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ test_a_gen_with_messages passed")

async def test_a_mod_gen_openai():
    """Test modular generation with OpenAI"""
    placeholders = {"TEXT": "The quick brown fox jumps over the lazy dog."}
    result = await a_mod_gen(
        TEST_MODULES,
        provider='oai',
        model='gpt-4.1',
        placeholders=placeholders
    )
    assert isinstance(result, dict)
    assert "analysis" in result
    assert "recommendations" in result
    print("✓ test_a_mod_gen_openai passed")

async def test_a_mod_gen_anthropic():
    """Test modular generation with Anthropic"""
    placeholders = {"TEXT": "The quick brown fox jumps over the lazy dog."}
    result = await a_mod_gen(
        TEST_MODULES,
        provider='ant',
        model='claude-4-sonnet-20250514',
        placeholders=placeholders
    )
    assert isinstance(result, dict)
    assert "analysis" in result
    assert "recommendations" in result
    print("✓ test_a_mod_gen_anthropic passed")

async def test_a_mod_gen_with_placeholders():
    """Test modular generation with placeholders"""
    modules = [
        {
            "instruction": "Generate a story about !<ANIMAL>! and !<PLACE>!",
            "name": "story"
        }
    ]
    placeholders = {
        "ANIMAL": "dragon",
        "PLACE": "castle"
    }
    result = await a_mod_gen(
        modules,
        provider='oai',
        model='gpt-4.1',
        placeholders=placeholders
    )
    assert isinstance(result, dict)
    assert "story" in result
    assert "dragon" in result["story"].lower()
    assert "castle" in result["story"].lower()
    print("✓ test_a_mod_gen_with_placeholders passed")

async def test_a_mod_gen_debug_mode():
    """Test modular generation in debug mode"""
    placeholders = {"TEXT": "The quick brown fox jumps over the lazy dog."}
    result = await a_mod_gen(
        TEST_MODULES,
        provider='oai',
        model='gpt-4.1',
        placeholders=placeholders,
        debug=True
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    parsed, raw_response, filled_prompt = result
    assert isinstance(parsed, dict)
    assert isinstance(raw_response, str)
    assert isinstance(filled_prompt, str)
    print("✓ test_a_mod_gen_debug_mode passed")

async def test_a_get_embedding():
    """Test async embedding generation"""
    text = "This is a test sentence."
    embedding = await a_get_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0
    print("✓ test_a_get_embedding passed")

async def test_a_gen_invalid_provider():
    """Test error handling for invalid provider"""
    try:
        await a_gen(SIMPLE_PROMPT, provider='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown provider" in str(e)
        print("✓ test_a_gen_invalid_provider passed")

async def test_a_mod_gen_invalid_modules():
    """Test error handling for invalid modules"""
    invalid_modules = [
        {"name": "test"}  # Missing required 'instruction' field
    ]
    try:
        await a_mod_gen(invalid_modules, provider='oai', model='gpt-4.1')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have an 'instruction' field" in str(e)
        print("✓ test_a_mod_gen_invalid_modules passed")

async def test_a_mod_gen_retry_logic():
    """Test retry logic in modular generation"""
    # Create a module that will likely fail parsing
    modules = [
        {
            "instruction": "Return a number:",
            "name": "number"
        }
    ]
    result = await a_mod_gen(
        modules,
        provider='oai',
        model='gpt-4.1',
        max_attempts=2
    )
    assert isinstance(result, dict)
    print("✓ test_a_mod_gen_retry_logic passed")

async def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_a_gen_openai,
        test_a_gen_anthropic,
        test_a_gen_with_messages,
        test_a_mod_gen_openai,
        test_a_mod_gen_anthropic,
        test_a_mod_gen_with_placeholders,
        test_a_mod_gen_debug_mode,
        test_a_get_embedding,
        test_a_gen_invalid_provider,
        test_a_mod_gen_invalid_modules,
        test_a_mod_gen_retry_logic
    ]
    
    print("\nRunning async tests...")
    for test in tests:
        try:
            await test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            raise e
    print("\nAll tests passed")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
