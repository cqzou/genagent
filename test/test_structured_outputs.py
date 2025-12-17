"""Tests for structured outputs with Pydantic models.

Tests the native structured outputs API for both Anthropic and OpenAI providers.
"""

import sys
sys.path.append(".")

from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from genagent import gen, mod_gen

load_dotenv(override=True)


# --- Pydantic Models for Testing ---

class ContactInfo(BaseModel):
    """Simple model for contact extraction."""
    name: str
    email: str
    company: Optional[str] = None


class AnalysisResult(BaseModel):
    """Model for text analysis results."""
    summary: str
    key_points: List[str]
    sentiment: str


class MathResult(BaseModel):
    """Model for math problem results."""
    problem: str
    answer: int
    explanation: str


class ImageAnalysis(BaseModel):
    """Model for image analysis results."""
    location_type: str
    main_features: List[str]
    time_of_day: str
    season: str


# --- Test Functions ---

def test_gen_structured_openai():
    """Test gen() with Pydantic model using OpenAI."""
    prompt = "Extract contact info: John Smith (john.smith@example.com) works at Acme Corp."

    result = gen(
        prompt,
        provider='oai',
        model='gpt-4.1',
        response_format=ContactInfo
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "name" in result, "Missing 'name' field"
    assert "email" in result, "Missing 'email' field"
    assert "john" in result["email"].lower(), f"Email mismatch: {result['email']}"
    print(f"  Result: {result}")
    print("  test_gen_structured_openai passed")


def test_gen_structured_anthropic():
    """Test gen() with Pydantic model using Anthropic."""
    prompt = "Extract contact info: Jane Doe (jane@startup.io) is the CEO of StartupIO."

    result = gen(
        prompt,
        provider='ant',
        model='claude-sonnet-4-5-20250929',
        response_format=ContactInfo
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "name" in result, "Missing 'name' field"
    assert "email" in result, "Missing 'email' field"
    assert "jane" in result["email"].lower(), f"Email mismatch: {result['email']}"
    print(f"  Result: {result}")
    print("  test_gen_structured_anthropic passed")


def test_gen_structured_complex_openai():
    """Test gen() with complex Pydantic model using OpenAI."""
    prompt = """Analyze this text:
    'The new product launch exceeded expectations with 50% more sales than projected.
    Customer feedback has been overwhelmingly positive, though some noted shipping delays.'
    """

    result = gen(
        prompt,
        provider='oai',
        model='gpt-4.1',
        response_format=AnalysisResult
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "summary" in result, "Missing 'summary' field"
    assert "key_points" in result, "Missing 'key_points' field"
    assert "sentiment" in result, "Missing 'sentiment' field"
    assert isinstance(result["key_points"], list), "key_points should be a list"
    print(f"  Result: {result}")
    print("  test_gen_structured_complex_openai passed")


def test_gen_structured_complex_anthropic():
    """Test gen() with complex Pydantic model using Anthropic."""
    prompt = """Analyze this text:
    'The quarterly report shows declining revenue for the third consecutive quarter.
    However, cost-cutting measures have improved profit margins slightly.'
    """

    result = gen(
        prompt,
        provider='ant',
        model='claude-sonnet-4-5-20250929',
        response_format=AnalysisResult
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "summary" in result, "Missing 'summary' field"
    assert "key_points" in result, "Missing 'key_points' field"
    assert "sentiment" in result, "Missing 'sentiment' field"
    assert isinstance(result["key_points"], list), "key_points should be a list"
    print(f"  Result: {result}")
    print("  test_gen_structured_complex_anthropic passed")


def test_mod_gen_with_response_model_openai():
    """Test mod_gen() with response_model parameter using OpenAI."""
    modules = [
        {"instruction": "Solve this math problem: What is 15 * 7?"},
        {"name": "problem", "instruction": "State the problem"},
        {"name": "answer", "instruction": "Provide the numerical answer"},
        {"name": "explanation", "instruction": "Explain how you solved it"}
    ]

    result = mod_gen(
        modules,
        provider='oai',
        model='gpt-4.1',
        response_model=MathResult
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "answer" in result, "Missing 'answer' field"
    assert result["answer"] == 105, f"Wrong answer: {result['answer']}"
    print(f"  Result: {result}")
    print("  test_mod_gen_with_response_model_openai passed")


def test_mod_gen_with_response_model_anthropic():
    """Test mod_gen() with response_model parameter using Anthropic."""
    modules = [
        {"instruction": "Solve this math problem: What is 12 * 8?"},
        {"name": "problem", "instruction": "State the problem"},
        {"name": "answer", "instruction": "Provide the numerical answer"},
        {"name": "explanation", "instruction": "Explain how you solved it"}
    ]

    result = mod_gen(
        modules,
        provider='ant',
        model='claude-sonnet-4-5-20250929',
        response_model=MathResult
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "answer" in result, "Missing 'answer' field"
    assert result["answer"] == 96, f"Wrong answer: {result['answer']}"
    print(f"  Result: {result}")
    print("  test_mod_gen_with_response_model_anthropic passed")


def test_mod_gen_legacy_still_works():
    """Test that legacy mod_gen without response_model still works."""
    modules = [
        {"instruction": "Answer this question about colors."},
        {"name": "primary_color", "instruction": "Name one primary color"},
        {"name": "reason", "instruction": "Why is it considered primary?"}
    ]

    result = mod_gen(
        modules,
        provider='oai',
        model='gpt-4.1'
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "primary_color" in result, "Missing 'primary_color' field"
    assert "reason" in result, "Missing 'reason' field"
    print(f"  Result: {result}")
    print("  test_mod_gen_legacy_still_works passed")


def test_mod_gen_with_placeholders_and_response_model():
    """Test mod_gen with both placeholders and response_model."""

    class StoryAnalysis(BaseModel):
        main_character: str
        setting: str
        mood: str

    modules = [
        {"instruction": "Analyze this mini-story: '!<STORY>!'"},
        {"name": "main_character", "instruction": "Who is the main character?"},
        {"name": "setting", "instruction": "Where does it take place?"},
        {"name": "mood", "instruction": "What is the overall mood?"}
    ]

    placeholders = {
        "STORY": "The old wizard stood alone in his tower, watching the sunset over the misty mountains."
    }

    result = mod_gen(
        modules,
        provider='oai',
        model='gpt-4.1',
        placeholders=placeholders,
        response_model=StoryAnalysis
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "main_character" in result, "Missing 'main_character' field"
    assert "setting" in result, "Missing 'setting' field"
    assert "mood" in result, "Missing 'mood' field"
    print(f"  Result: {result}")
    print("  test_mod_gen_with_placeholders_and_response_model passed")


def test_gen_with_messages_structured():
    """Test gen() with message list and structured output."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts contact information."},
        {"role": "user", "content": "Extract: Bob Wilson (bob@tech.com) from TechCorp"}
    ]

    result = gen(
        messages,
        provider='oai',
        model='gpt-4.1',
        response_format=ContactInfo
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "name" in result, "Missing 'name' field"
    assert "bob" in result["email"].lower(), f"Email mismatch: {result['email']}"
    print(f"  Result: {result}")
    print("  test_gen_with_messages_structured passed")


def run_all_tests():
    """Run all structured output tests."""
    tests = [
        ("OpenAI gen() structured", test_gen_structured_openai),
        ("Anthropic gen() structured", test_gen_structured_anthropic),
        ("OpenAI gen() complex model", test_gen_structured_complex_openai),
        ("Anthropic gen() complex model", test_gen_structured_complex_anthropic),
        ("OpenAI mod_gen() with response_model", test_mod_gen_with_response_model_openai),
        ("Anthropic mod_gen() with response_model", test_mod_gen_with_response_model_anthropic),
        ("Legacy mod_gen() compatibility", test_mod_gen_legacy_still_works),
        ("mod_gen() with placeholders + response_model", test_mod_gen_with_placeholders_and_response_model),
        ("gen() with messages + structured", test_gen_with_messages_structured),
    ]

    print("\n" + "=" * 60)
    print("Running Structured Outputs Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()
