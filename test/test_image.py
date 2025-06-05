import sys
sys.path.append("../")
from genagent.llm_utils import gen, create_image_message

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "tell me which one you like more"
    }
]

messages.append(create_image_message(
    "image 1",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
))

messages.append(create_image_message(
    "image 2",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Beach%2C_Gili_Meno_Island%2C_Indonesia.jpg/2560px-Beach%2C_Gili_Meno_Island%2C_Indonesia.jpg"
))

print("Testing with Anthropic...")
print(gen(messages, provider="ant", model="claude-4-sonnet-20250514"))
print("\nTesting with OpenAI...")
print(gen(messages, provider="oai", model="gpt-4.1"))