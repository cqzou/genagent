import sys
sys.path.append(".")
from genagent.llm_utils import mod_gen

# Example: Analyze an image and extract structured information
modules = [
    {
        "instruction": "Look at this nature image and analyze it.",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    },
    {
        "name": "location_type",
        "instruction": "What type of location is shown?"
    },
    {
        "name": "main_features",
        "instruction": "List the main features visible in the image"
    },
    {
        "name": "time_of_day",
        "instruction": "What time of day does this appear to be?"
    },
    {
        "name": "season",
        "instruction": "What season does this appear to be?"
    }
]

result = mod_gen(modules, provider='oai', model='gpt-4o-mini')
for key, value in result.items():
    print(f"{key}: {value}")

# Example with multiple images
modules = [
    {
        "instruction": "here is 'image 14326433'",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    },
    {
        "instruction": "here is 'image 333333'",
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    },
    {
        "name": "comparison",
        "instruction": "What are the main differences between these two images? "
    },
    {
        "name": "scale_difference",
        "instruction": "How do the scales of these images differ?"
    }
]

result2 = mod_gen(modules, provider='oai', model='gpt-4.1')
for key, value in result2.items():
    print(f"{key}: {value}")
