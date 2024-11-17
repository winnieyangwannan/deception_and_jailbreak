import textgrad

from textgrad.engine_experimental.litellm import LiteLLMEngine
from litellm import completion
import httpx

import litellm
import os

litellm.set_verbose = True


####


messages = [
    {"content": "There's a llama in my garden 😱 What should I do?", "role": "user"}
]

# e.g. Call 'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct' from Serverless Inference API
response = completion(
    model="huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"content": "Hello, how are you?", "role": "user"}],
    stream=True,
)

print(response)


LiteLLMEngine(
    "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct", cache=False
).generate([{"content": "Hello, how are you?", "role": "user"}])

#############
LiteLLMEngine("gpt-4o", cache=True).generate(
    content="hello, what's 3+4", system_prompt="you are an assistant"
)

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image_data = httpx.get(image_url).content

LiteLLMEngine("gpt-4o", cache=True).generate(
    content=[image_data, "what is this my boy"], system_prompt="you are an assistant"
)
