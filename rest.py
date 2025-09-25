import os
from huggingface_hub import InferenceClient


hf = InferenceClient(model="gpt2")  # no token
output = hf.text_generation("Hello world")
print(output)
