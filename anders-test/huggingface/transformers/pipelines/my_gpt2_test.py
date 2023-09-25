import os
from transformers import pipeline


pipe = pipeline(model="gpt2", device=0)
print(pipe.model.name_or_path)
print(pipe.task)

a = pipe("What's your name?")
print(a[0]["generated_text"])

a = pipe("How to get a job?")
print(a[0]["generated_text"])