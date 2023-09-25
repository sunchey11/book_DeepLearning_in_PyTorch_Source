import os
from transformers import pipeline
def data():
    for i in range(10):
        yield f"My example {i}"


pipe = pipeline(model="gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    print("\n")
    print(out[0]["generated_text"])
    generated_characters += len(out[0]["generated_text"])
print(generated_characters)