import os
# https://huggingface.co/gpt2?text=My+name+is+Julien+and+I+like+to
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# 我尝试将结果转为字符串，但是没有成功
print(type(output))
print(output[0])
print(output[0].shape)

org_str = tokenizer.decode(output[0][0])

print(org_str)