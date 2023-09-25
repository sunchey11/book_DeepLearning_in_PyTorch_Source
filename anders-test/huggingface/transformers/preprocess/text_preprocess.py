from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)
in_param = encoded_input["input_ids"]
print(type(in_param))
org_str = tokenizer.decode(in_param)
print(org_str)