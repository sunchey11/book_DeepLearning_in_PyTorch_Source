from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)
# 第二部分
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", 
     "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

from transformers import AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)

pt_outputs = pt_model(**pt_batch)
print(pt_outputs)

from torch import nn

pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

import os

def abs_path(name):
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    imgpath = os.path.join(file_dir, name)
    return imgpath

pt_save_directory = abs_path("./pt_save_pretrained")
tokenizer.save_pretrained(pt_save_directory)
pt_model.save_pretrained(pt_save_directory)
# now load it
pt_model2 = AutoModelForSequenceClassification.from_pretrained(pt_save_directory)
print('finished')