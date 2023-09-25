from transformers import AutoTokenizer,pipeline

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

classifier = pipeline(model=model_name)
print(classifier.model.name_or_path)
print(classifier.task)

pt_outputs = pt_model(**pt_batch)
print(type(pt_outputs))
print(pt_outputs.logits)

from torch import nn

pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

# 显示结果
max_values = pt_outputs.logits.argmax(dim=1)
print(max_values)
print(pt_model.config.id2label)
for i in range(max_values.shape[0]):
    v = max_values[i].item()
    label = pt_model.config.id2label[v]
    print(label)


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