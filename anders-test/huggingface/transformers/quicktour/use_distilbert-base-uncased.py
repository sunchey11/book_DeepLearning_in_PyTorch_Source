# use_pipeline.py这个程序中的任务，使用Model来做

import torch
from transformers import AutoTokenizer,DistilBertTokenizer, DistilBertForSequenceClassification
def showParentClass(clazz):
    """显示父类"""
    indent = ''
    p = clazz
    print(clazz)
    while hasattr(p, '__base__'):
        p = p.__base__
        indent += '  '
        print(indent+'|__'+ str(p))

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer_x = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer_x和tokenizer虽然不一样，但是都能用
showParentClass(type(tokenizer)) #class 'transformers.models.distilbert.tokenization_distilbert.DistilBertTokenizer'
model_x = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model 和model_x的类是一样的，但是config.id2label属性不一样
showParentClass(type(model))
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
print(type(inputs)) #<class 'transformers.tokenization_utils_base.BatchEncoding'>
with torch.no_grad():
    model_result = model(**inputs)
    logits = model_result.logits
print(type(logits))
print(logits.shape)

predicted_class_id = logits.argmax().item()
print(predicted_class_id)
print(model.config.id2label)
label = model.config.id2label[predicted_class_id]
print(label)