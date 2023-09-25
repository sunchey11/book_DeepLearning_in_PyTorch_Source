from transformers import pipeline

def showParentClass(clazz):
    """显示父类"""
    indent = ''
    p = clazz
    print(clazz)
    while hasattr(p, '__base__'):
        p = p.__base__
        indent += '  '
        print(indent+'|__'+ str(p))
# 不用pipe，用model做的代码在use_distilbert-base-uncased.py
classifier = pipeline("sentiment-analysis")
showParentClass(type(classifier))
print(classifier.model.name_or_path)
print(classifier.task)
showParentClass(type(classifier.model))

results = classifier(["We are very happy to show you the 🤗 Transformers library.", 
                      "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")