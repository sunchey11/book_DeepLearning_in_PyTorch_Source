# https://zhuanlan.zhihu.com/p/535100411
from transformers import AutoModel
checkpoint = "bert-base-chinese"

def showParentClass(clazz):
    """显示父类"""
    indent = ''
    p = clazz
    print(clazz)
    while hasattr(p, '__base__'):
        p = p.__base__
        indent += '  '
        print(indent+'|__'+ str(p))
    
model = AutoModel.from_pretrained(checkpoint)
print(type(model))  #<class 'transformers.models.bert.modeling_bert.BertModel'>
print(type(model).__base__) # <class 'transformers.models.bert.modeling_bert.BertPreTrainedModel'>
print(type(model).__base__.__base__) 
showParentClass(type(model))

from transformers import pipeline
# 运行该段代码要保障你的电脑能够上网，会自动下载预训练模型，大概420M
unmasker = pipeline("fill-mask",model = "bert-base-uncased")  # 这里引入了一个任务叫fill-mask，该任务使用了base的bert模型
print(unmasker.model.name_or_path)
print(unmasker.task)
print(type(unmasker))  #<class 'transformers.pipelines.fill_mask.FillMaskPipeline'>
print(type(unmasker).__base__) # <class 'transformers.pipelines.base.Pipeline'>
print(type(unmasker).__base__.__base__) # 
showParentClass(type(unmasker))
result = unmasker("The goal of life is [MASK].", top_k=5) # 输出mask的指，对应排名最前面的5个，也可以设置其他数字
print(result)
