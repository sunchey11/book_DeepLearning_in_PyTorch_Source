# https://zhuanlan.zhihu.com/p/535100411
def showParentClass(clazz):
    """显示父类"""
    indent = ''
    p = clazz
    print(clazz)
    while hasattr(p, '__base__'):
        p = p.__base__
        indent += '  '
        print(indent+'|__'+ str(p))
import torch
from transformers import BertModel, BertTokenizer, BertConfig
# 首先要import进来
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
config = BertConfig.from_pretrained('bert-base-chinese')
config.update({'output_hidden_states':True}) # 这里直接更改模型配置
model = BertModel.from_pretrained("bert-base-chinese",config=config)
showParentClass(type(model))

# 上文的示例代码已经实例话了，这里不重复了；
print(tokenizer.encode("生活的真谛是美和爱"))  # 对于单个句子编码
print(tokenizer.encode_plus("生活的真谛是美和爱","说的太好了")) # 对于一组句子编码