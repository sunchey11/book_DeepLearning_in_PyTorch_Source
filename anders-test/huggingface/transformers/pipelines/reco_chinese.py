# https://huggingface.co/docs/transformers/pipeline_tutorial
# 识别中文的测试
import os
from transformers import pipeline

def abs_path(name):
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    imgpath = os.path.join(file_dir, name)
    return imgpath
def showParentClass(clazz):
    """显示父类"""
    indent = ''
    p = clazz
    print(clazz)
    while hasattr(p, '__base__'):
        p = p.__base__
        indent += '  '
        print(indent+'|__'+ str(p))

generator = pipeline(model="openai/whisper-large-v2")
# generator = pipeline(model="openai/whisper-large-v3")
print(generator.model.name_or_path) #openai/whisper-large-v2
print(generator.task)
print(showParentClass(type(generator.model))) 
# r = generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
# r = generator(abs_path("./mlk.flac"))
r = generator(abs_path("./cc.wav"),
              generate_kwargs  = {"task":"transcribe", "language":"chinese"})
print(r)