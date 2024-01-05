# https://huggingface.co/docs/transformers/pipeline_tutorial

import os
from transformers import pipeline

def abs_path(name):
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    imgpath = os.path.join(file_dir, name)
    return imgpath
generator = pipeline(task="automatic-speech-recognition")

print(generator.model.name_or_path) #facebook/wav2vec2-base-960h
print(generator.task)
# 把这个声音文件下载到本地了
# r = generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
r = generator(abs_path("./mlk.flac"))
print(r)