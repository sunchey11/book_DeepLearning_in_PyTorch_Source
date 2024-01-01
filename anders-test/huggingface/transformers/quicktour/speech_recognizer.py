
# 必须装soundfile librosa，否则报错ImportError: To support decoding audio files, please install 'librosa' and 'soundfile'.
# pip install soundfile librosa

import torch
from transformers import pipeline

# https://huggingface.co/docs/transformers/quicktour
def showParentClass(clazz):
    """显示父类"""
    indent = ''
    p = clazz
    print(clazz)
    while hasattr(p, '__base__'):
        p = p.__base__
        indent += '  '
        print(indent+'|__'+ str(p))
        
speech_recognizer = pipeline("automatic-speech-recognition", 
                             model="facebook/wav2vec2-base-960h")
print(speech_recognizer.model.name_or_path) #openai/whisper-large-v2
print(speech_recognizer.task)
print(showParentClass(type(speech_recognizer.model))) 

from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
# 对数组前面的4个音频，进行识别
result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])

print('finished')