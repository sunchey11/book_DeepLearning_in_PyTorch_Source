from transformers import AutoProcessor, AutoModel
import os
def abs_path(name):
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    imgpath = os.path.join(file_dir, name)
    return imgpath

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")
text = "Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."
text = "明早咱们开会."
inputs = processor(
    text=[text],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)

from IPython.display import Audio

sampling_rate = model.generation_config.sample_rate
a = Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)
a.autoplay = True
print(a.autoplay)

import scipy

# sampling_rate = model.config.codec_config.sampling_rate
scipy.io.wavfile.write(abs_path("bark_out.wav"), rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())