from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
print(len(dataset))
print(dataset[0]["audio"])
print(dataset[0]["audio"]['path'])
# cast
print('cast')
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
print(dataset[0]["audio"])
print(dataset[0]["audio"]['path'])

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
audio_input = [dataset[0]["audio"]["array"]]
r = feature_extractor(audio_input, sampling_rate=16000)
print(r)