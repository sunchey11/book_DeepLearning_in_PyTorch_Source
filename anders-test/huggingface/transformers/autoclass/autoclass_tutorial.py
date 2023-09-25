# https://huggingface.co/docs/transformers/autoclass_tutorial
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))

# ImageProcessor
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# AutoFeatureExtractor
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

# AutoProcessor
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")