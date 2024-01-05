from transformers import pipeline
import os
# 使用图像识别的程序，来识别一只猫
def abs_path(name):
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    imgpath = os.path.join(file_dir, name)
    return imgpath

vision_classifier = pipeline(model="google/vit-base-patch16-224")

print(vision_classifier.model.name_or_path) 
print(vision_classifier.task)

image_path = "pipeline-cat-chonk.jpeg"
preds = vision_classifier(
    images= abs_path(image_path)
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
print(preds)