from transformers import pipeline
import os
def abs_path(name):
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    imgpath = os.path.join(file_dir, name)
    return imgpath

vqa = pipeline(model="impira/layoutlm-document-qa")
# vqa = pipeline(model="microsoft/layoutlm-base-cased")

print(vqa.model.name_or_path) 
print(vqa.task) #document-question-answering

image_path = "invoice.png"
# image_path = "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png"
anwser = vqa(
    image= abs_path(image_path),
    question="What is the invoice number?",
)
print(anwser)
anwser = vqa(
    image= abs_path(image_path),
    question="invoice number",
)
print(anwser)