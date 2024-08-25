import numpy as np
import gradio as gr
import torch
from transformers import AutoImageProcessor, ViTForImageClassification, AutoModelForImageClassification
from peft import PeftConfig, PeftModel
import os 
from PIL import Image


##### Loading Model #####
classes = ['chihuahua', 'newfoundland', 'english setter', 'Persian', 'yorkshire terrier', 'Maine Coon', 'boxer', 'leonberger', 'Birman', 'staffordshire bull terrier', 'Egyptian Mau', 'shiba inu', 'wheaten terrier', 'miniature pinscher', 'american pit bull terrier', 'Bombay', 'British Shorthair', 'german shorthaired', 'american bulldog', 'Abyssinian', 'great pyrenees', 'Siamese', 'Sphynx', 'english cocker spaniel', 'japanese chin', 'havanese', 'Russian Blue', 'saint bernard', 'samoyed', 'scottish terrier', 'keeshond', 'Bengal', 'Ragdoll', 'pomeranian', 'beagle', 'basset hound', 'pug']

label2id = {c:idx for idx,c in enumerate(classes)}
id2label = {idx:c for idx,c in enumerate(classes)}

model_name = "vit-base-patch16-224"
model_checkpoint = f"google/{model_name}"

repo_name = f"md-vasim/{model_name}-finetuned-lora-oxfordPets"

config = PeftConfig.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
# Load the Lora model
inference_model = PeftModel.from_pretrained(model, repo_name)
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

def inference(image):
    encoding = image_processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = inference_model(**encoding)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    result = inference_model.config.id2label[predicted_class_idx]
    return result 


##### Interface #####

title = "Welcome to Vision Transformers Classification"

description = """ 
Vision Transformers (ViTs) are used in image classification tasks by breaking down an image into small patches, treating each patch like a token in a sequence. These tokens are then processed by the Transformer model, which uses self-attention to learn and capture the relationships between different parts of the image. This global understanding enables ViTs to classify images with high accuracy, making them effective for identifying objects, scenes, or patterns in various computer vision applications. ViTs are particularly powerful in scenarios requiring detailed and context-aware image analysis.
"""

output_box = gr.Textbox(
            label="Output"
        )

examples=[
        ["./static/shiba_inu_174.jpg"],
        ["./static/pomeranian_89.jpg"],
    ]

demo = gr.Interface(
    inference, 
    gr.Image(type="pil"), 
    output_box, 
    examples=examples,
    title=title,
    description=description,
    )

if __name__ == "__main__":
    demo.launch()
