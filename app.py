import streamlit as st
import torch
import cv2
import numpy as np
import torch_pruning as tp
import time
from PIL import Image
from torchvision import transforms, models

st.set_option('deprecation.showfileUploaderEncoding', False)

classes = ["fake", "real"]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource

def load_model():
    model = None
    new_model = models.efficientnet_v2_s()

    loaded_state_dict = torch.load('../trained_model/tp-model.pth', map_location=device.type)
    tp.load_state_dict(new_model, state_dict=loaded_state_dict)
    model = new_model
    return model

def classify(model, image_transforms, image, classes):
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    image = image.to(device)

    start_time = time.time()
    output = model(image)
    end_time = time.time()
    _, predicted = torch.max(output.data, 1)

    st.success("Inference time: {:.4f} seconds".format(end_time - start_time))
    st.success(classes[predicted.item()])

model = load_model()

st.write(
    """
    Terra Fake Image Detector
    """
)

file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])


if file is None:
    st.text("No file selected")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = classify(model, image_transforms, image, classes)
    # clas_names = ["Real", "Fake"]
    # result_string = f"This image is {100 * (1 - predictions[0][0]):.2f}% Fake and {100 * predictions[0][0]:.2f}% Real."
    # st.success(result_string)