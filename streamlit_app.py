import streamlit as st
import torch
import matplotlib.pyplot as plt
import cv2 as cv
import random
import os
import torchvision.models as models
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from streamlit_image_select import image_select

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
from matplotlib import image
from glob import glob

import pandas as pd


# read the csv file and store the data in a pandas dataframe first row is the header
@st.cache
def load_data():
    df = pd.read_csv("./imagenette2.csv", header=0, sep=";")
    return df


"# Model Explanations"

"""
There are a lot of ways to explain a model. 
In this example, we will visualizing filters and feature maps, use the Grad-CAM algorithm and use the LIME algorithm.
"""

"""
## Model
We will use a pretrained ResNet18 model.

"""

"""
## Why, Who, What, When, Where, How
According to [Hohnman et. al.](https://arxiv.org/abs/1801.06889 "Visual Analytics in Deep Learning: An Interrogative Survey for the Next Frontiers") paper:

Question | Criterion | Explanation
--- | --- | ---
Why | | 
Who | | 
What |  | 
When |  |
Where |  |
How |  | 




"""

"""
## Data
https://github.com/fastai/imagenette
TODO

"""

# separater
st.markdown("---")
# https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
"""
## Visualizing Filters and Feature Maps
Convolutional neural networks (CNN) are the architectures of choice for dealing with picture data in deep learning. 
CNNs have shown to give various cutting-edge deep learning and computer vision solutions and benchmarks. 
Convolutional neural networks have several uses, including image recognition, object identification, and semantic segmentation.
However, when it comes to how a convolutional neural network determines what an image is, things grow more complicated. 

The question might take several forms:
- How did the neural network determine that the image depicted is of a cat?
- Why was a cat classified as a bird in the image?
- What did the convolutional neural network see in the intermediate layers?
"""


def visualize_deep(outputs, index):
    """
    Visualize the first 64 feature maps of the convolutional layer

    outputs: all feature maps
    index: index of layer
    """
    fig = plt.figure(figsize=(30, 30))
    layer_viz = outputs[index][0, :, :, :].data
    for i, filter in enumerate(layer_viz):
        if i == 64:  # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap="gray")
        plt.axis("off")
    return fig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the model
# net = models.resnet18()
# for param in net.parameters():
#     param.requires_grad = False
# net.fc = nn.Linear(in_features=4096, out_features=10)
# net = net.to(device)

# load the model
st.cache()


def load_model():
    model = models.resnet50(pretrained=True)
    # get all the model children as list
    model_children = list(model.children())
    return model, model_children


model, model_children = load_model()
model_weights = []  # we will save the conv layer weights in this list
conv_layers = []  # we will save the 49 conv layers in this list

# Extract all conv layers. We are not interested in the dense layers so we skip them
# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)


# visualize the filters (size 8x8) of the first conv layer
# visualizing other filters doesn't make sense, since they have a size of only 3x3
fig = plt.figure(figsize=(20, 17))
for i, filter in enumerate(model_weights[0]):
    plt.subplot(
        8, 8, i + 1
    )  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap="gray")
    plt.axis("off")
# plt.show()
st.pyplot(fig)


"### Let's upload an image and see what the model sees."

image_file = st.file_uploader(
    "Upload Images 📸", type=["png", "jpg", "jpeg"], key="image"
)  # upload image


image_file_select = image_select(
    "Select any image from below:",
    ["./pics/cat.jpg", "./pics/frankfurt.jpg", "./pics/sunset.jpg"],
)

# check if an image is uploaded or selected from the list of images


if image_file_select is not None:
    img = cv.imread(image_file_select)
if image_file is not None:
    img = cv.imread(f"./pics/{image_file.name}")
if img is not None:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    # define the transforms
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # print(img.size())
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)

    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    layer = st.slider(
        "Which layer should be ploted?", min_value=0, max_value=counter - 1, value=1
    )

    with st.spinner(
        f"Wait... visualize the first 64 feature maps of the {layer}. convolutional layer"
    ):
        st.pyplot(visualize_deep(outputs, layer))
    st.success("Done! 🪄")


# separater
st.markdown("---")

"""
## Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization 

Grad-CAM is a technique for class activation visualization. 
It is a simple and effective method for visualizing the class activation maps (CAM) for a given image. 
It is based on the gradient information of any target concept (say logits for ‘dog’ or even a caption), flowing into the final convolutional layer of a CNN architecture. 
The method is described in the paper [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization](https://arxiv.org/abs/1610.02391 "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization").

"""
"### Let's have a look to the data set."
image_list_test = glob("./pics/imagenette2-320/val/n02102040/*.JPEG")

fig, axarr = plt.subplots(5, 5, figsize=(35, 35))
for i in range(5):
    for j in range(5):
        axarr[i, j].imshow(image.imread(random.choice(image_list_test)))
        axarr[i, j].axis("off")
st.pyplot(fig)

# model = resnet18(pretrained=True)
target_layers = [model.layer4[-1]]
# initialize the GradCAM object
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
# choose target class
# tench = 0
# English springer = 217
# cassette player =  482
# chain saw
# church
# French horn
# garbage truck
# gas pump
# golf ball
# parachute

# dic with class names and their index
CHOICES = {
    0: "Tench",
    217: "English springer",
    482: "Cassette player",
    491: "Chainsaw",
    497: "Church",
    566: "French horn",
    467: "Garbage truck",
    571: "Gas pump",
    574: "Golf ball",
    701: "Parachute",
}


def format_func(option):
    return CHOICES[option]


target_class = st.selectbox(
    "Select the target class", options=list(CHOICES.keys()), format_func=format_func
)

targets = [ClassifierOutputTarget(target_class)]

image_file_gradcam = st.file_uploader(
    "Upload Images 📸", type=["png", "jpg", "jpeg"], key="gradcam"
)  # upload image

df = load_data()
if image_file_gradcam is not None:
    path = df.loc[df["filename"] == image_file_gradcam.name, "path"].values[0]
    img = cv.imread(path)
    image_test = np.array(img)

    input_tensor = torch.from_numpy(
        np.expand_dims(np.transpose(image_test / 255.0, (2, 0, 1)), 0).astype(
            np.float32
        )
    )

    with st.spinner(
        f"Wait... visualize the Grtad-CAM of the {image_file_gradcam.name} image"
    ):

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True,
        )

        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(
            np.array(
                image_test / 255,
            ).astype(np.float32),
            grayscale_cam,
            use_rgb=True,
        )
        # show the visualization in streamlit

        st.image(Image.fromarray(visualization, "RGB"))
    st.success("Done! 🪄")
