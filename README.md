
# Explainable AI Assignment 2 - Model Explanations
In this assignment, you are challenged to explain a model. For this, you will research exisiting approaches and apply them to your model and interpret the results.

## General Information Submission

For the intermediate submission, please enter the group and dataset information. Coding is not yet necessary.

**Team Name:** the AI explainers

**Group Members**

| Student ID    | First Name  | Last Name      | E-Mail                    |  Workload [%] |
| --------------|-------------|----------------|----------------------     |---------------|
| 11942036      | Abdul Basit | Banbhan        |abdul.banbhan@jku.at       |33%           |
| 12126769      | Hala        | Al-Jarajrah    |k12126769@students.jku.at  |33%           |
| 12130348      | Nader       | Essam          |k12130348@students.jku.at  |33%           |

## Explainability framework: Hohman et al
A brief examination major questions [Visual Analytics in Deep Learning: An Interrogative Survey for the Next Frontiers](https://arxiv.org/abs/1801.06889) by Hohman et. al from 2018.

<p align="center">
    <img src="https://fredhohman.com/visual-analytics-in-deep-learning/images/deepvis.png" width="50%" height="50%">
</p>

## Why, Who, What, When, Where, How
TODO

Question | Criterion | Explanation
---      | ---       |         ---
Why      |           | 
Who      |  | 
What     |  |
When     |  |
Where |  |
How   |  | 


## Model

ResNet50 is a convolutional neural network that is trained on the ImageNet dataset, which is a large dataset of images that are organized into 1000 different classes. The network is 50 layers deep, which means it has 50 layers of neurons, including the input and output layers. The network is trained to take an image as input and predict which of the 1000 classes the image belongs to. To do this, the network uses a combination of convolutional layers, which extract features from the images, and fully connected layers, which interpret these features and make the final prediction. It is a popular choice for many computer vision tasks because it is trained to recognize a wide range of objects and scenes, and it can be easily fine-tuned for other specific image classification tasks. The "50" in its name refers to the fact that it has 50 layers, which is relatively deep for a convolutional neural network.

### How to get and use the model?

To use ResNet50 in PyTorch, you can use the torchvision.models module, which contains pre-trained versions of many popular deep learning models. To use ResNet50, you would first need to install PyTorch and torchvision. Then, one can use the following code to import the ResNet50 model:

```python
from torchvision import models

resnet50 = models.resnet50(pretrained=True)
```

This will download the pre-trained ResNet50 model from the PyTorch model zoo and create a new instance of the model. The pretrained argument specifies whether to use the pre-trained weights or randomly initialize the model weights. In this case, we are using the pre-trained weights.

## Dataset

We use ImageNet dataset. ImageNet is a large dataset of images that is commonly used in image classification and computer vision research. It was first introduced in 2009 and has become one of the most widely used datasets for training and evaluating image recognition models. The dataset contains more than 14 million images that are organized into 1000 different classes, such as different types of animals, objects, and scenes. ImageNet has played a key role in the development of many state-of-the-art image classification models, including ResNet50.

<p align="center">
    <img src="https://miro.medium.com/max/750/1*IlzW43-NtJrwqtt5Xy3ISA.jpeg" width="30%" height="30%">
</p>



## Explainability Approaches

Find explainability approaches for the selected models. Repeat the section below for each approach to describe it.

## Approaches

### 1. Feature Maps Visualization

 Visualizing filters and feature maps in convolutional neural networks (CNNs) is a technique used to better understand how these models work and what they are learning. Filters in a CNN are typically small spatial dimensions (e.g. 3x3 or 5x5) that are used to detect specific patterns in the input data. When a filter is applied to an input image, it produces a "feature map" that highlights the regions in the image that match the pattern the filter is looking for. By visualizing these filters and feature maps, we can gain insight into what the CNN is learning and how it is making predictions. This can be useful for debugging the model and improving its performance.

#### Why should one use the method?

There are several reasons why one might use the method of visualizing filters and feature maps in a CNN. Some of the main reasons include:

1. Understanding how the CNN works: By visualizing the filters and feature maps, we can gain a better understanding of how the CNN is processing the input data and making predictions. This can help us to better interpret the results of the model and identify any potential problems or areas for improvement.

2. Debugging the model: Visualizing the filters and feature maps can help us to identify any issues with the model, such as filters that are not learning useful patterns or feature maps that are not capturing the relevant information from the input data. This can be useful for debugging the model and improving its performance.

3. Improving model performance: By gaining a deeper understanding of how the CNN is working, we can make more informed decisions about how to improve the model. For example, we may be able to identify which filters are not learning useful patterns, and therefore remove them from the model to improve its performance.

4. Communicating results to others: Visualizing the filters and feature maps can also be a useful tool for communicating the results of a CNN to others. By providing clear, intuitive visualizations of the model's inner workings, we can make it easier for others to understand and interpret the results.


### Filters 
<p align="center">
    <img src="https://user-images.githubusercontent.com/27974341/207622544-f8e82d79-2649-4690-8f01-7a601f367e36.png" width="30%" height="30%">
</p>

### Convolutional Layers
<p align="center">
    <img src="https://user-images.githubusercontent.com/27974341/207623124-b2fffe5f-da28-419f-8e4d-a11cff450643.png" width="20%" height="20%">                  <img src="https://user-images.githubusercontent.com/27974341/207623120-fcee09f4-33a9-4782-aabe-89fa40b9e240.png" width="20%" height="20%">                  <img src="https://user-images.githubusercontent.com/27974341/207623112-680d10f4-fa7b-457b-8a0e-c528a07c9c90.png" width="20%" height="20%">
</p>



As the picture proceeds through the layers, the details from the photographs gradually fade away. They appear to be noise, but there is undoubtedly a pattern in those feature maps that human eyes cannot identify, but a neural network can.

It is difficult for a human to determine that there is a cat in the image by the time it reaches the last convolutional layer (layer 48). These last layer outputs are critical for the fully connected neurons that comprise the classification layers in a convolutional neural network.

### Feature Visualization by Optimization - ReLU
#### ReLU in the 5. layer  &emsp; &emsp; &emsp;           ReLU in the 10. layer      &emsp;  &emsp; &emsp;       ReLU in the 16. layer
<p align="center">
   <img src="https://user-images.githubusercontent.com/27974341/207628164-bcc04537-c999-488f-8ea4-8af60899e5b0.png" width="30%" height="30%">   <img src="https://user-images.githubusercontent.com/27974341/207628513-08f73eea-889e-4b51-8646-cacf50748bfa.png" width="30%" height="30%">  <img src="https://user-images.githubusercontent.com/27974341/207628669-588da539-f6e8-4cb5-b659-7e2907348719.png" width="30%" height="30%"> 
</p>
<p>
    <em>ReLU in the 5. layer</em>
</p>


### Feature Visualization by Optimization - Convolutional Layer 

#### 3. Convolutional Layer
<p align="center">
    <img src="https://user-images.githubusercontent.com/27974341/207629022-1c688116-e224-4d4c-88f0-0a253dde7442.png" width="30%" height="30%">
</p>

#### 18. Convolutional Layer
<p align="center">
    <img src="https://user-images.githubusercontent.com/27974341/207629145-88eace42-4d75-4294-a352-e800abd3402f.png" width="30%" height="30%">
</p>

#### 47. Convolutional Layer
<p align="center">
    <img src="https://user-images.githubusercontent.com/27974341/207629404-b6ad5151-f33d-40f7-8d32-0950f5758ee1.png" width="30%" height="30%">
</p>

---------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------

### 2. Gradient-weighted Class Activation Mapping (Grad-CAM):
Gradient-weighted Class Activation Mapping (Grad-CAM) is a highly class-discriminative method that uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept. It is a class-discriminative localization technique that generates visual explanations for any CNN-based network without requiring architectural changes or re-training. The technique is an improvement of other approaches in versatility and accuracy. It is complex but the output is intuitive. From a high-level, an image is taken as input and a model is created that is cut off at the layer for which we want to create a Grad-CAM heat-map. 

The following code is used to initialize the GradCAM:

```python
cam = GradCAM(model=model, target_layers=target_layers)
```

<img src="https://user-images.githubusercontent.com/92387828/207347417-41ab3e69-9358-4b5c-bdf9-7f65d59422dd.jpg" width="30%" height="30%"> ![](https://user-images.githubusercontent.com/92387828/207347090-d771e587-0c08-421d-807f-592bd4361a7e.PNG)


#### Why should one use the method?
The gradCAM function computes an importance map that is enables understanding of the model. In short, the method is interpretable even by non-experts.

#### What could be visualized?
Grad-CAM is a popular technique for visualizing where a convolutional neural network model is looking. It shows the most important features and their influence on the model prediction.

#### When is this method used?
Grad-CAM is applied to a neural network that is done training where the weights of the neural network are fixed. An image is fed into the network to calculate the Grad-CAM heatmap for that image for a chosen class of interest. Here, the method was used after training.

#### Who would benefit from this method?
Both methods benefit model developers in which the method would make their work easier and faster, also more understandable. It is also interpretable by non-experts.

#### How can this method explain the model?
The Grad-CAM technique utilizes the gradients of the classification score with respect to the final convolutional feature map, to identify the parts of an input image that most impact the classification score. The places where this gradient is large are exactly the places where the final score depends most on the data.

#### Where could this method be used?
The method could be used for any model in which it is usually used to help explain problems, which have a lot of various features.  wherever there is a need to understand and interpret the inner workings of a CNN, such as when training and fine-tuning the model or when using the model to make predictions on new data.

---------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------
### 3. Counterfactual Explainations using Adversarial Example:

Counterfactual explanations can be used to explain predictions of individual instances. The “event” is the predicted outcome of an instance, the “causes” are the particular feature values of this instance that were input to the model and “caused” a certain prediction. 
Displayed as a graph, the relationship between the inputs and the prediction is very simple: The feature values cause the prediction.

<p align="center">
    <img src="https://christophm.github.io/interpretable-ml-book/images/graph.jpg" width="30%" height="30%">
</p>

Adversarial examples are counterfactual examples with the aim to deceive the model, There are many techniques to create adversarial examples. Most approaches suggest minimizing the distance between the adversarial example and the instance to be manipulated, while shifting the prediction to the desired (adversarial) outcome. Some methods require access to the gradients of the model, which of course only works with gradient based models such as neural networks, other methods only require access to the prediction function, which makes these methods model-agnostic.

In this Assignment, we created an Adversarial Example using FGSM (Fast Gradient Sign Method)
#### Why should one use the Adversarial Example?
Deceving a model shows us how robust the model can be, in sense of classification model, we can figure out which classes are most vulnerable in the model.

#### When is this method used?
Adversarial Examples can be create when we have an access to a model that is done training.

#### Who would benefit from this method?
The model developers can benefit from this method when debugging to figure out which classes that the network is most vulnerable at.


## Ineractive Demo via Streamlit
TODO
Making ineractive visualizations helps to understand the model better and get a better insight and that all about in XAI. For this, we will use [Streamlit](https://streamlit.io/). Streamlit is a Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.

[vis_feature_maps.webm](https://user-images.githubusercontent.com/27974341/206809086-2d41b28a-2fa2-4b69-b7cf-24a3d46f1c3b.webm)



[gradcam_demo.webm](https://user-images.githubusercontent.com/27974341/207174388-fc513dc1-b281-4eeb-a96b-03e7a8d26969.webm)


## Final Submission
The submission is done with this repository. Make to push your code until the deadline.

The repository has to include the implementations of the picked approaches and the filled out report in this README.

* Sending us an email with the code is not necessary.
* Update the *environment.yml* file if you need additional libraries, otherwise the code is not executeable.
* Save your final executed notebook(s) as html (File > Download as > HTML) and add them to your repository.

## Development Environment

Checkout this repo and change into the folder:
```
git clone https://github.com/jku-icg-classroom/xai_model_explanation_2022-<GROUP_NAME>.git
cd xai_model_explanation_2022-<GROUP_NAME>
```

Load the conda environment from the shared `environment.yml` file:
```
conda env create -f environment.yml
conda activate xai_model_explanation
```

> Hint: For more information on Anaconda and enviroments take a look at the README in our [tutorial repository](https://github.com/JKU-ICG/python-visualization-tutorial).

Then launch Jupyter Lab:
```
jupyter lab
```

Alternatively, you can also work with [binder](https://mybinder.org/), [deepnote](https://deepnote.com/), [colab](https://colab.research.google.com/), or any other service as long as the notebook runs in the standard Jupyter environment.
