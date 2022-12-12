
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

![](https://fredhohman.com/visual-analytics-in-deep-learning/images/deepvis.png)

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


## Model & Data

* Which model are you going to explain? What does it do? On which data is it used?
* From where did you get the model and the data used?
* Describe the model.

### Explainability Approaches

Find explainability approaches for the selected models. Repeat the section below for each approach to describe it.

### Approaches

#### 1. Feature Maps Visualization

 Visualizing filters and feature maps in convolutional neural networks (CNNs) is a technique used to better understand how these models work and what they are learning. Filters in a CNN are typically small spatial dimensions (e.g. 3x3 or 5x5) that are used to detect specific patterns in the input data. When a filter is applied to an input image, it produces a "feature map" that highlights the regions in the image that match the pattern the filter is looking for. By visualizing these filters and feature maps, we can gain insight into what the CNN is learning and how it is making predictions. This can be useful for debugging the model and improving its performance.

##### Why should one use the method?

There are several reasons why one might use the method of visualizing filters and feature maps in a CNN. Some of the main reasons include:

1. Understanding how the CNN works: By visualizing the filters and feature maps, we can gain a better understanding of how the CNN is processing the input data and making predictions. This can help us to better interpret the results of the model and identify any potential problems or areas for improvement.

2. Debugging the model: Visualizing the filters and feature maps can help us to identify any issues with the model, such as filters that are not learning useful patterns or feature maps that are not capturing the relevant information from the input data. This can be useful for debugging the model and improving its performance.

3. Improving model performance: By gaining a deeper understanding of how the CNN is working, we can make more informed decisions about how to improve the model. For example, we may be able to identify which filters are not learning useful patterns, and therefore remove them from the model to improve its performance.

4. Communicating results to others: Visualizing the filters and feature maps can also be a useful tool for communicating the results of a CNN to others. By providing clear, intuitive visualizations of the model's inner workings, we can make it easier for others to understand and interpret the results.


##### What could be visualized?

When visualizing filters and feature maps in a CNN, there are several things that can be visualized. These include:

1. Filters: The filters in a CNN are the small spatial dimensions (e.g. 3x3 or 5x5) that are used to detect specific patterns in the input data. By visualizing the filters, we can see what kind of patterns the CNN is looking for in the input data.

2. Feature maps: When a filter is applied to an input image, it produces a "feature map" that highlights the regions in the image that match the pattern the filter is looking for. By visualizing the feature maps, we can see how well the filters are detecting the patterns in the input data.

3. Activation maps: Activation maps show the areas of the input image that are most "activated" by the filters, i.e. the areas that are most relevant to the predictions made by the CNN. By visualizing the activation maps, we can see which parts of the input image are most important to the model's predictions.

##### When is this method used?

The method of visualizing filters and feature maps in a CNN is typically used during the development and testing phases of a machine learning project. It is most often used when training and fine-tuning a CNN, as this is when we have the most control over the model and can make the most informed decisions about how to improve its performance.

##### Who would benefit from this method?

There are several groups of people who may benefit from the method of visualizing filters and feature maps in a convolutional neural network (CNN). These include:

1. Machine learning researchers and practitioners: This method can be useful for researchers and practitioners who are working on developing and training CNNs. By visualizing the filters and feature maps, they can gain a better understanding of how the CNN is working and what it is learning, which can be helpful for debugging the model and improving its performance.

2. Data scientists and analysts: Data scientists and analysts who are working with image data may find this method useful for gaining insights into how a CNN is processing and interpreting the data. By visualizing the filters and feature maps, they can see which patterns the CNN is detecting in the data and how it is using these patterns to make predictions.

3. Students and educators: Students and educators who are learning about CNNs and how they work may benefit from using this method as a way to gain a better understanding of the inner workings of these models. By visualizing the filters and feature maps, they can see how the CNN is processing the input data and making predictions, which can help to make the concepts more concrete and intuitive.

##### How can this method explain the model?

By visualizing the filters and feature maps of a ResNet50 model, we can gain insights into how the model is processing the input data and making predictions.

For example, we can visualize the filters in the first convolutional layer of a ResNet50 model to see what kind of patterns the model is looking for in the input data. We can then visualize the feature maps produced by these filters to see how well the filters are detecting these patterns in the input data. Finally, we can visualize the activation maps to see which parts of the input image are most important to the model's predictions.

##### Where could this method be used?

The method of visualizing filters and feature maps in a convolutional neural network (CNN) can be used in a variety of settings, including research labs, academic institutions, and industry. In general, this method can be used wherever there is a need to understand and interpret the inner workings of a CNN, such as when training and fine-tuning the model or when using the model to make predictions on new data.

For example, machine learning researchers and practitioners may use this method when developing and training CNNs for a wide range of applications, such as image classification, object detection, and segmentation. Data scientists and analysts may also use this method when working with image data to gain insights into how a CNN is processing and interpreting the data. Additionally, students and educators may use this method as a tool for learning about CNNs and how they work.


#### 2. LIME: Local Interpretable Model-agnostic Explanations

LIME is a technique that approximates any black box machine learning model with a local, interpretable model to explain each individual prediction. The LIME method approximates complicated black-box model to a simpler glass-box one and is usually used with problems having very large number of explanatory variables. In that case the resulting, simpler glass-box model is easier to interpret. The main idea is to train a simpler glass-box model on artificial data so that it approximates the predictions of a complicated black-box model. LIME helps to explain and understand the model locally, but can also be helpful with checking which features are considered as more important, and which seem not to be useful.

##### Why should one use the method?
This method creates a simpler glass-box that is implemented based on the black-box model and is easier to interpret.

##### What could be visualized?
Probabilities of the most important features and their influence (positive or negative) on the model prediction.

##### When is this method used?
It can be used after training to evaluate the model as well as during training. Using the LIME method during training will help identify if some of the features are not important for training, allowing us to consider reducing the number of features and seeing this feature's effects on model prediction. Here LIME was used after training.

##### Who would benefit from this method?
Model developers benefit from this method as it would make their work easier and faster, also more understandable.

##### How can this method explain the model?
The simpler glass-box model is trained on artificial data in a way that its predictions are similar to the predictions of original model.

##### Where could this method be used?
It could be used for many models in which it is usually used to help explain problems, which have a lot of various features.

* Breifly summarize the approach. 
* Categorize this explainability approach according to the criteria by Hohman et al.
* Interpret the results here. How does it help to explain your model?

### Summary of Approaches
Write a brief summary reflecting on all approaches.

## Ineractive Demo via Streamlit
TODO
Making ineractive visualizations helps to understand the model better and get a better insight and that all about in XAI. For this, we will use [Streamlit](https://streamlit.io/). Streamlit is a Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.

[vis_feature_maps.webm](https://user-images.githubusercontent.com/27974341/206809086-2d41b28a-2fa2-4b69-b7cf-24a3d46f1c3b.webm)


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
