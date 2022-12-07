
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


## Report

### Model & Data

* Which model are you going to explain? What does it do? On which data is it used?
* From where did you get the model and the data used?
* Describe the model.

### Explainability Approaches
Find explainability approaches for the selected models. Repeat the section below for each approach to describe it.

#### Approaches
##### 1. LIME: Local Interpretable Model-agnostic Explanations

LIME is a technique that approximates any black box machine learning model with a local, interpretable model to explain each individual prediction. The LIME method approximates complicated black-box model to a simpler glass-box one and is usually used with problems having very large number of explanatory variables. In that case the resulting, simpler glass-box model is easier to interpret. The main idea is to train a simpler glass-box model on artificial data so that it approximates the predictions of a complicated black-box model. LIME helps to explain and understand the model locally, but can also be helpful with checking which features are considered as more important, and which seem not to be useful.

###### Why should one use the method?
This method creates a simpler glass-box that is implemented based on the black-box model and is easier to interpret.

###### What could be visualized?
Probabilities of the most important features and their influence (positive or negative) on the model prediction.

###### When is this method used?
It can be used after training to evaluate the model as well as during training. Using the LIME method during training will help identify if some of the features are not important for training, allowing us to consider reducing the number of features and seeing this feature's effects on model prediction. Here LIME was used after training.

###### Who would benefit from this method?
Model developers benefit from this method as it would make their work easier and faster, also more understandable.

###### How can this method explain the model?
The simpler glass-box model is trained on artificial data in a way that its predictions are similar to the predictions of original model.

###### Where could this method be used?
It could be used for many models in which it is usually used to help explain problems, which have a lot of various features.

* Breifly summarize the approach. 
* Categorize this explainability approach according to the criteria by Hohman et al.
* Interpret the results here. How does it help to explain your model?

### Summary of Approaches
Write a brief summary reflecting on all approaches.
