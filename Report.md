# Alphabet Soup Deep Learning Model Report 

## Overview of the Analysis

### Purpose of the Analysis
TThe purpose of this analysis was to develop a deep learning model capable of predicting whether an applicant would be successful in securing funding from Alphabet Soup, a philanthropic organization. The dataset contained financial and categorical information about applicants, which was preprocessed and used to train a binary classification model using a deep neural network. The goal was to achieve an accuracy higher than 75% while optimizing the model’s structure and training parameters.

## Results

### Data Preprocessing
 **Target Variable:** "IS_SUCCESSFUL" is a binary variable indicating whether an applicant received funding (1) or not (0).

**Feature Variables:**

* APPLICATION_TYPE: Categorical variable representing the type of application.

* AFFILIATION: Organization affiliation type.

* CLASSIFICATION: Category under which the nonprofit is classified.

* USE_CASE: Purpose for funding.

* ORGANIZATION: Type of organization.

* STATUS: Status of the application.

* INCOME_AMT: Revenue range of the applicant.

* SPECIAL_CONSIDERATIONS: Indicates if special considerations are needed.

* ASK_AMT: Requested funding amount. 

**Removed Variables:** "EIN" and "NAME" were dropped, as they are unique identifiers that do not contribute to the predictive model.

## Compiling, Training, and Evaluating the Model

### Neural Network Architecture

The number of neurons, layers and functions were the following: 

**Input Layer**

* Number of features = X_train.shape[1] (varied based on preprocessing adjustments).

**Hidden Layers**

1. First Hidden Layer: 64 neurons, ReLU activation.

2. Second Hidden Layer: 32 neurons, ReLU activation.

3. Third Hidden Layer: 16 neurons, ReLU activation (This was added on latteroptimization attempts).

**Output Layer**

* 1 neuron with Sigmoid activation for binary classification.

### Training Parameters

* Loss Function: binary_crossentropy (appropriate for binary classification problems).

* Optimizer: Adam with a learning rate of 0.001 (later experimented with 0.0005).

* Batch Size: 32.

* Epochs: 200 (started with a 100 then 150. We also used EarlyStopping to halt training when accuracy stopped improving).

* Standardization: Data was scaled using StandardScaler() to improve training efficiency.

### Optimization Attempts

To improve accuracy beyond 75%, the following modifications were tested:

1. Added More Hidden Layers and Neurons: Increased layers from 2 to 3 and neurons from 32 to 64 in the first hidden layer.

2. Adjusted Learning Rate: Lowered learning rate to 0.0005 to allow finer adjustments during training.

3. Early Stopping Implementation: Stopped training after 10 consecutive epochs without improvement.

4. Feature Selection Adjustments: Removed variables with low variance (columns with only 2-3 unique values).


### Model Performance

* **Baseline Model Accuracy:** 72.91% (Initial Attempt)

* **Optimized Model Accuracy:** 72.59%

Despite these optimizations, the model did not reach 75% accuracy, indicating potential limitations in the dataset or the approach.

## Summary

The final deep learning model achieved 72.59% accuracy, which, while decent, did not meet the 75% target.

Although I'm not very familiar with other models a quick search with AI will provide other models and what are the
best option depending on the task you're trying to achieve. That being said the AI suggested 4 models which are: Random Forest Classifier, Gradient Boosting (XGBoost, LightGBM, CatBoost), Support Vector Machines (SVM), Hyperparameter Tuning with Neural Networks.

My recommendation since we are trying to boost the accuracy would be either Random Forest or Gradient Boosting, since those two often provide better accuracy and handle data issues more effectively. Some points to take in consideration with each model:

**Random Forest Classifier:** 

* Pros: Handles categorical and numerical data well, robust to overfitting, and interpretable.

* Cons: Slower with large datasets.

**Gradient Boosting:**

* Pros: Excellent for structured data, high predictive power, and feature importance ranking.

* Cons: More complex to tune.