## **Table of Contents**

- [4. 1. Cross Validation](#4-1-cross-validation)
- [4. 2. Scores](#4-2-scores)
  * [4. 2. 1. ROC AUC Score](#4-2-1-roc-auc-score)
  * [4. 2. 2. Precision Score](#4-2-2-precision-score)
  * [4. 2. 3. Recall Score](#4-2-3-recall-score)
  * [4. 2. 4. F1 Score](#4-2-4-f1-score)
  * [4. 2. 5. Balanced Accuracy Score](#4-2-5-balanced-accuracy-score)

---

## [4. 1. Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

In order to evaluate the performance of a Machine Learning model, we can employ several methods. One commonly used method that can be universally applied is to train on a subset of the data and predict and evaluate on the remainder. This is used to investigate the performance of the model on previously unseen data. Generally speaking, one wants the model to generalize well, referring to the idea that a machine learning model should be able to capture the trend of the data. If this is not possible because the model is too simple, we refer to this as `underfitting`. Contrary to that, a model that is too complex could `overfit` and starts to capture sample-specific noise and thus will not be able to maintain accuracy when using another dataset.

For a typical approach, one would split all available data into train, validation, and test set. Here, the train and validation sets are used to optimize a Machine Learning method, and the test set is used for a final test on unseen data. In the proteomics context, however, data is limited, and taking away additional data is often not possible.

One way, to get an estimate on how a model would generalize while still trying to use all available data is Cross-Validation (CV). Here, data is repeatedly split into train and validation sets. 

Proto-learn is using a stratified-k-fold split, meaning that the original class ratio will be preserved for the splits. 
Also, it is possible to shuffle the data and repeatedly shuffle the data and splitting it. The average of multiple splits gives a more robust estimate of the model performance.  `cv_splits` and `cv_repeats`.
Here is the representative image for K-Fold CV approach:

## [4. 2. Scores](https://scikit-learn.org/stable/modules/model_evaluation.html)
In Machine Learning (ML), there are several metrics to be employed for measuring the performance of the model, and for evaluating the quality of predictions.

### [4. 2. 1. ROC AUC Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
This score reflects the computation of Area Under the Curve (`AUC`) of Receiver Operating Characteristics (`ROC`). It is also known as Area Under the Receiver Operating Characteristics (AUROC). 

AUC - ROC curve is employed for measuring the performance of the models for classification problems at all different classification thresholds.

Technically, `ROC` is a probability curve and it includes two dimensions/parameters called `TPR` and `FPR` (in which TPR locates on the y-axis while FPR is on the x-axis), and `AUC` is simply the two-dimensional area under the `ROC` curve. 

1. True Positive Rate (TPR)

    The synonyms of TPR are `Recall` and `Sensitivity`. 

    Here is the formulation for TPR:

    <img src="https://latex.codecogs.com/gif.latex?TPR=\frac{TP}{(TP)&plus;(FN)}" title="TPR=\frac{TP}{TP+FN}" />

2. False Positive Rate (FPR)

    It is defined as:

    <img src="https://latex.codecogs.com/gif.latex?FPR=\frac{FP}{FP&plus;TN}" title="FPR=\frac{FP}{FP+TN}" />


Here is how the ROC curve and AUC look like:

![ROC-AUC](https://user-images.githubusercontent.com/49681382/90752147-b75aff80-e2df-11ea-9444-e8762496aa9f.png)

_Source: [ROC Curve Image](https://www.datasciencecentral.com/profiles/blogs/roc-curve-explained-in-one-picture)_


Also, be noted that `AUC` score lies between 0 and 1. So, if `AUC` has a score of 0.0, it means that 100% of the model predictions is wrong while the model also performs great classification task with an `AUC` score of 1.0. 

In other words, a higher AUC score means a better model for the classification task.

### [4. 2. 2. Precision Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)
Precision score refers to a ratio of correctly classified positive samples to the total classified positive samples. 

In other words, `Precision` score questions that how many tissue samples were labeled as positive among the other positive samples? 

Here, for `Precision` score, the best value is `1` while the worst is `0`.

The `precision` score is formulated as below:

<img src="https://latex.codecogs.com/gif.latex?Precision=\frac{TP}{TP&plus;FP}" title="Precision=\frac{TP}{TP+FP}" /> 

### [4. 2. 3. Recall Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)
`Recall` score is also known as `Sensitivity`. 

This metric computes the fraction of positively predicted samples out of all results that should have been returned.

The `Recall` score is formulated as below:

<img src="https://latex.codecogs.com/gif.latex?Recall=\frac{TP}{TP&plus;FN}" title="Recall=\frac{TP}{TP+FN}" /> 

Here, for the `Recall` score, the best value is `1` while the worst is `0`.

> **`Precision` vs. `Recall`**
> 
> - If your goal is minimizing false positives (samples that normally negative/absent but wrongly classified as positive/present), it is appropriate to focus on `Precision`.
>
> - If your goal is minimizing false negatives (samples that normally positive/present but wrongly classified as negative/absent), it is appropriate to focus on `Recall`.

### [4. 2. 4. F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
`F1 Score` is the weighted average (or in other words harmonic mean) of both `Precision` and `Recall`.

Here, for the `F1 Score`, the best value is `1` while the worst is `0`.

The `F1 Score` is expressed as following equaiton:

<img src="https://latex.codecogs.com/gif.latex?F1&space;=&space;2&space;*&space;\frac{Precision&space;*&space;Recall}{Precision&space;&plus;&space;Recall}" title="F1 = 2 * \frac{Precision * Recall}{Precision + Recall}" />

### [4. 2. 5. Balanced Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)
To handle imbalanced datasets that might inflate the model performance in binary and multiclass classification tasks, the `balanced accuracy score` is employed. 

So, It computes the average of `Recall` for each class. Hence, this `Balanced Accuracy Score` is equal to `Accuracy` if the datasets are balanced.

Here, for the `Balanced Accuracy` Score, the best value is `1` while the worst is `0`.
