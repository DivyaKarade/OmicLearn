## **Table of Contents**

- [**Table of Contents**](#table-of-contents)
- [4. 1. Cross Validation](#4-1-cross-validation)
- [4. 2. Scores](#4-2-scores)
  - [4. 2. 1. ROC AUC Score](#4-2-1-roc-auc-score)
  - [4. 2. 2. PR AUC Score](#4-2-2-pr-auc-score)
  - [4. 2. 3. Precision Score](#4-2-3-precision-score)
  - [4. 2. 4. Recall Score](#4-2-4-recall-score)
  - [4. 2. 5. F1 Score](#4-2-5-f1-score)
  - [4. 2. 6. Balanced Accuracy Score](#4-2-6-balanced-accuracy-score)

---

## [4. 1. Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

In order to evaluate the performance of a Machine Learning (ML) model, we can employ several methods. One commonly used method that can be universally applied is to train on a subset of the data and predict and evaluate on the remainder. This is used to investigate the performance of the model on previously unseen data. Generally speaking, one wants the model to generalize well, referring to the idea that a machine learning model should be able to capture the trend of the data. If this is not possible because the model is too simple, we refer to this as `underfitting`. Contrary to that, a model that is too complex could `overfit` and starts to capture sample-specific noise and thus will not be able to maintain accuracy when using another dataset.

For a typical approach, one would split all available data into train, validation, and test set. Here, the train and validation sets are used to optimize a Machine Learning method, and the test set is used for a final test on unseen data. In the proteomics context, however, data is limited, and taking away additional data is often not possible.

One way to get an estimate on how a model would generalize while still trying to use all available data is Cross-Validation (CV). Here, data is repeatedly split into train and validation sets. 

Proto-learn is using a stratified-k-fold split, meaning that the original class ratio will be preserved for the splits. 
Also, it is possible to shuffle the data and repeatedly shuffle the data and splitting it. The average of multiple splits gives a more robust estimate of the model performance.  The number of splits and the number of repeats can be changed with `cv_splits` and `cv_repeats`.

Also, within Proto Learn, [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) and [StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) methods might be used in addition to [RepeatedStratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html).

## [4. 2. Scores](https://scikit-learn.org/stable/modules/model_evaluation.html)
In ML, there are several metrics to be employed for measuring the performance of the model, and for evaluating the quality of predictions.

### [4. 2. 1. ROC AUC Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
This score reflects the computation of Area Under the Curve (`AUC`) of Receiver Operating Characteristics (`ROC`). It is also known as Area Under the Receiver Operating Characteristics (AUROC). 

A good brief introduction for understanding the `ROC` can be found [here](https://www.datasciencecentral.com/profiles/blogs/roc-curve-explained-in-one-picture). 

### [4. 2. 2. PR AUC Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
This `Precision-Recall Curve` (or `PR Curve`) is a plot of the `precision` on the y-axis and the `recall` on the x-axis at different thresholds.

Also, the `PR AUC Score` is the calculation of Area Under the Curve (`AUC`) of `Precision-Recall Curve` (`PR Curve`).

> For imbalanced datasets, `PR Curve` might be a better choice.

A detailed explanation for `PR Curve` can be found [here](https://acutecaretesting.org/en/articles/precision-recall-curves-what-are-they-and-how-are-they-used).

### [4. 2. 3. Precision Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)

Precision refers to a ratio of correctly classified positive samples to the total classified positive samples. 

### [4. 2. 4. Recall Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)
`Recall` score is also known as `Sensitivity`.  This metric computes the fraction of true positive results out of all positively predicted results.

### [4. 2. 5. F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
`F1 Score` is the weighted average of both `Precision` and `Recall`.

### [4. 2. 6. Balanced Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)

The balanced accuracy calculates the average `Recall` for each class.
