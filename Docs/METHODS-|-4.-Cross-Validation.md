## **Table of Contents**

- [4. Cross Validation](#4-cross-validation)
- [Sources](#sources)

---

## 4. Cross Validation
It is needed to evaluate the performance of Machine Learning (ML) models. So, for this purpose, they should be tested with unseen data. Then, one can say that the model is `underfitting`, `overfitting`, or `well generalized`. 

In addition, `overfitting ` and `underfitting ` are the two biggest reasons behind the poor performance of ML methods.

So, for the difference between the `underfitting` and `overfitting`:

If the ML model does not capture the trend of the data or generalize the given data, it is called as `underfitting`.
On the other side, if the model fits the given (training) data too well and close, `overfitting` occurs and it negatively reflects to generalize it. 

Here is the graph comparison:

![ML_Fits](https://user-images.githubusercontent.com/49681382/88183263-e8410980-cc39-11ea-9fc0-8caa9dc260ed.png)

_Source: [AWS Docs | Model Fitting](https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html)_

So, `Cross Validation (CV)` concept, here, is used for testing the effectiveness of an ML model by resampling our limited data.
For this purpose, there are several CV techniques.

- One of widely used CV approach is called `Train-Test Split Approach`.

In this method, the data is randomly splitted into two parts by `train_test_split` function in `scikit-learn` library: 

1. For training purpose 

2.For testing purpose

Here, for splitting, there are two widely used ratios that are `70:30` and `80:20`. 

- Another popular CV approach is called ` K Fold Cross Validation`:
In this approach, each data point is used for both training and testing purposes, meaning that it can end up with a less biased model.

Here is the representative image for K-Fold CV approach:

![KFold_cv](https://user-images.githubusercontent.com/49681382/88184493-723da200-cc3b-11ea-9a83-305bf5bc2fa6.png)

_Source for [K-Fold CV Image](https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f)_

So, in Proto Learn, you are also able to set your own numbers for both `cv_splits` and `cv_repeats`.

## Sources

[https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)

[https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f](https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f)