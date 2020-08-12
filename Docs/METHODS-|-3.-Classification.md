## **Table of Contents**

- [3. Classification](#3-classification)
	- [3. 1. AdaBoost](#3-1-adaboost)
	- [3. 2. LogisticRegression](#3-2-logisticregression)
	- [3. 3. RandomForest](#3-3-randomforest)
	- [3. 4. XGBoost](#3-4-xgboost)
	- [3. 5. DecisionTree](#3-5-decisiontree)
- [Sources](#sources)

---

## 3. Classification

In Machine Learning (ML), classification can be defined as a predictive modeling problem in which the label of classes are predicted for given data points such as classifying the mails `spam` or `not spam`. Sometimes, `classes` are also called as ` targets or labels or categories`.

Here, the `scikit-learn` library provides several classification methods.

![Class.](https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)

_Source: [Scikit-learn Classifier Comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html?highlight=classifiers#)_


### 3. 1. AdaBoost
The main principle behind `AdaBoost` is to fit a sequence of weak learners on repeatedly modified versions of the data points and it is a successful boosting algorithm to use in binary classification problems.

So, `AdaBoost ` can be used for boosting any machine learning algorithm performance and the best use case is using with weak learners. 
Also, `AdaBoost` adds weak models sequentially and trained using the weighted training dataset.

![ProtoLearn_Boosting](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1542651255/image_2_pu8tu6.png)

_Source: [DataCamp AdaBoost Classifier](https://www.datacamp.com/community/tutorials/adaboost-classifier-python)_

> Weak learners are the models that have slightly better achievement compared to random chance.

In addition, for classification and regression problems, `AdaBoost` can be preferred.

In `AdaBoost`, there are 3 important issues regarding data preparation:

**1. Data Quality**

Since the ensemble method tries to correct misclassifications in the training dataset, the training dataset should have high-quality.

**2. Noisy Data:**

Noise in output variables can raise problems. So, these are should be removed.

**3. Outliers:**

Outliers also should be removed from the training dataset since they can force the ensemble down

> For technical details of `AdaBoost` class in  `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/ensemble.html#adaboost).

### 3. 2. LogisticRegression
`LogisticRegression` provided by `scikit-learn` is a type of linear model for classification rather than regression. For `LogisticRegression`, there are some other aliases such as `logit regression`, `maximum-entropy classification (MaxEnt)` or `the log-linear classifier`.

In `LogisticRegression`, a logistic function is used for modeling the probabilities for describing the possible outcomes of a single trial.

> **What is a logistic function?**
> 
> The `logistic function `(a.k.a. `sigmoid function`) is developed for describing the properties of population growth in ecology. It rises quickly and is able to max out at the carrying capacity. Also, the sigmoid function is used for mapping predicted values to probabilities. 
>
> ![Logit](https://ml-cheatsheet.readthedocs.io/en/latest/_images/sigmoid.png)
> 
> Source for [the image](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#types-of-logistic-regression)


There are several types of logistic regressions:

1. Binary (Pass/Fail)
2. Multi (Cats, Dogs, Sheep)
3. Ordinal (Low, Medium, High)

Similar to `AdaBoost`, there are also some important points for `LogisticRegression`:

1. Remove Noise

Please, be sure that you removed outliers and misclassified instances from your training dataset since this method assumes no error in the output variable (y).

2. Remove Correlated Inputs

Consider removing highly correlated inputs since the model might overfit if there are multiple highly-correlated inputs.

3. Failing to converge

The model might fail if the data is very sparse and multiple highly-correlated inputs exist in the dataset.

> For technical details of `LogisticRegression` class in  `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression).


### 3. 3. RandomForest

`RandomForest` or `random decision forests` are a type of ensemble learning method for classification, regression. So, `RandomForest` is able to fit a number of decision tree classifiers with several **randomly selected** sub-samples from the dataset and is able to improve accuracy and control the overfitting by using averaging. 

Here is how this algorithm works:

1. It takes randomly selected sub-samples from the training dataset

2. Create a decision tree for each sample and make predictions

3. The voting process for each prediction

4. Select the most voted prediction as final one

![rfc-datacamp](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1526467744/voting_dnjweq.jpg)

_Source: [DataCamp | RFC](https://www.datacamp.com/community/tutorials/random-forests-classifier-python)_

In the `RandomForest` method, there will be several decision trees with different sizes and different branches.

![RFC](https://www.globalsoftwaresupport.com/wp-content/uploads/2018/02/ggff5544hh.png)

_Source: [Random Forest Classifer](https://www.globalsoftwaresupport.com/random-forest-classifier/)_



### 3. 4. XGBoost
`XGBoost (eXtreme Gradient Boosting)` is also one of the most popular ML algorithms for both regression or classification tasks since it offers a wide variety of tuning parameters and great performance and speed compared to others. 

XGBoost (Extreme Gradient Boosting) is a member of boosting algorithms family and it is able to perform many calculations fast and accurately with parallel tree boosting (a.k.a GBDT or GBM).

> So, what is Boosting basically?
>
> 'Boosting' term points some algorithms that are able to transform weak learners to strong ones. 
> The idea behind boosting is that it trains weak learners sequentially. So, then, the weights are given to the outcomes of the model based on the previous instant's outcomes. 
> ![BoostingIlistr.](https://upload.wikimedia.org/wikipedia/commons/b/b5/Ensemble_Boosting.svg)
>
> _Source for the [illustration of boosting algorithm](https://en.wikipedia.org/wiki/File:Ensemble_Boosting.svg)_



Also, note that `XGBoost` is willing to fill the missing values in the dataset you provided.

> For technical details of `XGBoost` , please visit [here](https://xgboost.readthedocs.io/en/latest/python/).

### 3. 5. DecisionTree
Decision trees are used in predictive modeling approaches in ML. In this algorithm, flowchart-like tree structure and in this scheme, there are several internal decision nodes (which resemble feature), the branch (resemble a rule), and leaf nodes (resemble the outcomes).

![DTA](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1545934190/1_r5ikdb.png) 

_Source: [DataCamp | Decision Trees](https://www.datacamp.com/community/tutorials/decision-tree-classification-python)_

So, what is the **difference between Random Forests and Decision Trees**?

Random forests contain several decision trees.

While `RandomForest` is able to prevent overfitting by randomly selecting the subsets, deep decision trees face with overfitting.

Decision trees are faster compared to the random forests in terms of computationally.

While `random forests` are difficult to interpret, a decision tree is easy to interpret.

For technical details of `DecisionTree` class in  `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

## Sources
[https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html?highlight=classifiers#)

[https://machinelearningmastery.com/types-of-classification-in-machine-learning/](https://machinelearningmastery.com/types-of-classification-in-machine-learning/)

[https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/](https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/)

[https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

[https://machinelearningmastery.com/logistic-regression-for-machine-learning/](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)

[https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#types-of-logistic-regression](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#types-of-logistic-regression)

[https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)

[https://scikit-learn.org/stable/supervised_learning.html](https://scikit-learn.org/stable/supervised_learning.html)

[https://scikit-learn.org/stable/modules/ensemble.html#adaboost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

[https://www.datacamp.com/community/tutorials/xgboost-in-python](https://www.datacamp.com/community/tutorials/xgboost-in-python)

[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

[https://www.datacamp.com/community/tutorials/random-forests-classifier-python](https://www.datacamp.com/community/tutorials/random-forests-classifier-python)

[https://medium.com/greyatom/a-quick-guide-to-boosting-in-ml-acf7c1585cb5](https://medium.com/greyatom/a-quick-guide-to-boosting-in-ml-acf7c1585cb5)

[https://www.datacamp.com/community/tutorials/decision-tree-classification-python](https://www.datacamp.com/community/tutorials/decision-tree-classification-python)
