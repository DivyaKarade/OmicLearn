## **Table of Contents**

- [2. Feature selection](#2-feature-selection)
  * [2. 1. Decision Tree](#2-1-decision-tree)
  * [2. k-best (mutual_info_classif)](#2-2-k-best-mutual_info_classif)
  * [2. 3. k-best (f_classif)](#2-3-k-best-f_classif)
- [Source](#source)

---

## 2. Feature selection
In Machine Learning (ML), feature selection is one of the most important issues. Since this step is important, the `scikit` library also offers a module namely `feature_selection` implementing algorithms for feature selection. 

So, this module can be used for selecting the features, boosting the performance on very high-dimensional datasets, reducing the number of features (dimensionality) in given data, or improving accuracy scores of estimators.

**So, there are lots of advantages of feature selection:**

- It enables researchers to remove irrelevant features that act as noise and to improve the performance of the model.
- It enables researchers to avoid `overfitting`.
- It enables researchers to debug it easily and to make it easier to understand.

![feat_sel](https://user-images.githubusercontent.com/49681382/89779396-eae79e00-db17-11ea-88e5-b206436596d1.png)

_Source: [An Introduction to Feature Selection](https://towardsdatascience.com/an-introduction-to-feature-selection-dd72535ecf2b)_

Also, the `scikit-learn` library includes `Univariate feature selection` methods using univariate statistical tests and one of them is `SelectKBest` that will be explained on this page.

Here is an example work for computing p-values when univariate feature selection is applied to the dataset after some noisy features are added.

In this chart, the p-values are plotted for each feature together with the weights of SVMs.

![UFS Plot](https://user-images.githubusercontent.com/49681382/89784248-060adb80-db21-11ea-87e8-5afee37dcdd3.png)

_Source: [Univariate Feature Selection](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#univariate-feature-selection)_


### 2. 1. Decision Tree
The `scikit-learn` library also offers a module called `tree` that includes decision tree-based models for both classification and regression tasks.

So, `Decision Tree (DT)` is an example method for non-parametric supervised learning.  The main idea behind this `Decision Tree` is to build a model for calculating the value of a target variable.

Here, in `sklearn.tree` module, there is a method called `DecisionTreeClassifier`.

> For technical details of `DecisionTreeClassifier` class in `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier).

---

Besides `Decision Tree`, `SelectKBest` is able to select the features in the dataset based on the `k` highest scores.

Also, `SelectKBest` is available for two categories:

- **For regression** problems: `f_regression`, `mutual_info_regression`
- **For classification** problems: `chi2`, `f_classif`, `mutual_info_classif`

### 2. 2 k-best (mutual_info_classif)
Here, with `feature_selection.mutual_info_classif`, mutual information for a discrete target variable is estimated. 

Also, this function is based on nonparametric methods with k-nearest neighbors distances and this method also can be used for univariate features selection,

> For technical details of `feature_selection.mutual_info_classif` class in `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.mutual_info_classif).

### 2. 3. k-best (f_classif)
For the given data, the method namely `feature_selection.f_classif` is able to calculate the ANOVA (ANalysis Of VAriance) F-value.

In other words, for classification tasks, this method computes ANOVA F-value between label/feature.

> **As an important note**, if the data is sparse in feature selection step, these 3 methods `chi2`, `mutual_info_regression`, `mutual_info_classif` will deal with the data without making it dense.

> For technical details of `feature_selection.f_classif` class in `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif).

## Source
[https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)

[https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)

[https://towardsdatascience.com/an-introduction-to-feature-selection-dd72535ecf2b](https://towardsdatascience.com/an-introduction-to-feature-selection-dd72535ecf2b)

