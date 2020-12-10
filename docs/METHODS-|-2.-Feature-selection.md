## **Table of Contents**

- [2. Feature selection](#2-feature-selection)
   * [2. 1. ExtraTrees](#2-1-ExtraTrees)
   * [2. 2. k-best (chi2)](#2-2-k-best-chi2)
   * [2. 3. k-best (mutual_info_classif)](#2-3-k-best-mutual_info_classif)
   * [2. 4. k-best (f_classif)](#2-4-k-best-f_classif)


---

## 2. Feature selection

Feature selection is a crucial part when building a machine learning pipeline. This refers to making only a subset of data available for the machine learning classifier, i.e., only taking ten proteins. For training, we would like to select only features that contribute to our prediction so that the classifier does not try to learn from unrelated features and ultimately will generalize well. This is especially important for a clinical proteomics setup as we can choose for a large number of features (protein signals) while often having only small sample numbers. Reducing the number of proteins will also help us identify core players contributing to a classification model. Within OmicLearn, several algorithms are implemented that allow reducing the number of features.

> **Note:** Proteomics features can be highly correlated (Multicollinear). This leads to the problem that features importance can be somewhat ambiguous, i.e., removing a protein with high feature importance does not necessarily decrease the overall accuracy if the machine learning classifier can extract the information from linearly correlated protein signals.

### [2. 1. ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

One way to reduce the number of features is by using a randomized decision trees (a.k.a. extra-trees) approach, where a classifier is trained to distinguish the classes, and the features with the highest importance are selected.  

---

Another way for feature selection is by using the `SelectKBest` strategy. Here, features are selected based on the `k` highest scores. Here, we have the following options available: `chi2`, `f_classif`, `mutual_info_classif`.

### [2. 2. k-best (chi2)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)
Here, the chi-squared stats between features and the class is used as the k-score.

### [2. 3. k-best (mutual_info_classif)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.mutual_info_classif)

Here, an estimate for the mutual information of variables is used as the k-score.

### [2. 4. k-best (f_classif)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)

Here,  an estimate for the ANOVA (ANalysis Of VAriance) F-value is used as the k-score.