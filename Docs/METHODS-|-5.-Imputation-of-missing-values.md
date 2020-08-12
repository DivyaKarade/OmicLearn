## **Table of Contents**

- [5. Imputation of missing values](#5-imputation-of-missing-values)
  * [5. 1. Zero](#5-1-zero)
  * [5. 2. Mean](#5-2-mean)
  * [5. 3. Median](#5-3-median)
  * [5. 4. IterativeImputer](#5-4-iterativeimputer)
  * [5. 5. KNNImputer](#5-5-knnimputer)
  * [5. 6. None](#5-6-none)
  * [5. 7. Comparison](#5-7-comparison)
- [Sources](#sources)

---

## 5. Imputation of missing values
In machine learning (ML), one of the most seen problems is having missing/blank (`NaNs`) values in the dataset given.  

However, for using the estimators or algorithms, it is needed to fill these missing values because these estimators or methods assume that all values are numeric in the array or column. 

One basic solution might be deleting the rows that contain missing values however it ends up losing your samples/observations. 

So, the best strategy might be imputing those missing values with several options.

### 5. 1. Zero
In this option, the missing values are filled by `Zero (0)`.

### 5. 2. Mean
Using `Mean` for imputation, the missing values are replaced with the `Mean` of the column.

### 5. 3. Median
Using `Median` for imputation, the missing values are replaced with the `Median` of the column.

### 5. 4. IterativeImputer
Compared to other methods, `IterativeImputer` is more sophisticated. 

Basically, `IterativeImputer` is a multivariate imputer and it is able to estimate the features by looking at all the others.

In other words, it imputes the missing values by using the function of other features after modeling them with missing values and it uses a Round-Robin fashion.

> For technical details of `IterativeImputer` class in `scikit-learn`, please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer).

### 5. 5. KNNImputer
With `KNNImputer`, the missing values are imputed by using using the `k-Nearest Neighbors` approach. 

In this method, Euclidean distance metric is used to find out the nearest neighbors. Then, the missing values are replaced with the mean value of the nearest neighbors exist in the dataset.

> For technical details of `KNNImputer` class in `scikit-learn`, please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer).

### 5. 6. None
It does not impute the missing values.

### 5. 7. Comparison

Here is the comparison of these imputation methods:

![imputation_wiki](https://user-images.githubusercontent.com/49681382/89999078-0331f700-dc97-11ea-9565-d0ca31f77128.png)

_Source: [Plot Comparison for Imputing Missing Values](https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#imputing-missing-values-before-building-an-estimator)_

## Sources
[https://scikit-learn.org/stable/modules/impute.html](https://scikit-learn.org/stable/modules/impute.html)
