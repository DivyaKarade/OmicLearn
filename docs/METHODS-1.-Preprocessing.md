## **Table of Contents**

- [1. Preprocessing](#1-preprocessing)
- [1. 1. Standardization](#1-1-standardization)
  * [1. 1. 1. StandardScaler](#1-1-1-standardscaler)
  * [1. 1. 2. MinMaxScaler](#1-1-2-minmaxscaler)
  * [1. 1. 3. RobustScaler](#1-1-3-robustscaler)
  * [1. 1. 4. PowerTransformer](#1-1-4-powertransformer)
  * [1. 1. 5. QuantileTransformer](#1-1-5-quantiletransformer)
  * [1. 1. 6. Additional Notes](#1-1-6-additional-notes)
- [1. 2. Imputation of missing values](#1-2-imputation-of-missing-values)
  * [1. 2. 1. Zero](#1-2-1-zero)
  * [1. 2. 2. Mean](#1-2-2-mean)
  * [1. 2. 3. Median](#1-2-3-median)
  * [1. 2. 4. IterativeImputer](#1-2-4-iterativeimputer)
  * [1. 2. 5. KNNImputer](#1-2-5-knnimputer)
  * [1. 2. 6. None](#1-2-6-none)
- [1. 3. Data encoding](#1-3-data-encoding)
---

## 1. Preprocessing

A critical step in machine learning (ML) is data preprocessing. It is used to convert data that can have very different scales and exhibit outliers to be more uniform to be used with ML algorithms. Here, we can distinguish three separate aspects that are of particular interest when dealing with proteomics data:

* Standardization
* Imputation of missing values
* Data encoding

This part is primarily based on the [Scikit-learn documentation about preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html), where additional information can be found.

### 1. 1. Standardization

A common requirement for machine learning estimators is that datasets are standardized. The rationale behind this requirement can be easily understood when considering an iterative optimization method, such as the `gradient descent`. Here, the probability of finding a global optimum within a certain number of iterations strongly depends on the step size in each iteration (learning rate). Arguably, when having values outside of a normal range, optimization with a default learning rate is less likely to succeed.

As different classifiers use different optimizers, they are more or less suspective to proper standardization. Scalers for linear transformation and non-linear transformation such as [QuantileTransformer](#1-1-5-quantiletransformer) and  [PowerTransformer](#1-1-4-powertransformer) can be distinguished.

Within Proto Learn, the following options can be selected:

### [1. 1. 1. StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

The option [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) standardizes features by removing the mean and scaling it to unit variance. This is also known as z-Normalization and is widely used in Proteomics. 
The transformation is done according to the following formula:

> z = (x - u) / s
>
> In this formula, while `x` refers to a sample, `u` refers to mean and `s` refers to the standard deviation of the data.

Note that the StandardScaler is very susceptible to outliers as they have a significant influence when calculating the mean and standard deviation.

### [1. 1. 2. MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
Another option to scale data is to transform it according to their minimum and maximum values. Here, data is rescaled so that the minimum corresponds to 0 and the maximum to 1 according to the following formula:

> X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

> X_scaled = X_std * (max - min) + min

Note that, similar to the MinMaxScaler StandardScaler is very susceptible to outliers as they would define the minimum / maximum.

### [1. 1. 3. RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
To achieve a more robust scaling, we can employ the `RobustScaler`. Here, the data is scaled on percentiles and hence not easily influenced by some outliers. More precisely, the median and the `IQR (Interquartile Range) = Q3 (75th Quantile) - Q1 (25th Quantile) ` are used.

### [1. 1. 4. PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)

`PowerTransformer`, can apply a power transformation for each feature in the dataset to make it more Gaussian-like and is useful when dealing with skewed datasets. Here, two options are available: `Yeo-Johnson`, which can work with negative data, and `Box-Cox`, that is limited to positive data.

### [1. 1. 5. QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
`QuantileTransformer` provided by `scikit-learn` uses quantile information to transform features to follow a gaussian distribution (Option `Gaussian output` or a uniform output (Option 'Uniform output`).

### 1. 1. 6. Additional Notes

> Notes: Scikit-learn also provides additional scalers such as the `Normalizer`. We decided not to include this for the analysis of proteomic datasets for now.

An overview of how different scaling methods change the dataset can be found [here](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html):

---

## 1. 2. Imputation of missing values

Proteomic measurements often face the problem that the dataset will have missing values. This is especially the case for DDA acquisition when a precursor is not picked for fragmentation. To use a proteomic dataset with a machine learning optimizer, it is required to develop a strategy to replace the missing values (impute). Here a key challenge is on how the data should be imputed. For regular ML tasks, rows with missing values are often simply deleted; however when applying this to a proteomic dataset, a lot of data would be discarded as the number of missing values is significant. Especially in a clinical context, the imputation of values can be critical as ultimately, this will be the foundation on whether a disease state will be classified or not. Consider the case where an imbalanced dataset exists, and a z-normalization is performed: The mean protein intensity would be zero, this would correspond to the larger class, and when imputing with zeros, one would bias the classification only due to the imputation.

Only some algorithms, such as `xgboost` have implemented methods that can handle missing values and do not need missing value imputation.

This part is primarily based on the [Scikit-learn documentation about imputation](https://scikit-learn.org/stable/modules/impute.html), where additional information can be found.
### 1. 2. 1. Zero
In this option, the missing values are filled with `0`.

### 1. 2. 2. Mean
Using `Mean` for imputation, missing protein values are replaced with the `mean` of the same protein.

### 1. 2. 3. Median
Using `Median` for imputation, missing protein values are replaced with the `median` of the same protein.

### 1. 2. 4. [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer)
The `IterativeImputer` is a more sophisticated approach trying to estimate missing values from other values. This can be very beneficial in a proteomics context as a lot of protein intensities are linearly correlated. Hence, one is, in principle, capable of estimating a protein intensity based on other intensities.

### 1. 2. 5. [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer)
Similar to the `IterativeImputer`, the `KNNImputer` is trying to estimate missing values from existing values. Here, this is done by using a `k-Nearest Neighbors` approach. In brief, a Euclidean distance metric is used to find out the nearest neighbors, and the missing value is estimated by taking the mean of the neighbors.

### 1. 2. 6. None
When selecting None, no missing value imputation is performed. If the dataset exists, only some classifiers that can handle missing values, such as `xgboost` will be selectable.

---
## 1. 3. Data encoding
Another step in ML is that data needs to be encoded. When having categorical data, they need to be transformed. For proteomics data, this is typically unnecessary as we already have the protein intensity, which is a discrete variable. Within Proto Learn, we also allow to use additional features that could be categorical. Whenever a column contains non-numerical values, we use the [label encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) from scikit-learn, which transforms categorical values numerical values (i.e., `male`|`female` will be `0`|`1`).
