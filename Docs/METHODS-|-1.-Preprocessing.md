## **Table of Contents**

- [1. Preprocessing](#1-preprocessing)
  * [1. 1. StandardScaler](#1-1-standardscaler)
  * [1. 2. MinMaxScaler](#1-2-minmaxscaler)
  * [1. 3. MaxAbsScaler](#1-3-maxabsscaler)
  * [1. 4. RobustScaler](#1-4-robustscaler)
  * [1. 5. PowerTransformer](#1-5-powertransformer)
  * [1. 6. QuantileTransformer](#1-6-quantiletransformer)
  * [1. 7. Normalizer](#1-7-normalizer)
  * [1. 8. Comparison](#1-8-Comparison)
- [Sources](#Sources)

---

## 1. Preprocessing
In machine learning (ML), data preprocessing is a crucial step and it is used to convert and set the raw data for making it feasible for the rest of the analysis. Also, some features in the datasets might have very different scales and outliers. For the visualization of data and predictive performance of ML algorithms are affected by these two characteristics. 

In addition, while scalers are linear transformers and display their differences in estimating the parameters which are used, [QuantileTransformer](#1-6-quantiletransformer) and  [PowerTransformer](#1-5-powertransformer) provide non-linear transformations. 

Here are the options listed in Proto Learn under the `Preprocessing` step:

### 1. 1. StandardScaler
In the preprocessing step, standardization is an integral step for data analysis and for many machine learning estimators. 
Here, the `scikit-learn` library offers `StandardScaler` class for standardizing features by removing the mean and scaling to unit variance. 

For feature scaling, here is the equation for how standard score is calculated with `StandardScaler` class:

> z = (x - u) / s
> 
> In this formula, while `x` refers to a sample, `u` refers to mean and `s` refers to the standard deviation of the training samples.

As a note, this method considers `NaNs` as missing values, meaning that they are disregarded in fit, and they are maintained in the process of transformation.

> For technical details of `StandardScaler` class in  `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

### 1. 2. MinMaxScaler
`scikit-learn` library offers `MinMaxScaler` for transforming features by rescaling each feature to a given range such as `[0, 1] (zero-to-one)` as also default. In other words, as an alternative for `StandardScaler`, `MinMaxScaler` also is able to scale the features in between `min` and `max` values (generally between `zero` and `one` (`[0-1]`)). So, for each feature, the maximum absolute values are scaled to the size of the unit with `MinMaxScaler`.

On the other hand,  all inliers in the narrow range `[0, 0.005]` are compressed by this scaling process for the transformed number of households.

Here is the used transformation for `feature_range=(min, max)`:
```
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X_scaled = X_std * (max - min) + min
```

As a note, this method considers `NaNs` as missing values, meaning that they are disregarded in fit, and they are maintained in the process of transformation.

Also, similar to `StandardScaler`, `MinMaxScaler` is also affected by outliers. So, this scaling option enables researchers to have the robustness to tiny standard deviations of features and keep `zero` data points (entries) in sparse data. 

> For technical details of `MinMaxScaler` class in  `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

### 1. 3. MaxAbsScaler
`MaxAbsScaler` performs a scaling process similar to `MinMaxScaler`. However, `MaxAbsScaler` displays a different feature in the process of mapping the values. 
So, in `MaxAbsScaler` provided by `scikit-learn`, the absolute values are mapped in the range of `[0, 1]`. Using a dataset composed of positive-only data, `MaxAbsScaler` performs similarly to `MinMaxScaler`. 

Therefore, likewise `MinMaxScaler`, `MaxAbsScaler` is also very sensitive to large outliers.

As a note, this method considers `NaNs` as missing values, meaning that they are disregarded in fit, and they are maintained in the process of transformation.

> For technical details of `MaxAbsScaler` class in `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).

### 1. 4. RobustScaler
For many ML estimators and datasets, standardization is an important and common requirement. As other scalers do, this standardization process is handled with the help of eliminating the mean and scaling to unit variance. On the other side, if the dataset includes outliers, some methods are affected in a negative way. For these cases, as `RobustScaler` do, median, and `IQR (Interquartile Range) = Q3 (75th Quantile) - Q1 (25th Quantile) ` are used.


Unlike other scalers, `RobustScaler` provided by  `scikit-learn` use statistics which behave robustly to outliers since this scaler lies on percentiles and scaling & centering statistics are computed independently on each feature. So, in other words, `RobustScaler` eliminates the median and scale the dataset for IQR. 

So, then, for the `transform` method, median and IQR are used. 

> For technical details of `RobustScaler` class in  `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

### 1. 5. PowerTransformer

`PowerTransformer`, that belongs to the family of parametric, monotonic transformations for making data more Gaussian-like, is able to apply a power transformation for each feature in the dataset and this is a handful for heteroscedasticity (heteroskedasticity) situations. 

> Here is Heteroscedasticity vs. Homoscedasticity
>
> ![heteroscedasticity vs Homoscedasticity](https://user-images.githubusercontent.com/49681382/87959674-e94c2c80-cabb-11ea-9b2f-ee0134e07bf2.png)
_Sources: WikiWand [Heteroscedasticity](https://www.wikiwand.com/en/Heteroscedasticity), [Homoscedasticity](https://www.wikiwand.com/en/Homoscedasticity)_

For scaler `PowerTransformer`, there are two available methods (transforms):

1. `Yeo-Johnson`
2. `Box-Cox` 

To stabilize variance and make the skewness minimum, the best (optimal) parameter is estimated through maximum likelihood.

>** What is the difference between `Yeo-Johnson` and `Box-Cox`?**
>
> `Box-Cox` in `PowerTransformer` needs input **positive** data, while `Yeo-Johnson` is able to perform the process with **both positive or negative data.**

As a note, for transformed data, `PowerTransformer` applies `zero-mean` and `unit-variance` normalization as default and this method considers `NaNs` as missing values, meaning that they are disregarded in fit, and they are maintained in the process of transformation.

### 1. 6. QuantileTransformer
`QuantileTransformer` provided by `scikit-learn` uses quantile information for transforming each feature independently with a non-linear transformation in which the function of the probability density for each feature is mapped to a uniform dist or in other words, mapped to a range of `[0,1]`. It means that inliers are not distinguished from outliers.
 
Here, in this method, there are two types of output for `output_distribution` parameter:

1. `Gaussian output`
2. `Uniform output`

As a note, for extreme values, saturation artifacts are introduced by the `Gaussian output` non-parametric transformer.

Also, this method considers `NaNs` as missing values, meaning that they are disregarded in fit, and they are maintained in the process of transformation.

> For technical details of `QuantileTransformer` class in  `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html).

> **For the comparing `PowerTransformer `and `QuantileTransformer`**, they both provide non-linear transformations.
>  
> However, while `QuantileTransformer` performs this non-linear transformation where distances between marginal outliers and inliers are shrunk,
> 
> `PowerTransformer` performs it where data is mapped to a normal distribution for stabilizing variance and making the skewness minimum.

### 1. 7. Normalizer
`Normalizer` is able to rescale the vector by normalizing samples independently to unit norm.

Here, `Normalizer` rescales each sample with at least one non zero component independently of other samples. Thus, it becomes its norm equal one.

As a note, this `Normalizer` estimator provided by `scikit-learn` is stateless, meaning that this method does not take a key role however it is useful if it is used in the pipelines.  

> For technical details of `Normalizer` class in  `scikit-learn` , please visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html).

### 1. 8. Comparison
![OmicEra Proto Learn preprocessing](https://user-images.githubusercontent.com/49681382/87949483-bea7a700-caae-11ea-91b4-7b158fd1f869.png)
_Source: [Scikit-learn | Plot All Scalers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#)_

## Sources
[https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html#)

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)

[http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html](http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html)

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)

[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)

[https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)