**Omic Learn** aims to alleviate access to the latest techniques in machine learning to be used for omics datasets. Here, the tool heavily relies on [scikit-learn](https://scikit-learn.org/stable/) and additionally integrates state-of-the-art algorithms such as xgboost.

A typical machine learning pipeline consists of `Preprocessing`, `Feature Selection`, `Classification` and `Validation` of your method. 

With Omic Learn, you have the possibility to choose from several different choices and explore their effect when analyzing your data. 

Additional information about the individual processing steps can be found here:

- [1. Preprocessing](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-preprocessing)
- [1. 1. Standardization](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-1-standardization)
  * [1. 1. 1. StandardScaler](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-1-1-standardscaler)
  * [1. 1. 2. MinMaxScaler](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-1-2-minmaxscaler)
  * [1. 1. 3. RobustScaler](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-1-3-robustscaler)
  * [1. 1. 4. PowerTransformer](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-1-4-powertransformer)
  * [1. 1. 5. QuantileTransformer](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-1-5-quantiletransformer)
  * [1. 1. 6. Additional Notes](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-1-6-additional-notes)
- [1. 2. Imputation of missing values](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-2-imputation-of-missing-values)
  * [1. 2. 1. Zero](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-2-1-zero)
  * [1. 2. 2. Mean](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-2-2-mean)
  * [1. 2. 3. Median](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-2-3-median)
  * [1. 2. 4. IterativeImputer](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-2-4-iterativeimputer)
  * [1. 2. 5. KNNImputer](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-2-5-knnimputer)
  * [1. 2. 6. None](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-2-6-none)
- [1. 3. Data encoding](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-3-data-encoding)

- [2. Feature selection](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-2.-Feature-selection#2-feature-selection)
   * [2. 1. ExtraTrees](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-2.-Feature-selection#2-1-ExtraTrees)
   * [2. 2. k-best (chi2)](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-2.-Feature-selection#2-2-k-best-chi2)
   * [2. 3. k-best (mutual_info_classif)](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-2.-Feature-selection#2-3-k-best-mutual_info_classif)
   * [2. 4. k-best (f_classif)](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-2.-Feature-selection#2-4-k-best-f_classif)

- [3. Classification](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-classification)
  * [3. 1. AdaBoost](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-1-adaboost)
  * [3. 2. LogisticRegression](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-2-logisticregression)
  * [3. 3. RandomForest](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-3-randomforest)
  * [3. 4. XGBoost](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-4-xgboost)
  * [3. 5. DecisionTree](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-5-decisiontree)
  * [3. 6. KNeighborsClassifier](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-6-kneighborsclassifier)
  * [3. 7. LinearSVC](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-7-linearsvc)


- [4. 1. Cross Validation](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-1-cross-validation)
- [4. 2. Scores](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-2-scores)
  * [4. 2. 1. ROC AUC Score](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-2-1-roc-auc-score)
  * [4. 2. 2. PR AUC Score](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-2-2-pr-auc-score)
  * [4. 2. 3. Precision Score](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-2-3-precision-score)
  * [4. 2. 4. Recall Score](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-2-4-recall-score)
  * [4. 2. 5. F1 Score](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-2-5-f1-score)
  * [4. 2. 6. Balanced Accuracy Score](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-2-6-balanced-accuracy-score)
  * [4. 2. 7. Confusion Matrix](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-2-7-confusion-matrix)
  