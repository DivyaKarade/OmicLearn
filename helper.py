import sys
import pandas as pd
import streamlit as st
from bokeh.plotting import figure

import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, QuantileTransformer, PowerTransformer, Normalizer

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
import numpy as np

from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, plot_roc_curve, auc, plot_confusion_matrix

import sklearn.metrics as metrics

from bokeh.models.glyphs import Text

from bokeh.plotting import ColumnDataSource
from bokeh.models import CustomJS, Slider, Div
from bokeh.layouts import column, row

from bokeh.models import HoverTool
import itertools
from bokeh.palettes import Dark2_5 as palette


blue_color = '#0068c9'
red_color = '#f63366'
gray_color ='#f3f4f7'

scores = ['roc_auc', 'precision', 'recall', 'f1', 'balanced_accuracy']
#score = st.sidebar.selectbox("Optimiziation metric", scores)

scorer_dict = {}
scorer_dict = {metric:metric+'_score' for metric in scores}
scorer_dict = {key: getattr(metrics, metric) for key, metric in scorer_dict.items()}


def make_recording_widget(f, widget_values):
    """Return a function that wraps a streamlit widget and records the
    widget's values to a global dictionary.
    """
    def wrapper(label, *args, **kwargs):
        widget_value = f(label, *args, **kwargs)
        widget_values[label] = widget_value
        return widget_value

    return wrapper

@st.cache(persist=True)
def load_data(file_buffer):
    """
    Load data to pandas dataframe
    """
    df = pd.DataFrame()
    if file_buffer is not None:
        try:
            df = pd.read_excel(file_buffer)
        except:
            df = pd.read_csv(file_buffer, sep=';')
    return df

@st.cache(persist=True)
def transform_dataset(subset, additional_features, proteins):
    """
    Transforms data with label encoder
    """

    transformed_columns = []

    for _ in additional_features:
        if subset[_].dtype in [np.dtype('O'), np.dtype('str')]:
            le = LabelEncoder()
            transformed = le.fit_transform(subset[_])
        else:
            transformed = subset[_]
        transformed_columns.append(pd.DataFrame(transformed, columns=[_], index=subset[_].index))

    if len(transformed_columns) > 1:
        transformed = pd.concat(transformed_columns, axis=1)
    elif len(transformed_columns) == 1:
        transformed = transformed_columns[0]
    else:
        transformed = []

    # Join with proteins
    protein_features = subset[proteins].astype('float')

    if len(transformed) >= 1 and len(protein_features) >= 1:
        X = pd.concat([protein_features, transformed], axis=1)
    else:
        if len(transformed) == 0 :
            X = protein_features
        elif len(protein_features) == 0:
            X = transformed
        else:
            pass
    return X

#['None', 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer(Gaussian)','QuantileTransformer(uniform)','Normalizer']
@st.cache(persist=True)
def normalize_dataset(X, normalization):
    """
    Normalize data with normalizer
    """

    if normalization == 'None':
        pass
    elif normalization == 'Standard':
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif normalization == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif normalization == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif normalization == 'RobustScaler':
        scaler = RobustScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif normalization == 'PowerTransformer':
        scaler = PowerTransformer()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif normalization == 'QuantileTransformer(Gaussian)':
        scaler = QuantileTransformer(output_distribution='normal')
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif normalization == 'QuantileTransformer(uniform)':
        scaler = QuantileTransformer(output_distribution='uniform')
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif normaliation == 'Normalizer':
        scaler = Normalizer()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        raise NotImplementedError('Normalization not implemented')

    return X


def select_features(feature_method, X, y, max_features):

    if feature_method == 'DecisionTree':
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X.fillna(0), y)
        feature_importance = clf.feature_importances_
        top_sortindex = np.argsort(feature_importance)[::-1]
        p_values = np.empty(len(feature_importance))
        p_values[:] = np.nan

    elif 'k-best' in feature_method:
        if feature_method == 'k-best (mutual_info)':
            clf = SelectKBest(mutual_info_classif, max_features)
        elif feature_method == 'k-best (f_classif)':
            clf = SelectKBest(f_classif, max_features)
        else:
            raise NotImplementedError('Feature method {} not implemented.'.format(feature_method))
        clf = clf.fit(X.fillna(0), y)
        feature_importance = clf.scores_
        p_values = clf.pvalues_
        if p_values is None:
            p_values = np.empty(len(feature_importance))
            p_values[:] = np.nan
        top_sortindex = np.argsort(feature_importance)[::-1]

    else:
        raise NotImplementedError('Method {} not implemented.'.format(feature_method))

    top_features = X.columns[top_sortindex][:max_features][::-1].tolist()
    top_features_importance = feature_importance[top_sortindex][:max_features][::-1]
    top_features_pvalues = p_values[top_sortindex][:max_features][::-1]

    #Some but not all return p-values

    return top_features, top_features_importance, top_features_pvalues

def plot_feature_importance(features, feature_importance, pvalues):
    """
    Creates a bokeh barplot to plot feature importance
    """
    n_features = len(features)

    p = figure(y_range=features, title='Top {} features'.format(n_features), tooltips =[("Name", "@Name"), ("Importance", "@Feature_importance"),("p-value", "@P_value")], tools="pan,reset,save,wheel_zoom")

    feature_df = pd.DataFrame(list(zip(features, feature_importance, pvalues)), columns=['Name', 'Feature_importance','P_value'])
    p.hbar(right='Feature_importance', y='Name', source=feature_df, height=0.9, line_color='white', color=red_color)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.x_range.start = 0
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    return p


def impute_nan(X, missing_value):
    """
    Missing value imputation
    """
                #['SimpleImputer (Zero)', 'SimpleImputer (Mean)', 'SimpleImputer (Median)', 'IterativeImputer', 'KNNImputer', 'None']
    if missing_value == 'Zero':
        X = X.fillna(0)
    elif missing_value =='Mean':
        X = X.fillna(X.mean(axis=0))
    elif missing_value =='Median':
        X = X.fillna(X.median(axis=0))
    elif missing_value == 'None':
        pass
    elif missing_value == 'Mean of column':
        X.fillna(X.mean())
    else:
        raise NotImplementedError('Method {} not implemented'.format(missing_value))

    return X

def return_classifier(classifier, random_state):
    """
    Returns classifier object based on name
    """
    if classifier == 'XGBoost':
        clf = XGBClassifier(random_state = random_state)
    elif classifier == 'LogisticRegression':
        clf = linear_model.LogisticRegression(random_state = random_state)
    elif classifier == 'RandomForest':
        clf = ensemble.RandomForestClassifier(random_state = random_state)
    elif classifier == 'DecisionTree':
        clf = tree.DecisionTreeClassifier(random_state = random_state)
    elif classifier == 'AdaBoost':
        clf = ensemble.AdaBoostClassifier(random_state = random_state)

    return clf


def perform_cross_validation(X, y, classifier, cv_splits, cv_repeats, random_state, bar):

    clf = return_classifier(classifier, random_state)
    rskf = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)

    roc_curve_results = []
    split_results = []

    _cv_results = {}
    _cv_results['num_feat'] = []
    _cv_results['n_obs'] = []
    _cv_results['n_class_0'] = []
    _cv_results['n_class_1'] = []
    _cv_results['class_ratio'] = []

    for metric_name, metric_fct in scorer_dict.items():
        _cv_results[metric_name] = []

    for i, (train_index, test_index) in enumerate(rskf.split(X,y)):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)

        fpr, tpr, cutoffs = roc_curve(y_test, y_score[:, 1])

        for metric_name, metric_fct in scorer_dict.items():
            if metric_name == 'roc_auc':
                _cv_results[metric_name].append(metric_fct(y_test, y_score[:,1]))
            else:
                _cv_results[metric_name].append(metric_fct(y_test, y_pred))

        _cv_results['num_feat'].append(X.shape[-1])
        _cv_results['n_obs'].append(len(y))
        _cv_results['n_class_0'].append(np.sum(y))
        _cv_results['n_class_1'].append(np.sum(~y))
        _cv_results['class_ratio'].append(np.sum(y)/len(y))

        roc_curve_results.append((fpr, tpr, cutoffs))
        split_results.append((y_test.values, y_pred))

        bar.progress((i+1)/(cv_splits*cv_repeats))

    return _cv_results, roc_curve_results, split_results


def perform_cohort_validation(X, y, subset, cohort_column, classifier, random_state, bar):

    clf = return_classifier(classifier, random_state)

    roc_curve_results_cohort = []
    cohort_results = []

    _cohort_results = {}
    _cohort_results['num_feat'] = []
    _cohort_results['n_obs'] = []
    _cohort_results['n_class_0'] = []
    _cohort_results['class_ratio'] = []

    for metric_name, metric_fct in scorer_dict.items():

        _cohort_results[metric_name] = []


    cohorts = subset[cohort_column].unique().tolist()
    cohort_combos = []
    for c_1 in cohorts:
        for c_2 in cohorts:
            if c_1 != c_2:
                cohort_combos.append((c_1, c_2))

    roc_curve_results_cohort = []

    for i, cohort_combo in enumerate(cohort_combos):

        c_1, c_2 = cohort_combo

        train_index = subset[cohort_column] == c_1
        test_index = subset[cohort_column] == c_2

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)

        fpr, tpr, cutoffs = roc_curve(y_test, y_score[:, 1])

        for metric_name, metric_fct in scorer_dict.items():
            if metric_name == 'roc_auc':
                _cohort_results[metric_name].append(metric_fct(y_test, y_score[:,1]))
            else:
                _cohort_results[metric_name].append(metric_fct(y_test, y_pred))

        _cohort_results['num_feat'].append(X.shape[-1])
        _cohort_results['n_obs'].append(len(y))
        _cohort_results['n_class_0'].append(np.sum(y))
        _cohort_results['class_ratio'].append(np.sum(y)/len(y))

        roc_curve_results_cohort.append((fpr, tpr, cutoffs))
        cohort_results.append((y_test.values, y_pred))

        bar.progress((i+1)/len(cohort_combos))

    return _cohort_results, roc_curve_results_cohort, cohort_results, cohort_combos

def calculate_cm(y_test, y_pred):
    """
    Calculate confusion matrix
    """

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(y_test)):
        if y_test[i] == y_pred[i] ==True:
           tp += 1
        if y_pred[i] == True and y_test[i] == False:
           fp += 1
        if y_test[i] == y_pred[i] == False:
           tn += 1
        if y_pred[i]== False and y_test[i] == True:
           fn += 1

    tpr = tp/(tp+fn)
    fpr = 1-tpr
    tnr = tn/(tn+fp)
    fnr = 1-tnr

    return (tp, fp, tn, fn), (tpr, fpr, tnr, fnr)

def plot_confusion_matrices(class_0, class_1, results, names):

    cm_results = [calculate_cm(*_) for _ in results]
    texts = []
    for j in cm_results:
        texts.append(['{}\n{:.0f} %'.format(_[0], _[1]*100) for _ in zip(*j)])

    cats = ['_'.join(class_0), '_'.join(class_1)]

    x_ = [cats[0], cats[0], cats[1], cats[1]]
    y_ = [cats[0], cats[1], cats[1], cats[0]]

    slider = Slider(start=0, end=len(results)-1, value=0, step=1, title='')

    p = figure(x_range=cats, y_range=cats[::-1])

    div = Div(text= names[0], style={'font-size': '100%', 'color': 'black'})

    source = ColumnDataSource(data=dict(x=x_, y=y_, alpha = cm_results[0][1], text = texts[0]))

    callback = CustomJS(args=dict(source=source, div=div, slider=slider, cm_results = cm_results, texts=texts, names=names),
                    code="""

    var data = source.data;
    var x = data['x'];
    var y = data['y'];
    var text = data['text'];
    var alpha = data['alpha'];
    div.text = names[slider.value]

    for (var i = 0; i < x.length; i++) {
        text[i] = texts[slider.value][i];
        alpha[i] = cm_results[slider.value][1][i];
    };

    source.change.emit();
""")

    p.rect(x='x', y='y', alpha = 'alpha', source=source, width=1, height=1, line_color=None)

    p.xgrid.visible = False
    p.ygrid.visible = False

    p.xaxis.axis_label = 'Predicted label'
    p.yaxis.axis_label = 'True label'

    glyph = Text(x="x", y="y", text="text", text_color="black", text_align='center', text_baseline='middle')
    p.add_glyph(source, glyph)

    slider.js_on_change('value', callback)

    layout = column(div, slider, p)

    return layout

def plot_roc_curve_cv(roc_curve_results):
    """
    Plot roc curve for cross validation

    """

    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    roc_aucs = []
    hover = HoverTool(names=["mean"])

    p = figure(tooltips =[("False positive rate", "@base_fpr"),("True positive rate, upper", "@upper"), ("True positive rate, mean", "@mean_tprs"), ("True positive rate, lower", "@lower")], tools=[hover,"pan,reset,save,wheel_zoom"])

    for fpr, tpr, threshold in roc_curve_results:
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        p.line(fpr, tpr, color=blue_color, alpha=0.1)

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0]=0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = mean_tprs + std
    tprs_lower = mean_tprs - std

    mean_rocauc = np.mean(roc_aucs).round(2)
    sd_rocauc = np.std(roc_aucs).round(2)

    roc_df = pd.DataFrame({'base_fpr':base_fpr,'mean_tprs':mean_tprs,'lower':tprs_lower,'upper':tprs_upper})

    p.varea(base_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.5, legend = '±1 std. dev')
    p.line(x = 'base_fpr', y='mean_tprs', source = roc_df, color='black', line_width=2, name = 'mean', legend = 'Mean ROC\n(AUC = {:.2f}±{:.2f})'.format(mean_rocauc, sd_rocauc))
    p.line([0, 1], [0, 1], line_color =red_color, line_dash ='dashed')

    p.legend.location = "bottom_right"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.axis_label = 'False Positive Rate'
    p.yaxis.axis_label = 'True Positive Rate'

    return p


def plot_roc_curve_cohort(roc_curve_results_cohort, cohort_combos):
    """
    Plot roc curve for cohort comparison
    """
    p = figure(tooltips =[("False positive rate", "@fpr"),("True positive rate", "@tpr"),("Train","@train"),("Test","@test")], tools=["pan,reset,save,wheel_zoom"])

    tprs = []
    #base_fpr = np.linspace(0, 1, 101)
    base_fpr = np.linspace(0, 1, 101)
    roc_aucs = []

    colors = itertools.cycle(palette)
    for idx, res in enumerate(roc_curve_results_cohort):
        fpr, tpr, threshold = res
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        roc_df = pd.DataFrame({'fpr':fpr,'tpr':tpr, 'train':cohort_combos[idx][0], 'test':cohort_combos[idx][1]})

        p.line(x = 'fpr', y='tpr', source = roc_df, color = next(colors), alpha=1, legend = 'Train on {}, Test on {}, AUC {:.2f}'.format(cohort_combos[idx][0], cohort_combos[idx][1], roc_auc))

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0]=0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = mean_tprs + std
    tprs_lower = mean_tprs - std

    mean_rocauc = np.mean(roc_aucs).round(2)
    sd_rocauc = np.std(roc_aucs).round(2)

    roc_df = pd.DataFrame({'base_fpr':base_fpr,'mean_tprs':mean_tprs,'lower':tprs_lower,'upper':tprs_upper})

    p.line([0, 1], [0, 1], line_color =red_color, line_dash ='dashed')
    p.legend.location = "bottom_right"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.axis_label = 'False Positive Rate'
    p.yaxis.axis_label = 'True Positive Rate'

    return p


def get_system_report():
    """
    Returns the package versions
    """

    report = {}

    report['python_version'] = sys.version[:5]
    report['pandas_version'] = pd.__version__
    report['sklearn_version'] = sklearn.__version__

    return report
