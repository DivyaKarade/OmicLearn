# Main
import os, sys
import base64
import itertools
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from itertools import chain

# Sklearn
import sklearn
import sklearn.metrics as metrics
from sklearn.impute import KNNImputer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif, SelectKBest
from sklearn.metrics import roc_curve, plot_roc_curve, auc, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, QuantileTransformer, PowerTransformer
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

# Plotly Graphs
import plotly
import plotly.express as px
import plotly.graph_objects as go

blue_color = '#0068c9'
red_color = '#f63366'
gray_color ='#f3f4f7'

scores = ['roc_auc', 'precision', 'recall', 'f1', 'balanced_accuracy']

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
def load_data(file_buffer, delimiter):
    """
    Load data to pandas dataframe
    """
    df = pd.DataFrame()
    if file_buffer is not None:
        if delimiter == "Excel File":
            df = pd.read_excel(file_buffer)
        elif delimiter == "Comma (,)":
            df = pd.read_csv(file_buffer, sep=',')
        elif delimiter == "Semicolon (;)":
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

@st.cache(persist=True)
def normalize_dataset(X, normalization):
    """
    Normalize data with normalizer
    """

    if normalization == 'None':
        pass
    elif normalization == 'StandardScaler':
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index = X.index)
    elif normalization == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index = X.index)
    elif normalization == 'RobustScaler':
        scaler = RobustScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index = X.index)
    elif normalization == 'PowerTransformer':
        scaler = PowerTransformer()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index = X.index)
    elif normalization == 'QuantileTransformer(Gaussian)':
        scaler = QuantileTransformer(output_distribution='normal')
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index = X.index)
    elif normalization == 'QuantileTransformer(uniform)':
        scaler = QuantileTransformer(output_distribution='uniform')
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index = X.index)
    else:
        raise NotImplementedError('Normalization not implemented')

    return X


def select_features(feature_method, X, y, max_features, random_state):

    if feature_method == 'ExtraTrees':
        clf = ensemble.ExtraTreesClassifier(random_state = random_state)
        clf = clf.fit(X.fillna(0), y)
        feature_importance = clf.feature_importances_
        top_sortindex = np.argsort(feature_importance)[::-1]
        p_values = np.empty(len(feature_importance))
        p_values[:] = np.nan

    elif 'k-best' in feature_method:
        if feature_method == 'k-best (mutual_info_classif)':
            clf = SelectKBest(mutual_info_classif, max_features)
        elif feature_method == 'k-best (f_classif)':
            clf = SelectKBest(f_classif, max_features)
        elif feature_method == 'k-best (chi2)':
            clf = SelectKBest(chi2, max_features)
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
    Creates a Plotly barplot to plot feature importance
    """

    n_features = len(features)
    feature_df = pd.DataFrame(list(zip(features, feature_importance, pvalues)), columns=['Name', 'Feature_importance','P_value'])

    p = px.bar(feature_df, x="Feature_importance", y="Name", color='Name',
            orientation='h',
            hover_data=["Name", "Feature_importance", "P_value"],
            labels={
                    "Feature_importance": "Feature importance",
            },
            height=600,
            title='Top {} features'.format(n_features))
    p.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor= 'rgba(0, 0, 0, 0)')
    p.update_xaxes(showline=True, linewidth=1, linecolor='black')
    p.update_yaxes(showline=True, linewidth=1, linecolor='black')
    return p, feature_df


def impute_nan(X, missing_value, random_state):
    """
    Missing value imputation
    """

    if missing_value == 'Zero':
        X = X.fillna(0)
    elif missing_value =='Mean':
        X = X.fillna(X.mean(axis=0))
    elif missing_value =='Median':
        X = X.fillna(X.median(axis=0))
    elif missing_value == 'None':
        pass
    elif missing_value == 'IterativeImputer':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp_mean = IterativeImputer(random_state=random_state)
        X = pd.DataFrame(imp_mean.fit_transform(X), columns = X.columns)
    elif missing_value == 'KNNImputer':
        imputer = KNNImputer(n_neighbors=2)
        X = pd.DataFrame(imputer.fit_transform(X), columns = X.columns)
    else:
        raise NotImplementedError('Method {} not implemented'.format(missing_value))

    return X

def return_classifier(classifier, random_state, n_estimators, n_neighbors):
    """
    Returns classifier object based on name
    """
    if classifier == 'XGBoost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state = random_state)
    elif classifier == 'LogisticRegression':
        clf = linear_model.LogisticRegression(random_state = random_state, n_jobs=-1)
    elif classifier == 'KNeighborsClassifier':
        clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, algorithm = 'auto', n_jobs=-1)
    elif classifier == 'RandomForest':
        clf = ensemble.RandomForestClassifier(random_state = random_state, n_jobs=-1)
    elif classifier == 'DecisionTree':
        clf = tree.DecisionTreeClassifier(random_state = random_state)
    elif classifier == 'AdaBoost':
        clf = ensemble.AdaBoostClassifier(n_estimators = n_estimators, random_state = random_state)
    elif classifier == 'LinearSVC':
        # TODO: `penalty` and `C` parameter might be retrieved from user
        clf = svm.LinearSVC(penalty='l2', C=2, random_state = random_state)

    return clf


def perform_cross_validation(X, y, classifier, cv_splits, cv_repeats, random_state, n_estimators, n_neighbors, bar):

    clf = return_classifier(classifier, random_state, n_estimators, n_neighbors)
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

        if classifier == "LinearSVC":
            from sklearn.calibration import CalibratedClassifierCV
            # TODO: `method` and `cv` parameter should be considered!
            clf = CalibratedClassifierCV(clf, cv=2) 
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)

        fpr, tpr, cutoffs = roc_curve(y_test, y_score[:, 1])

        for metric_name, metric_fct in scorer_dict.items():
            if metric_name == 'roc_auc':
                _cv_results[metric_name].append(metric_fct(y_test, y_score[:,1]))
            elif metric_name in ['precision', 'recall', 'f1']:
                _cv_results[metric_name].append(metric_fct(y_test, y_pred, zero_division=0))
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


def perform_cohort_validation(X, y, subset, cohort_column, classifier, random_state, n_estimators, n_neighbors, bar):

    clf = return_classifier(classifier, random_state, n_estimators, n_neighbors)

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
            elif metric_name in ['precision', 'recall', 'f1']:
                _cohort_results[metric_name].append(metric_fct(y_test, y_pred, zero_division=0))
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
    "Plotly chart for confusion matrices"

    cm_results = [calculate_cm(*_) for _ in results]
    #also include a summary confusion_matrix
    y_test_ = np.array(list(chain.from_iterable([_[0] for _ in results])))
    y_pred_ = np.array(list(chain.from_iterable([_[1] for _ in results])))

    cm_results.insert(0, calculate_cm(y_test_, y_pred_))
    texts = []
    for j in cm_results:
        texts.append(['{}\n{:.0f} %'.format(_[0], _[1]*100) for _ in zip(*j)])
    cats = ['_'.join(class_0), '_'.join(class_1)]

    x_ = [cats[0], cats[0], cats[1], cats[1]]
    y_ = [cats[0], cats[1], cats[1], cats[0]]

    #  Heatmap
    custom_colorscale = [[0, '#e8f1f7'], [1, "#3886bc"]]
    data = [go.Heatmap(x=x_, y=y_, z=cm_results[step][1], visible=False, hoverinfo='none', colorscale = custom_colorscale) for step in range(len(cm_results))]
    data[0]['visible'] = True

    # Build slider steps
    steps = []
    for i in range(len(data)):
        step = dict(
            method = 'update', 
            args = [
                # Make the i'th trace visible
                {'visible': [t == i for t in range(len(data))]},

                {'annotations' : [
                                dict(
                                    x = x_[k],
                                    y = y_[k],
                                    xref= "x1",
                                    yref= "y1",
                                    showarrow = False,
                                    text = texts[i][k].replace("\n", "<br>"),
                                    font= dict(size=16, color="black")
                                )
                                for k in range(len(x_))
                                ]
                }

            ],
        label = names[i]
        )
        steps.append(step)

    layout_plotly = {
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "True value"},
        "annotations": steps[0]['args'][1]['annotations']
    }
    p = go.Figure(data=data, layout=layout_plotly)

    # Add slider
    sliders = [dict(currentvalue={"prefix": "CV Split: "}, pad = {"t": 72}, active = 0, steps = steps)]
    p.layout.update(sliders=sliders)
    p.update_layout(autosize=False,width=700,height=700)

    return p

def plot_roc_curve_cv(roc_curve_results):
    """Plotly chart for roc curve for cross validation"""

    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    roc_aucs = []
    p = go.Figure()

    for fpr, tpr, threshold in roc_curve_results:
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        p.add_trace(go.Scatter(x=fpr, y=tpr, hoverinfo='skip', mode='lines', line=dict(color=blue_color), showlegend=False,  opacity=0.2))
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

    p.add_trace(go.Scatter(x=base_fpr, y=tprs_lower, fill = None, line_color='gray', opacity=0.2, showlegend=False))
    p.add_trace(go.Scatter(x=base_fpr, y=tprs_upper, fill='tonexty', line_color='gray', opacity=0.2, name='±1 std. dev'))

    hovertemplate = "Base FPR %{x:.2f} <br> %{text}"
    text = ["Upper TPR {:.2f} <br> Mean TPR {:.2f} <br> Lower TPR {:.2f}".format(u, m, l) for u, m, l in zip(tprs_upper, mean_tprs, tprs_lower)]

    p.add_trace(go.Scatter(x=base_fpr, y=mean_tprs, text=text, hovertemplate=hovertemplate, hoverinfo = 'y+text', line=dict(color='black', width=2), name='Mean ROC\n(AUC = {:.2f}±{:.2f})'.format(mean_rocauc, sd_rocauc)))
    p.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=red_color, dash='dash'), showlegend=False))

    p.update_xaxes(showline=True, linewidth=1, linecolor='black')
    p.update_yaxes(showline=True, linewidth=1, linecolor='black')
    p.update_layout(autosize=True,
                    width=800,
                    height=700,
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    yaxis = dict(
                        scaleanchor = "x",
                        scaleratio = 1,
                        zeroline=True,
                        ),
                    # legend=dict(
                    #         yanchor="bottom",
                    #         y=0.01,
                    #         xanchor="right",
                    #         x=0.99)
                    )
    return p


def plot_roc_curve_cohort(roc_curve_results_cohort, cohort_combos):
    """Plotly chart for roc curve for cohort comparison"""

    tprs = []
    #base_fpr = np.linspace(0, 1, 101)
    base_fpr = np.linspace(0, 1, 101)
    roc_aucs = []
    p = go.Figure()
    for idx, res in enumerate(roc_curve_results_cohort):
        fpr, tpr, threshold = res
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        roc_df = pd.DataFrame({'fpr':fpr,'tpr':tpr, 'train':cohort_combos[idx][0], 'test':cohort_combos[idx][1]})
        text= "Train: {} <br>Test: {}".format(cohort_combos[idx][0], cohort_combos[idx][1])
        hovertemplate = "False positive rate: %{x:.2f} <br>True positive rate: %{y:.2f}" + "<br>" + text
        p.add_trace(go.Scatter(x=fpr, y=tpr, hovertemplate=hovertemplate, hoverinfo='all', mode='lines', name='Train on {}, Test on {}, AUC {:.2f}'.format(cohort_combos[idx][0], cohort_combos[idx][1], roc_auc)))
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

    p.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='black', dash='dash'), showlegend=False))
    p.update_xaxes(showline=True, linewidth=1, linecolor='black')
    p.update_yaxes(showline=True, linewidth=1, linecolor='black')
    p.update_layout(autosize=True,
                    width=800,
                    height=700,
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    yaxis = dict(
                        scaleanchor = "x",
                        scaleratio = 1,
                        zeroline=True,
                        )
                    )
    return p


def get_system_report():
    """
    Returns the package versions
    """

    report = {}
    report['proto_learn_version'] = "v0.1.0-dev"
    report['python_version'] = sys.version[:5]
    report['pandas_version'] = pd.__version__
    report['numpy_version'] = np.version.version
    report['sklearn_version'] = sklearn.__version__
    report['plotly_version'] = plotly.__version__

    return report

def get_svg_download_link(p, name='file.svg'):
    """Generates a link for a Plotly chart to be downloaded as SVG"""
    os.makedirs("downloads/", exist_ok=True)
    p.write_image("downloads/"+ name)
    with open("downloads/" + name) as f:
        svg = f.read()
    b64 = base64.b64encode(svg.encode()).decode()
    href = f'<a class="download_link" href="data:image/svg+xml;base64,%s" download="%s" >Download as *.svg</a>' % (b64, name)
    st.markdown(href, unsafe_allow_html=True)

def get_pdf_download_link(p, name='file.pdf'):
    os.makedirs("downloads/", exist_ok=True)
    """Generates a link for a Plotly chart to be downloaded as PDF"""
    p.write_image("downloads/"+ name)
    with open("downloads/" + name, "rb") as f:
        pdf = f.read()
    b64 = base64.encodebytes(pdf).decode()
    href = f'<a class="download_link" href="data:application/pdf;base64,%s" download="%s" >Download as *.pdf</a>' % (b64, name)
    st.markdown(href, unsafe_allow_html=True)

def get_csv_download_link(dataframe, name='file.csv'):
    os.makedirs("downloads/", exist_ok=True)
    """Generates a link for dataframes to be downloaded as CSV"""
    dataframe.to_csv("downloads/"+ name, index=False)
    with open("downloads/" + name, "rb") as f:
        csv = f.read()
    b64 = base64.b64encode(csv).decode()
    href = f'<a class="download_link" href="data:file/csv;base64,%s" download="%s" >Download as *.csv</a>' % (b64, name)
    st.markdown(href, unsafe_allow_html=True)
