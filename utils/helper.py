# Main
import os, sys
import base64
import numpy as np
import pandas as pd
import streamlit as st
from itertools import chain

# Sklearn
import sklearn
import sklearn.metrics as metrics
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import svm, tree, linear_model, neighbors, ensemble
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif, SelectKBest
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, QuantileTransformer, PowerTransformer

# Plotly Graphs
import plotly
import plotly.express as px
import plotly.graph_objects as go

blue_color = '#0068c9'
red_color = '#f63366'
gray_color ='#f3f4f7'

scores = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1', 'balanced_accuracy']
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


def normalize_dataset(X, normalization, normalization_params):
    """
    Normalize/Scale data with scalers
    """

    class scaler_():
        def transform(self, x):
            return x
        def fit(self, x):
            pass
        def set_params(self, x):
            pass

    if normalization == 'None':
        scaler = scaler_()
    elif normalization == 'StandardScaler':
        scaler = StandardScaler()
    elif normalization == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif normalization == 'RobustScaler':
        scaler = RobustScaler()
    elif normalization == 'PowerTransformer':
        scaler = PowerTransformer()
        scaler.set_params(**normalization_params)
    elif normalization == 'QuantileTransformer':
        scaler = QuantileTransformer()
        scaler.set_params(**normalization_params)
    else:
        raise NotImplementedError(f'Normalization method {normalization} not implemented')

    scaler.fit(X)
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index = X.index), scaler

def select_features(feature_method, X, y, max_features, n_trees, random_state):

    if feature_method == 'ExtraTrees':
        clf = ensemble.ExtraTreesClassifier(n_estimators=n_trees, random_state = random_state)
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

    elif feature_method == 'None':
        max_features = len(X.columns)
        top_sortindex = np.arange(len(y))
        p_values = np.zeros(len(y))
        feature_importance = np.zeros(len(y))
    else:
        raise NotImplementedError('Method {} not implemented.'.format(feature_method))

    top_features = X.columns[top_sortindex][:max_features][::-1].tolist()
    top_features_importance = feature_importance[top_sortindex][:max_features][::-1]
    top_features_pvalues = p_values[top_sortindex][:max_features][::-1]

    return top_features, top_features_importance, top_features_pvalues

def plot_feature_importance(feature_importance):
    """
    Creates a Plotly barplot to plot feature importance
    """
    fi = [pd.DataFrame.from_dict(_, orient='index') for _ in feature_importance]

    feature_df = pd.concat(fi)
    feature_df = feature_df.groupby(feature_df.index).sum()
    feature_df.columns = ['Feature_importance']
    feature_df = feature_df/feature_df.sum()
    feature_df = feature_df.sort_values(by='Feature_importance', ascending=False)

    feature_df = feature_df[feature_df['Feature_importance'] > 0]

    feature_df['Name'] = feature_df.index

    if len(feature_df) > 20:
        remainder = pd.DataFrame({'Feature_importance':[feature_df.iloc[30:].sum().values[0]],
        'Name':'Remainder'}, index=['Remainder'])
        feature_df = feature_df.iloc[:30] #Show at most 30 entries
        feature_df = feature_df.append(remainder)


    feature_df["Feature_importance"] = feature_df["Feature_importance"].map('{:.3f}'.format)
    #feature_df = feature_df.sort_values(by="Feature_importance", ascending=True)
    feature_df_wo_links = feature_df.copy()
    feature_df["Name"] = feature_df["Name"].apply(lambda x: '<a href="https://www.ncbi.nlm.nih.gov/search/all/?term={}" title="Search on NCBI" target="_blank">{}</a>'.format(x, x)
                                                    if not x.startswith('_') else x)
    feature_df["Plot_Name"] = feature_df_wo_links["Name"].apply(lambda x: '<a href="https://www.ncbi.nlm.nih.gov/search/all/?term={}" title="Search on NCBI" target="_blank">{}</a>'.format(x, x[0:21])
                                                    if not x.startswith('_') else x)


    marker_color = '#035672'
    title = 'Top 20 features from classifier'
    labels={"Feature_importance": "Feature importances from classifier", "Plot_Name": "Names"}

    # Hide pvalue if it does not exist
    hover_data = {"Plot_Name":False, "Name":True, "Feature_importance":True}

    p = px.bar(feature_df.iloc[::-1], x="Feature_importance", y="Plot_Name", orientation='h', hover_data=hover_data, labels=labels, height=600, title=title)
    p.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor= 'rgba(0, 0, 0, 0)', showlegend=False)
    p.update_traces(marker_color=marker_color)
    p.update_xaxes(showline=True, linewidth=1, linecolor='black')
    p.update_yaxes(showline=True, linewidth=1, linecolor='black')

    # Update `feature_df` for NaN in `P_values` and Column Naming
    feature_df.dropna(axis='columns', how="all", inplace=True)
    feature_df.drop("Plot_Name", inplace=True, axis=1)
    feature_df_wo_links.dropna(axis='columns', how="all", inplace=True)
    feature_df.rename(columns={'Name':'Name and NCBI Link', 'Feature_importance': 'Feature Importance'}, inplace=True)

    return p, feature_df, feature_df_wo_links

def impute_nan(X, missing_value, random_state):
    """
    Missing value imputation
    """
    class imputer_():
        def transform(self, x):
            return x
        def fit(self, x):
            pass

    if missing_value == 'Zero':
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    elif missing_value =='Mean':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif missing_value =='Median':
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
    elif missing_value == 'None':
        imp = imputer_()
    elif missing_value == 'IterativeImputer':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(random_state=random_state)
    elif missing_value == 'KNNImputer':
        imp = KNNImputer(n_neighbors=2)
    else:
        raise NotImplementedError('Method {} not implemented'.format(missing_value))

    imp.fit(X)
    X = pd.DataFrame(imp.transform(X), columns = X.columns)

    return X, imp

def return_classifier(classifier, classifier_params):
    """
    Returns classifier object based on name
    """
    # Max Features parameter for RandomForest and DecisionTree

    cp = classifier_params.copy()

    if classifier in ['LogisticRegression', 'KNeighborsClassifier','RandomForest']:
        cp['n_jobs'] = -1

    if classifier == 'LinearSVC':
        cv_generator = cp['cv_generator']
    else:
        cv_generator = None

    if classifier == 'XGBoost':
        from xgboost import XGBClassifier
        clf = XGBClassifier()
    elif classifier == 'LogisticRegression':
        clf = linear_model.LogisticRegression()
    elif classifier == 'KNeighborsClassifier':
        del cp['random_state']
        clf = neighbors.KNeighborsClassifier()
    elif classifier == 'RandomForest':
        clf = ensemble.RandomForestClassifier()
    elif classifier == 'DecisionTree':
        clf = tree.DecisionTreeClassifier()
    elif classifier == 'AdaBoost':
        clf = ensemble.AdaBoostClassifier()
    elif classifier == 'LinearSVC':
        del cp['cv_generator']
        clf = svm.LinearSVC()
    clf.set_params(**cp)
    return clf, cv_generator

def perform_cross_validation(state, cohort_column = None):

    clf, cv_generator = return_classifier(state.classifier, state.classifier_params)

    if state.cv_method == 'RepeatedStratifiedKFold':
        cv_alg = RepeatedStratifiedKFold(n_splits=state.cv_splits, n_repeats=state.cv_repeats, random_state=state.random_state)
    elif state.cv_method == 'StratifiedKFold':
        cv_alg = StratifiedKFold(n_splits=state.cv_splits, shuffle=True, random_state=state.random_state)
    elif state.cv_method == 'StratifiedShuffleSplit':
        cv_alg = StratifiedShuffleSplit(n_splits=state.cv_splits, random_state=state.random_state)
    else:
        raise NotImplementedError('This CV method is not implemented')

    _cv_results = {}
    _cv_curves = {}

    # Initialize reporting dict with empty lists
    for _ in ['num_feat', 'n_obs', 'n_class_0', 'n_class_1', 'class_ratio']:
        for x in ['_train','_test']:
            _cv_results[_+x] = []

    for _ in ['pr_auc','roc_curves_', 'pr_curves_', 'y_hats_','feature_importances_','features_']:
        _cv_curves[_] = []

    for metric_name, metric_fct in scorer_dict.items():
        _cv_results[metric_name] = []
    _cv_results['pr_auc'] = [] # ADD pr_auc manually

    X = state.X
    y = state.y

    if cohort_column is not None:
        cohorts = state.X_cohort.unique().tolist()
        cohort_combos = []
        cohort_combo_names = []

        indexer = np.arange(len(X))
        for c_1 in cohorts:
            for c_2 in cohorts:
                if c_1 != c_2:
                    cohort_combos.append((indexer[state.X_cohort == c_1], indexer[state.X_cohort == c_2]))
                    cohort_combo_names.append((c_1, c_2))

        iterator = cohort_combos

        cohort_combo_names_ = []

    else:
        iterator = cv_alg.split(X,y)

    X = X[state.features]

    for i, (train_index, test_index) in enumerate(iterator):

        # Missing value imputation
        X_train, imputer = impute_nan(X.iloc[train_index], state.missing_value, state.random_state)
        X_test = pd.DataFrame(imputer.transform(X.iloc[test_index]), columns = X.iloc[test_index].columns)

        # Normalization of data
        X_train, scaler = normalize_dataset(X_train, state.normalization, state.normalization_params)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index = X_test.index)

        # Define y
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        skip = False
        if cohort_column is not None:
            if (len(set(y_train)) == 1):
                st.warning(f"Only 1 class present in cohort {cohort_combo_names[i][0]}. Skipping training on {cohort_combo_names[i][0]} and predicting on {cohort_combo_names[i][1]}.")
                skip = True
            if (len(set(y_test)) == 1):
                st.warning(f"Only 1 class present in cohort {cohort_combo_names[i][1]}. Skipping training on {cohort_combo_names[i][0]} and predicting on {cohort_combo_names[i][1]}.")
                skip = True

            if not skip:

                cohort_combo_names_.append(cohort_combo_names[i])

        if not skip:
            #Feature selection
            features_, feature_importance_, p_values = select_features(state.feature_method, X_train, y_train, state.max_features, state.n_trees, state.random_state)

            X_train = X_train[features_]
            X_test = X_test[features_]

            # Fitting and predicting, and calculating prediction probabilities
            if state.classifier == "LinearSVC":
                # Since LinearSVC does not have `predict_proba()`
                from sklearn.calibration import CalibratedClassifierCV
                calibrated_clf = CalibratedClassifierCV(clf, cv=cv_generator)
                calibrated_clf.fit(X_train, y_train)
                y_pred = calibrated_clf.predict(X_test)
                y_pred_proba = calibrated_clf.predict_proba(X_test)
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)

            # Feature importances received from classifier
            if state.classifier == 'LogisticRegression':
                feature_importance = np.abs(clf.coef_[0])
            elif state.classifier == 'LinearSVC':
                coef_avg = 0
                for j in calibrated_clf.calibrated_classifiers_:
                    coef_avg = coef_avg + j.base_estimator.coef_
                coef_avg  = coef_avg / len(calibrated_clf.calibrated_classifiers_)
                feature_importance = coef_avg

            elif state.classifier in ['AdaBoost', 'RandomForest', 'DecisionTree', 'XGBoost']:
                feature_importance = clf.feature_importances_
            else:
                # Not implemented st.warning() for `KNeighborsClassifier`.
                feature_importance = None

            # ROC CURVE
            fpr, tpr, cutoffs = roc_curve(y_test, y_pred_proba[:, 1])

            # PR CURVE
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])

            for metric_name, metric_fct in scorer_dict.items():
                if metric_name == 'roc_auc':
                    _cv_results[metric_name].append(metric_fct(y_test, y_pred_proba[:,1]))
                elif metric_name in ['precision', 'recall', 'f1']:
                    _cv_results[metric_name].append(metric_fct(y_test, y_pred, zero_division=0))
                else:
                    _cv_results[metric_name].append(metric_fct(y_test, y_pred))

            # Results of Cross Validation
            _cv_results['num_feat_train'].append(X_train.shape[-1])
            _cv_results['n_obs_train'].append(len(y_train))
            _cv_results['n_class_0_train'].append(np.sum(y_train))
            _cv_results['n_class_1_train'].append(np.sum(~y_train))
            _cv_results['class_ratio_train'].append(np.sum(y_train)/len(y_train))

            _cv_results['num_feat_test'].append(X_test.shape[-1])
            _cv_results['n_obs_test'].append(len(y_test))
            _cv_results['n_class_0_test'].append(np.sum(y_test))
            _cv_results['n_class_1_test'].append(np.sum(~y_test))
            _cv_results['class_ratio_test'].append(np.sum(y_test)/len(y_test))
            _cv_results['pr_auc'].append(auc(recall, precision)) # ADD PR Curve AUC Score

            _cv_curves['pr_auc'].append(auc(recall, precision)) # ADD PR Curve AUC Score
            _cv_curves['roc_curves_'].append((fpr, tpr, cutoffs))
            _cv_curves['pr_curves_'].append((precision, recall, _))
            _cv_curves['y_hats_'].append((y_test.values, y_pred))

            if feature_importance is not None:
                _cv_curves['feature_importances_'].append(dict(zip(X_train.columns.tolist(), feature_importance)))
            else:
                _cv_curves['feature_importances_'] = None

        if cohort_column is not None:
            state.bar.progress((i+1)/len(cohort_combos))
        else:
            if state.cv_method == 'RepeatedStratifiedKFold':
                state.bar.progress((i+1)/(state.cv_splits*state.cv_repeats))
            else:
                state.bar.progress((i+1)/(state.cv_splits))

        if cohort_column is not None:
            _cv_curves['cohort_combos'] = cohort_combo_names_

    return _cv_results, _cv_curves

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
    data = [
        go.Heatmap(x=x_, y=y_, z=cm_results[step][1], visible=False,
        hoverinfo='none', colorscale = custom_colorscale)
        for step in range(len(cm_results))
        ]
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

def plot_roc_curve_cv(roc_curve_results, cohort_combos = None):
    """Plotly chart for roc curve for cross validation"""

    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    roc_aucs = []
    p = go.Figure()

    for idx, (fpr, tpr, threshold) in enumerate(roc_curve_results):
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        if cohort_combos is not None:
            text= "Train: {} <br>Test: {}".format(cohort_combos[idx][0], cohort_combos[idx][1])
            hovertemplate = "False positive rate: %{x:.2f} <br>True positive rate: %{y:.2f}" + "<br>" + text
            p.add_trace(go.Scatter(x=fpr, y=tpr, hovertemplate=hovertemplate, hoverinfo='all', mode='lines',
                        name='Train on {}, Test on {}, AUC {:.2f}'.format(cohort_combos[idx][0], cohort_combos[idx][1], roc_auc)))
        else:
            p.add_trace(go.Scatter(x=fpr, y=tpr, hoverinfo='skip', mode='lines', line=dict(color=blue_color), showlegend=False,  opacity=0.1))
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0]=0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = np.maximum(mean_tprs - std, 0)

    mean_rocauc = np.mean(roc_aucs).round(2)
    sd_rocauc = np.std(roc_aucs).round(2)

    if cohort_combos is None:
        p.add_trace(go.Scatter(x=base_fpr, y=tprs_lower, fill = None, line_color='gray', opacity=0.1, showlegend=False))
        p.add_trace(go.Scatter(x=base_fpr, y=tprs_upper, fill='tonexty', line_color='gray', opacity=0.1, name='±1 std. dev'))

        hovertemplate = "Base FPR %{x:.2f} <br>%{text}"
        text = ["Upper TPR {:.2f} <br>Mean TPR {:.2f} <br>Lower TPR {:.2f}".format(u, m, l) for u, m, l in zip(tprs_upper, mean_tprs, tprs_lower)]

        p.add_trace(go.Scatter(x=base_fpr, y=mean_tprs, text=text, hovertemplate=hovertemplate, hoverinfo = 'y+text',
                                line=dict(color='black', width=2), name='Mean ROC\n(AUC = {:.2f}±{:.2f})'.format(mean_rocauc, sd_rocauc)))

        p.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=red_color, dash='dash'), name="Chance"))

    else:
        p.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='black', dash='dash'), name="Chance"))

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
                    )
    return p

def plot_pr_curve_cv(pr_curve_results, class_ratio_test, cohort_combos = None):
    """Plotly chart for Precision-Recall PR curve"""

    precisions = []
    base_recall = np.linspace(0, 1, 101)
    pr_aucs = []
    p = go.Figure()

    for idx, (precision, recall, _) in enumerate(pr_curve_results):
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        if cohort_combos is not None:
            pr_df = pd.DataFrame({'recall':recall,'precision':precision, 'train':cohort_combos[idx][0], 'test':cohort_combos[idx][1]})
            text= "Train: {} <br>Test: {}".format(cohort_combos[idx][0], cohort_combos[idx][1])
            hovertemplate = "Recall: %{x:.2f} <br>Precision: %{y:.2f}" + "<br>" + text
            p.add_trace(go.Scatter(x=recall, y=precision, hovertemplate=hovertemplate, hoverinfo='all', mode='lines',
                                    name='Train on {}, Test on {}, AUC {:.2f}'.format(cohort_combos[idx][0], cohort_combos[idx][1], pr_auc)))
        else:
            p.add_trace(go.Scatter(x=recall, y=precision, hoverinfo='skip', mode='lines', line=dict(color=blue_color), showlegend=False,  opacity=0.1))
        precision = np.interp(base_recall, recall, precision, period=100)
        precision[0]=1.0
        precisions.append(precision)

    precisions = np.array(precisions)
    mean_precisions = precisions.mean(axis=0)
    std = precisions.std(axis=0)
    precisions_upper = np.minimum(mean_precisions + std, 1)
    precisions_lower = np.maximum(mean_precisions - std, 0)

    mean_prauc = np.mean(pr_aucs).round(2)
    sd_prauc = np.std(pr_aucs).round(2)

    if cohort_combos is None:
        p.add_trace(go.Scatter(x=base_recall, y=precisions_lower, fill = None, line_color='gray', opacity=0.1, showlegend=False))
        p.add_trace(go.Scatter(x=base_recall, y=precisions_upper, fill='tonexty', line_color='gray', opacity=0.2, name='±1 std. dev'))

        hovertemplate = "Base Recall %{x:.2f} <br>%{text}"
        text = ["Upper Precision {:.2f} <br>Mean Precision {:.2f} <br>Lower Precision {:.2f}".format(u, m, l)
                    for u, m, l in zip(precisions_upper, mean_precisions, precisions_lower)]

        p.add_trace(go.Scatter(x=base_recall, y=mean_precisions, text=text, hovertemplate=hovertemplate, hoverinfo = 'y+text',
                                line=dict(color='black', width=2), name='Mean PR\n(AUC = {:.2f}±{:.2f})'.format(mean_prauc, sd_prauc)))

        no_skill = np.mean(class_ratio_test)
        p.add_trace(go.Scatter(x=[0, 1], y=[no_skill, no_skill], line=dict(color=red_color, dash='dash'), name="Chance"))
    else:
        no_skill = np.mean(class_ratio_test)
        p.add_trace(go.Scatter(x=[0, 1], y=[no_skill, no_skill], line=dict(color='black', dash='dash'), name="Chance"))


    p.update_xaxes(showline=True, linewidth=1, linecolor='black')
    p.update_yaxes(showline=True, linewidth=1, linecolor='black')
    p.update_layout(autosize=True,
                    width=800,
                    height=700,
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    yaxis = dict(
                        scaleanchor = "x",
                        scaleratio = 1,
                        zeroline=True,
                        ),
                    )
    return p

def get_system_report():
    """
    Returns the package versions
    """

    report = {}
    report['omic_learn_version'] = "v0.9.0-dev"
    report['python_version'] = sys.version[:5]
    report['pandas_version'] = pd.__version__
    report['numpy_version'] = np.version.version
    report['sklearn_version'] = sklearn.__version__
    report['plotly_version'] = plotly.__version__

    return report

def get_download_link(exported_object, name):
    """
    Generate download link for charts in SVG and PDF formats and for dataframes in CSV format
    """

    os.makedirs("downloads/", exist_ok=True)
    extension = name.split(".")[-1]

    if extension == 'svg':
        exported_object.write_image("downloads/"+ name)
        with open("downloads/" + name) as f:
            svg = f.read()
        b64 = base64.b64encode(svg.encode()).decode()
        href = f'<a class="download_link" href="data:image/svg+xml;base64,%s" download="%s" >Download as *.svg</a>' % (b64, name)
        st.markdown('')
        st.markdown(href, unsafe_allow_html=True)

    elif extension == 'pdf':
        exported_object.write_image("downloads/"+ name)
        with open("downloads/" + name, "rb") as f:
            pdf = f.read()
        b64 = base64.encodebytes(pdf).decode()
        href = f'<a class="download_link" href="data:application/pdf;base64,%s" download="%s" >Download as *.pdf</a>' % (b64, name)
        st.markdown('')
        st.markdown(href, unsafe_allow_html=True)

    elif extension == 'csv':
        exported_object.to_csv("downloads/"+ name, index=False)
        with open("downloads/" + name, "rb") as f:
            csv = f.read()
        b64 = base64.b64encode(csv).decode()
        href = f'<a class="download_link" href="data:file/csv;base64,%s" download="%s" >Download as *.csv</a>' % (b64, name)
        st.markdown('')
        st.markdown(href, unsafe_allow_html=True)

    else:
        raise NotImplementedError('This output format function is not implemented')
