import random
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime
import utils.session_states as session_states
from utils.helper import get_download_link, make_recording_widget, load_data, transform_dataset
from utils.helper import select_features, plot_feature_importance, impute_nan, perform_cross_validation, plot_confusion_matrices
from utils.helper import perform_cohort_validation, plot_roc_curve_cv, plot_pr_curve_cv, get_system_report
icon = Image.open('./utils/omic_learn.png')

# Checkpoint for XGBoost
xgboost_installed = False
try:
    from xgboost import XGBClassifier
    import xgboost
    xgboost_installed = True
except ModuleNotFoundError:
    st.error('Xgboost not installed. To use xgboost install using `conda install py-xgboost`')

# Define all versions
report = get_system_report()
version = report['omic_learn_version']

# Objdict class to conveniently store a state
class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

# Functions / Element Creations
def main_components():

    # External CSS
    main_external_css = """
        <style>
            .footer {position: absolute; height: 50px; bottom: -150px; width:100%; padding:10px; text-align:center; }
            #MainMenu, .reportview-container .main footer {display: none;}
            .btn-outline-secondary {background: #FFF !important}
            .download_link {color: #f63366 !important; text-decoration: none !important; z-index: 99999 !important;
                            cursor:pointer !important; margin: 15px 0px; border: 1px solid #f63366;
                            text-align:center; padding: 8px !important; width: 200px;}
            .download_link:hover {background: #f63366 !important; color: #FFF !important;}
            h1, h2, h3, h4, h5, h6, a, a:visited {color: #f84f57 !important}
            label, stText, p, .caption {color: #035672 }
            .sidebar .sidebar-content {background: #035672 !important;}
            .sidebar-content label, stText, p, .caption {color: #FFF !important}
            .sidebar-content a {text-decoration:underline;}
            .tickBarMin, .tickBarMax {color: #f84f57 !important}
            .markdown-text-container p {color: #035672 !important}

            /* Tabs */
            .tabs { position: relative; min-height: 200px; clear: both; margin: 40px auto 0px auto; background: #efefef; box-shadow: 0 48px 80px -32px rgba(0,0,0,0.3); }
            .tab {float: left;}
            .tab label { background: #f84f57; cursor: pointer; font-weight: bold; font-size: 18px; padding: 10px; color: #fff; transition: background 0.1s, color 0.1s; margin-left: -1px; position: relative; left: 1px; top: -29px; z-index: 2; }
            .tab label:hover {background: #035672;}
            .tab [type=radio] { display: none; }
            .content { position: absolute; top: -1px; left: 0; background: #fff; right: 0; bottom: 0; padding: 30px 20px; transition: opacity .1s linear; opacity: 0; }
            [type=radio]:checked ~ label { background: #035672; color: #fff;}
            [type=radio]:checked ~ label ~ .content { z-index: 1; opacity: 1; }

            /* Feature Importance Plotly Link Color */
            .js-plotly-plot .plotly svg a {color: #f84f57 !important}
        </style>
    """
    st.markdown(main_external_css, unsafe_allow_html=True)

    # Fundemental elements
    widget_values = objdict()
    record_widgets = objdict()

    # Sidebar widgets
    record_widgets['button_'] = make_recording_widget(st.sidebar.button, widget_values)
    record_widgets['slider_'] = make_recording_widget(st.sidebar.slider, widget_values)
    record_widgets['multiselect_'] = make_recording_widget(st.sidebar.multiselect, widget_values)
    record_widgets['number_input_'] = make_recording_widget(st.sidebar.number_input, widget_values)
    record_widgets['selectbox_'] = make_recording_widget(st.sidebar.selectbox, widget_values)
    record_widgets['multiselect'] = make_recording_widget(st.multiselect, widget_values)

    return widget_values, record_widgets

# Show main text and data upload section
def main_text_and_data_upload(state):
    st.title("DEV | OmicLearn — ML for Omics Data")
    st.info("""
        * Upload your excel / csv file here. Maximum size is 200 Mb.
        * Each row corresponds to a sample, each column to a feature.
        * 'Features' such as protein IDs, gene names, lipids or miRNA IDs should be uppercase.
        * Additional features should be marked with a leading '_'.
    """)
    st.subheader("Dataset")
    file_buffer = st.file_uploader("Upload your dataset below", type=["csv", "xlsx", "xls"])
    st.markdown("By uploading a file, you agree that you accepting [the licence agreement](https://github.com/OmicEra/OmicLearn).")
    delimiter = st.selectbox("Determine the delimiter in your dataset", ["Excel File", "Comma (,)", "Semicolon (;)"])
    state['sample_file'] = st.selectbox("Or select sample file here:", ["None", "Alzheimer", "Sample"])
    state['df'] = load_data(file_buffer, delimiter)

    return state

# Choosing sample dataset and data parameter selections
def checkpoint_for_data_upload(state, record_widgets):

    multiselect = record_widgets.multiselect

    # Sample dataset / uploaded file selection
    if (state.sample_file != 'None') and (len(state.df) > 0):
        st.warning("Please, either choose a sample file or set it as `None` to work on your file")
        state['df'] = pd.DataFrame()
    elif state.sample_file != 'None':
        if state.sample_file == "Alzheimer":
            st.info("""
                **This dataset is retrieved from the following paper and the code for parsing is available at
                [GitHub](https://github.com/OmicEra/OmicLearn/blob/master/data/Alzheimer_paper.ipynb):**\n
                Bader, J., Geyer, P., Müller, J., Strauss, M., Koch, M., & Leypoldt, F. et al. (2020).
                Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease.
                Molecular Systems Biology, 16(6). doi: [10.15252/msb.20199356](http://doi.org/10.15252/msb.20199356) """)
        state['df'] = pd.read_excel('data/'+ state.sample_file + '.xlsx')
        st.write(state.df)
    elif len(state.df) > 0:
        st.text("Using the following dataset:")
        st.write(state.df)
    else:
        st.error('No dataset uploaded or selected.')

    state['n_missing'] = state.df.isnull().sum().sum()

    if len(state.df) > 0:
        if state.n_missing > 0:
            st.warning('Found {} missing values. Use missing value imputation or xgboost classifier.'.format(state.n_missing))

        # Distinguish the features from others
        state['proteins'] = [_ for _ in state.df.columns.to_list() if _[0] != '_']
        state['not_proteins'] = [_ for _ in state.df.columns.to_list() if _[0] == '_']

        # Dataset -- Subset
        st.markdown("\nSubset allows you to specify a subset of data based on values within a comma. \nThis way, you can exclude data that should not be used at all.")
        if st.checkbox("Create subset"):
            st.text('Create a subset based on values in the selected column')
            state['subset_column'] = st.selectbox("Select subset column:", ['None']+state.not_proteins)

            if state.subset_column != 'None':
                subset_options = state.df[state.subset_column].value_counts().index.tolist()
                subset_class = multiselect("Select values to keep:", subset_options, default=subset_options)
                state['df_sub'] = state.df[state.df[state.subset_column].isin(subset_class)].copy()
            elif state.subset_column == 'None':
                state['df_sub'] = state.df.copy()
                state['subset_column'] = 'None'
        else:
            state['df_sub'] = state.df.copy()
            state['subset_column'] = 'None'

        # Dataset -- Feature selections
        st.subheader("Classification target")
        state['target_column'] = st.selectbox("Select target column:", state.not_proteins)
        st.markdown("Unique elements in `{}` column:".format(state.target_column))
        unique_elements = state.df_sub[state.target_column].value_counts()
        st.write(unique_elements)
        unique_elements_lst = unique_elements.index.tolist()

        # # Dataset -- Define the classes
        st.subheader("Define classes".format(state.target_column))
        state['class_0'] = multiselect("Select Class 0:", unique_elements_lst, default=None)
        state['class_1'] = multiselect("Select Class 1:", [_ for _ in unique_elements_lst if _ not in state.class_0], default=None)
        state['remainder'] = [_ for _ in state.not_proteins if _ is not state.target_column]


        if state.class_0 and state.class_1:

            st.subheader("Additional features")
            st.text("Select additional features. All non numerical values will be encoded (e.g. M/F -> 0,1)")
            state['additional_features'] = multiselect("Select additional features for trainig:", state.remainder, default=None)

            if st.checkbox("Exclude features"):
                # File uploading target_column for exclusion
                exclusion_file_buffer = st.file_uploader("Upload your CSV (comma(,) seperated) file here in which each row corresponds to a feature to be excluded.", type=["csv"])
                exclusion_df = load_data(exclusion_file_buffer, "Comma (,)")
                if len(exclusion_df) > 0:
                    st.text("The following features will be exlcuded:")
                    st.write(exclusion_df)
                    exclusion_df_list = list(exclusion_df.iloc[:,0].unique())
                    state['exclude_features'] = multiselect("Select features to be excluded:", state.proteins, default=exclusion_df_list)
                else:
                    state['exclude_features'] = multiselect("Select features to be excluded:", state.proteins, default=[])
            else:
                state['exclude_features']  = []

            if st.checkbox("Manually select features"):
                st.markdown("Manually select a subset of features.")
                state.proteins = multiselect("Select your features manually:", state.proteins, default=None)


        # Dataset -- Cohort selections
        state['cohort_checkbox'] = st.checkbox("Cohort comparison")
        if state.cohort_checkbox:
            st.text('Select cohort column to train on one and predict on another:')
            not_proteins_excluded_target_option = state.not_proteins
            not_proteins_excluded_target_option.remove(state.target_column)
            state['cohort_column'] = st.selectbox("Select cohort column:", not_proteins_excluded_target_option)
        else:
            state['cohort_column'] = None

        if 'exclude_features' not in state:
            state['exclude_features'] = []

        state['proteins'] = [_ for _ in state.proteins if _ not in state.exclude_features]

    return state

# Generate sidebar elements
def generate_sidebar_elements(state, record_widgets):

    slider_ = record_widgets.slider_
    selectbox_ = record_widgets.selectbox_
    number_input_ = record_widgets.number_input_

    # Sidebar -- Image/Title
    st.sidebar.image(icon, use_column_width=True, caption="OmicLearn " + version)
    st.sidebar.markdown("# [Options](https://github.com/OmicEra/OmicLearn/wiki/METHODS)")

    # Sidebar -- Random State
    state['random_state'] = slider_("Random State:", min_value = 0, max_value = 99, value=23)

    # Sidebar -- Preprocessing
    st.sidebar.markdown('## [Preprocessing](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing)')
    normalizations = ['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer']
    state['normalization'] = selectbox_("Normalization method:", normalizations)

    normalization_params = {}

    if state.normalization == "PowerTransformer":
        normalization_params['method'] = selectbox_("Power transformation method:", ["Yeo-Johnson", "Box-Cox"]).lower()
    elif state.normalization == "QuantileTransformer":
        normalization_params['random_state'] = state.random_state
        normalization_params['n_quantiles'] = number_input_("Number of quantiles:", value = 100, min_value = 1, max_value = 2000)
        normalization_params['output_distribution'] = selectbox_("Output distribution method:", ["Uniform", "Normal"]).lower()
    if state.n_missing > 0:
        st.sidebar.markdown('## [Missing value imputation](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing#1-2-imputation-of-missing-values)')
        missing_values = ['Zero', 'Mean', 'Median', 'IterativeImputer', 'KNNImputer', 'None']
        state['missing_value'] = selectbox_("Missing value imputation", missing_values)
    else:
        state['missing_value'] = 'None'

    state['normalization_params'] = normalization_params

    # Sidebar -- Feature Selection
    st.sidebar.markdown('## [Feature selection](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-2.-Feature-selection)')
    feature_methods = ['ExtraTrees', 'k-best (mutual_info_classif)','k-best (f_classif)', 'k-best (chi2)', 'None']
    state['feature_method'] = selectbox_("Feature selection method:", feature_methods)

    if state.feature_method != 'None':
        state['max_features'] = number_input_('Maximum number of features:', value = 20, min_value = 1, max_value = 2000)
    else:
        # Define `max_features` as 0 if `feature_method` is `None`
        state['max_features'] = 0

    if state.feature_method == "ExtraTrees":
        state['n_trees'] = number_input_('Number of trees in the forest:', value = 100, min_value = 1, max_value = 2000)
    else:
        state['n_trees'] = 0

    # Sidebar -- Classification method selection
    st.sidebar.markdown('## [Classification](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-3.-Classification#3-classification)')
    if xgboost_installed:
        classifiers = ['AdaBoost','LogisticRegression','KNeighborsClassifier','RandomForest','DecisionTree','LinearSVC','XGBoost']
    else:
        classifiers = ['AdaBoost','LogisticRegression','KNeighborsClassifier','RandomForest','DecisionTree','LinearSVC']

    # Disable all other classification methods
    if (state.n_missing > 0) and (state.missing_value == 'None'):
        classifiers = ['XGBoost']

    state['classifier'] = selectbox_("Specify the classifier:", classifiers)

    classifier_params = {}
    classifier_params['random_state'] = state['random_state']

    if state.classifier == 'AdaBoost':
        classifier_params['n_estimators'] = number_input_('Number of estimators:', value = 100, min_value = 1, max_value = 2000)
        classifier_params['learning_rate'] = number_input_('Learning rate:', value = 1.0, min_value = 0.001, max_value = 100.0)

    elif state.classifier == 'KNeighborsClassifier':
        classifier_params['n_neighbors'] = number_input_('Number of neighbors:', value = 100, min_value = 1, max_value = 2000)
        classifier_params['weights'] = selectbox_("Select weight function used:", ["uniform", "distance"])
        classifier_params['algorithm'] = selectbox_("Algorithm for computing the neighbors:", ["auto", "ball_tree", "kd_tree", "brute"])

    elif state.classifier == 'LogisticRegression':
        classifier_params['penalty'] = selectbox_("Specify norm in the penalization:", ["l2", "l1", "ElasticNet", "None"]).lower()
        classifier_params['solver'] = selectbox_("Select the algorithm for optimization:", ["lbfgs", "newton-cg", "liblinear", "sag", "saga"])
        classifier_params['max_iter'] = number_input_('Maximum number of iteration:', value = 100, min_value = 1, max_value = 2000)
        classifier_params['C'] = number_input_('C parameter:', value = 1, min_value = 1, max_value = 100)

    elif state.classifier == 'RandomForest':
        classifier_params['n_estimators'] = number_input_('Number of estimators:', value = 100, min_value = 1, max_value = 2000)
        classifier_params['criterion'] =  selectbox_("Function for measure the quality:", ["gini", "entropy"])
        classifier_params['max_features'] = selectbox_("Number of max. features:", ["auto", "int", "sqrt", "log2"])
        if classifier_params['max_features'] == "int":
            classifier_params['max_features'] = number_input_('Number of max. features:', value = 5, min_value = 1, max_value = 100)

    elif state.classifier == 'DecisionTree':
        classifier_params['criterion'] =  selectbox_("Function for measure the quality:", ["gini", "entropy"])
        classifier_params['max_features'] = selectbox_("Number of max. features:", ["auto", "int", "sqrt", "log2"])
        if classifier_params['max_features'] == "int":
            classifier_params['max_features'] = number_input_('Number of max. features:', value = 5, min_value = 1, max_value = 100)

    elif state.classifier == 'LinearSVC':
        classifier_params['penalty'] = selectbox_("Specify norm in the penalization:", ["l2", "l1"])
        classifier_params['loss'] = selectbox_("Select loss function:", ["squared_hinge", "hinge"])
        classifier_params['C'] = number_input_('C parameter:', value = 1, min_value = 1, max_value = 100)
        classifier_params['cv_generator'] = number_input_('Cross-validation generator:', value = 2, min_value = 2, max_value = 100)

    elif state.classifier == 'XGBoost':
        classifier_params['learning_rate'] = eta = number_input_('Learning rate:', value = 0.3, min_value = 0.0, max_value = 1.0)
        classifier_params['min_split_loss'] = gamma = number_input_('Min. split loss:', value = 0, min_value = 0, max_value = 100)
        classifier_params['max_depth'] = number_input_('Max. depth:', value = 6, min_value = 0, max_value = 100)
        classifier_params['min_child_weight'] = number_input_('Min. child weight:', value = 1, min_value = 0, max_value = 100)


    state['classifier_params'] = classifier_params

    # Sidebar -- Cross-Validation
    st.sidebar.markdown('## [Cross Validation](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-4.-Validation#4-1-cross-validation)')
    state['cv_method'] = selectbox_("Specify CV method:", ["RepeatedStratifiedKFold", "StratifiedKFold", "StratifiedShuffleSplit"])
    state['cv_splits'] = number_input_('CV Splits:', min_value = 2, max_value = 10, value=5)

    # Define placeholder variables for CV
    if state.cv_method == 'RepeatedStratifiedKFold':
        state['cv_repeats'] = number_input_('CV Repeats:', min_value = 1, max_value = 50, value=10)

    return state

# Display results and plots
def classify_and_plot(state):

    state.bar = st.progress(0)
    # Cross-Validation
    st.markdown("Running Cross-Validation")
    cv_results, cv_curves = perform_cross_validation(state)

    st.header('Cross-Validation')

    # Feature Importances from Classifier
    st.subheader('Feature Importances from classifier')
    if state.cv_method == 'RepeatedStratifiedKFold':
        st.markdown(f'This is the average feature importance from all {state.cv_splits*state.cv_repeats} cross validation runs.')
    else:
        st.markdown(f'This is the average feature importance from all {state.cv_splits} cross validation runs.')
    if cv_curves['feature_importances_'] is not None:
        p, feature_df, feature_df_wo_links = plot_feature_importance(cv_curves['feature_importances_'])
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, 'clf_feature_importance.pdf')
            get_download_link(p, 'clf_feature_importance.svg')

        # Display `feature_df` with NCBI links
        st.subheader("Feature importances from classifier table")
        st.write(feature_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        get_download_link(feature_df_wo_links, 'clf_feature_importances.csv')
    else:
        st.warning('Feature importance attribute is not implemented for this classifier.')

    # ROC-AUC
    st.subheader('Receiver operating characteristic')
    p = plot_roc_curve_cv(cv_curves['roc_curves_'])
    st.plotly_chart(p)
    if p:
        get_download_link(p, 'roc_curve.pdf')
        get_download_link(p, 'roc_curve.svg')

    # Precision-Recall Curve
    st.subheader('Precision-Recall Curve')
    st.text("Precision-Recall (PR) Curve might be used for imbalanced datasets.")
    p = plot_pr_curve_cv(cv_curves['pr_curves_'], cv_results['class_ratio_test'])
    st.plotly_chart(p)
    if p:
        get_download_link(p, 'pr_curve.pdf')
        get_download_link(p, 'pr_curve.svg')

    # Confusion Matrix (CM)
    st.subheader('Confusion matrix')
    names = ['CV_split {}'.format(_+1) for _ in range(len(cv_curves['y_hats_']))]
    names.insert(0, 'Sum of all splits')
    p  = plot_confusion_matrices(state.class_0, state.class_1, cv_curves['y_hats_'], names)
    st.plotly_chart(p)
    if p:
        get_download_link(p, 'cm.pdf')
        get_download_link(p, 'cm.svg')

    # Results
    st.subheader('Run Results for `{}`'.format(state.classifier))

    state['summary'] = pd.DataFrame(pd.DataFrame(cv_results).describe())

    st.write(state.summary)
    get_download_link(state.summary, "run_results.csv")


    if state.cohort_checkbox:
        st.header('Cohort comparison')
        cohort_results, cohort_curves = perform_cross_validation(state, state.cohort_column)

        # ROC-AUC for Cohorts
        st.subheader('Receiver operating characteristic')
        p = plot_roc_curve_cv(cohort_curves['roc_curves_'], cohort_curves['cohort_combos'])
        st.plotly_chart(p)
        if p:
            get_download_link(p, 'roc_curve_cohort.pdf')
            get_download_link(p, 'roc_curve_cohort.svg')

        # PR Curve for Cohorts
        st.subheader('Precision-Recall Curve')
        st.text("Precision-Recall (PR) Curve might be used for imbalanced datasets.")
        p = plot_pr_curve_cv(cohort_curves['pr_curves_'], cohort_results['class_ratio_test'], cohort_curves['cohort_combos'])
        st.plotly_chart(p)
        if p:
            get_download_link(p, 'pr_curve_cohort.pdf')
            get_download_link(p, 'pr_curve_cohort.svg')

        st.subheader('Confusion matrix')
        names = ['Train on {}, Test on {}'.format(_[0], _[1]) for _ in cohort_curves['cohort_combos']]
        names.insert(0, 'Sum of cohort comparisons')

        # Confusion Matrix (CM) for Cohorts
        p = plot_confusion_matrices(state.class_0, state.class_1, cohort_curves['y_hats_'], names)
        st.plotly_chart(p)
        if p:
            get_download_link(p, 'cm_cohorts.pdf')
            get_download_link(p, 'cm_cohorts.svg')

        state['cohort_summary'] = pd.DataFrame(pd.DataFrame(cv_results).describe())
        st.write(state.cohort_summary)

        state['cohort_combos'] = cohort_curves['cohort_combos']
        state['cohort_results'] = cohort_results
        get_download_link(state.cohort_summary, "run_results_cohort.csv")


    return state

# Generate summary text
def generate_text(state):

    st.write("## Summary")
    text = ""
    # Packages
    packages_plain_text = """
        OmicLearn ({omic_learn_version}) was utilized for performing the data analysis, model execution, and generating the plots and charts.
        Machine learning was done in Python ({python_version}). Identifier tables were imported via the Pandas package ({pandas_version}) and manipulated using the Numpy package ({numpy_version}).
        The machine learning pipeline was employed using the scikit-learn package ({sklearn_version}).
        For generating the plots and charts, Plotly ({plotly_version}) library was used.
    """
    text += packages_plain_text.format(**report)

    # Normalization
    if state.normalization == 'None':
        text += 'No normalization on the data was performed. '
    else:
        params = [f'{k} = {v}' for k, v in state.normalization_params.items()]
        text += f"Data was normalized in each using a {state.normalization} ({' '.join(params)}) approach. "

        # Missing value impt.
        if state.missing_value != "None":
            text += 'To impute missing values, a {}-imputation strategy is used. '.format(state.missing_value)
        else:
            text += 'The dataset contained no missing values; hence no imputation was performed'

    # Features
    if state.feature_method == 'None':
        text += 'No feature selection algorithm was applied. '
    elif state.feature_method == 'ExtraTrees':
        text += 'Features were selected using a {} (n_trees={}) strategy with the maximum number of {} features. '.format(state.feature_method, state.n_trees, state.max_features)
    else:
        text += 'Features were selected using a {} strategy with the maximum number of {} features. '.format(state.feature_method, state.max_features)

    text += 'Normalization and feature selection was individually performed using the training data of each split. '


    params = [f'{k} = {v}' for k, v in state.classifier_params.items()]
    text += f"For classification, we used a {state.classifier}-Classifier ({' '.join(params)}) "

    if False: #ToDO: This is still buggy
        # Cross-Validation
        if state.cv_method == 'RepeatedStratifiedKFold':
            cv_plain_text = """
                When using (RepeatedStratifiedKFold) a repeated (n_repeats={}), stratified cross-validation (n_splits={}) approach to classify {} vs. {},
                we achieved a receiver operating characteristic (ROC) with an average AUC (area under the curve) of {:.2f} ({:.2f} std)
                and Precision-Recall (PR) Curve with an average AUC of {:.2f} ({:.2f} std).
            """
            text += cv_plain_text.format(state.cv_repeats, state.cv_splits, ''.join(state.class_0), ''.join(state.class_1),
                state.summary.loc['mean']['roc_auc'], state.summary.loc['std']['roc_auc'], state.summary.loc['mean']['pr_auc'], state.summary.loc['std']['pr_auc'])
        else:
            cv_plain_text = """
                When using {} cross-validation approach (n_splits={}) to classify {} vs. {}, we achieved a receiver operating characteristic (ROC)
                with an average AUC (area under the curve) of {:.2f} ({:.2f} std) and Precision-Recall (PR) Curve with an average AUC of {:.2f} ({:.2f} std).
            """
            text += cv_plain_text.format(state.cv_method, state.cv_splits, ''.join(state.class_0), ''.join(state.class_1),
                state.summary.loc['mean']['roc_auc'], state.summary.loc['std']['roc_auc'], state.summary.loc['mean']['pr_auc'], state.summary.loc['std']['pr_auc'])

        if state.cohort_column != 'None':
            text += 'When training on one cohort and predicting on another to classify {} vs. {}, we achieved the following AUCs: '.format(''.join(state.class_0), ''.join(state.class_1))
            for i, cohort_combo in enumerate(state.cohort_combos):
                text+= '{:.2f} when training on {} and predicting on {} '.format(state.cohort_results[i]['roc_auc'], cohort_combo[0], cohort_combo[1])
                text+= ', and {:.2f} for PR Curve when training on {} and predicting on {}. '.format(state.cohort_results[i]['pr_auc'], cohort_combo[0], cohort_combo[1])

    # Print the all text
    st.info(text)

# Create new list and dict for sessions
@st.cache(allow_output_mutation=True)
def get_sessions():
    return [], {}

# Saving session info
def save_sessions(widget_values, user_name):

    session_no, session_dict = get_sessions()
    session_no.append(len(session_no) + 1)
    session_dict[session_no[-1]] = widget_values
    sessions_df = pd.DataFrame(session_dict)
    sessions_df = sessions_df.T
    sessions_df = sessions_df.drop(sessions_df[sessions_df["user"] != user_name].index).reset_index(drop=True)
    new_column_names = {k:v.replace(":", "").replace("Select", "") for k,v in zip(sessions_df.columns,sessions_df.columns)}
    sessions_df = sessions_df.rename(columns=new_column_names)
    sessions_df = sessions_df.drop("user", axis=1)

    st.write("## Session History")
    st.dataframe(sessions_df.T.style.set_precision(4)) # Display only 3 decimal points in UI side
    get_download_link(sessions_df, "session_history.csv")

# Generate footer
def generate_footer_parts():

    # Citations
    citations = """
        <br> <b>APA Format:</b> <br>
        Winter, S., Karayel, O., Strauss, M., Padmanabhan, S., Surface, M., & Merchant, K. et al. (2020).
        Urinary proteome profiling for stratifying patients with familial Parkinson’s disease. doi: 10.1101/2020.08.09.243584.
    """

    # Put the footer with tabs
    footer_parts_html = """
        <div class="tabs">
            <div class="tab"> <input type="radio" id="tab-1" name="tab-group-1" checked> <label for="tab-1">Citations</label> <div class="content"> <p> {} </p> </div> </div>
            <div class="tab"> <input type="radio" id="tab-2" name="tab-group-1"> <label for="tab-2">Report Bugs</label> <div class="content">
                <p><br>
                    Firstly, thank you very much for taking your time and we appreciate all contributions. 👍 <br>
                    You can report the bugs or request a feature using the link below or sending us a e-mail:
                    <br><br>
                    <a class="download_link" href="https://github.com/OmicEra/OmicLearn/issues/new/choose" target="_blank">Report a bug via GitHub</a>
                    <a class="download_link" href="mailto:info@omicera.com">Report a bug via Email</a>
                </p>
            </div> </div>
        </div>

        <div class="footer">
            <i> OmicLearn {} </i> <br> <img src="https://omicera.com/wp-content/uploads/2020/05/cropped-oe-favicon-32x32.jpg" alt="OmicEra Diagnostics GmbH">
            <a href="https://omicera.com" target="_blank">OmicEra</a>.
        </div>
        """.format(citations, version)

    st.write("## Cite us & Report bugs")
    st.markdown(footer_parts_html, unsafe_allow_html=True)

# Main Function
def OmicLearn_Main():

    state = objdict()

    state['df'] = pd.DataFrame()
    state['class_0'] = None
    state['class_1'] = None

    # Main components
    widget_values, record_widgets = main_components()

    # Welcome text and Data uploading
    state = main_text_and_data_upload(state)

    # Checkpoint for whether data uploaded/selected
    state = checkpoint_for_data_upload(state, record_widgets)

    # Sidebar widgets
    state = generate_sidebar_elements(state, record_widgets)

    # Analysis Part
    if (state.df is not None) and (state.class_0 and state.class_1) and (st.button('Run Analysis', key='run')):

        state.features = state.proteins + state.additional_features

        class_names = [state.df[state.target_column].value_counts().index[0], state.df_sub[state.target_column].value_counts().index[1]]
        st.markdown("Using the following features: Class 0 `{}`, Class 1 `{}`".format(state.class_0, state.class_1))
        subset = state.df_sub[state.df_sub[state.target_column].isin(state.class_0) | state.df_sub[state.target_column].isin(state.class_1)].copy()

        state.y = subset[state.target_column].isin(state.class_0) #is class 0 will be 1!
        state.X = transform_dataset(subset, state.additional_features, state.proteins)

        if state.cohort_column is not None:
            state['X_cohort'] = subset[state.cohort_column]

        st.markdown('Using classifier `{}`.'.format(state.classifier))
        st.markdown(f'Using a total of  `{len(state.features)}` features.')

        if len(state.features) < 10:
            st.markdown(f'Features `{state.features}`.')

        # Plotting and Get the results
        state = classify_and_plot(state)

        # Generate summary text
        generate_text(state)

        # Session and Run info
        widget_values["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " (UTC)"

        for _ in state.summary.columns:
            widget_values[_+'_mean'] = state.summary.loc['mean'][_]
            widget_values[_+'_std'] = state.summary.loc['std'][_]

        user_name = str(random.randint(0,10000)) + "OmicLearn"
        session_state = session_states.get(user_name=user_name)
        widget_values["user"] = session_state.user_name
        save_sessions(widget_values, session_state.user_name)

        # Generate footer
        generate_footer_parts()            

    else:
        if len(state.df) > 0:
            if (state.class_0 is None) or (state.class_1 is None):
                st.error('Start with defining classes.')


# Run the OmicLearn
if __name__ == '__main__':
    try:
        OmicLearn_Main()
    except (ValueError, IndexError) as val_ind_error:
        st.error("There is a problem with values/parameters or dataset due to {}.".format(val_ind_error))
    except TypeError as e:
        # st.warning("TypeError exists in {}".format(e))
        pass
