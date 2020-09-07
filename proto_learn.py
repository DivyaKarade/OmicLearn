import random
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime
import utils.session_states as session_states
from utils.helper import get_svg_download_link, get_pdf_download_link, get_csv_download_link
from utils.helper import make_recording_widget, load_data, transform_dataset, normalize_dataset
from utils.helper import select_features, plot_feature_importance, impute_nan, perform_cross_validation, plot_confusion_matrices
from utils.helper import perform_cohort_validation, plot_roc_curve_cv, plot_roc_curve_cohort, plot_pr_curve_cv, plot_pr_curve_cohort, get_system_report
icon = Image.open('./utils/proto_learn.png')

# Checkpoint for XGBoost
xgboost_installed = False
try:
    from xgboost import XGBClassifier
    import xgboost
    xgboost_installed = True
except ModuleNotFoundError:
    st.error('Xgboost not installed. To use xgboost install using `conda install py-xgboost`')

# Define versions
report = get_system_report()
version = report['proto_learn_version']

# Functions / Element Creations
def main_components():
    # External CSS
    main_external_css = """
        <style>
            #MainMenu, .reportview-container .main footer {display: none;}
            .download_link {color: #f63366 !important; text-decoration: none !important; z-index: 99999 !important; 
                            cursor:pointer !important; margin: 15px 0px; border: 1px solid #f63366; 
                            text-align:center; padding: 8px !important; width: 200px;}
            .download_link:hover {background: #f63366 !important; color: #FFF !important;}
            h1, h2, h3, h4, h5, h6, a, a:visited {color: #f84f57 !important}
            label, stText, p, .caption {color: #035672 }
            .sidebar .sidebar-content {background: #035672 !important;}
            .sidebar-content label, stText, p, .caption {color: #FFF !important}
            .tickBarMin, .tickBarMax {color: #f84f57 !important}
            .markdown-text-container p {color: #035672 !important}
        </style>
    """
    st.markdown(main_external_css, unsafe_allow_html=True)

    widget_values = {}
    n_missing = 0
    class_0, class_1 = None, None

    # Sidebar widgets
    button_ = make_recording_widget(st.sidebar.button, widget_values)
    slider_ = make_recording_widget(st.sidebar.slider, widget_values)
    multiselect_ = make_recording_widget(st.sidebar.multiselect, widget_values)
    number_input_ = make_recording_widget(st.sidebar.number_input, widget_values)
    selectbox_ = make_recording_widget(st.sidebar.selectbox, widget_values)
    multiselect = make_recording_widget(st.multiselect, widget_values)

    return widget_values, n_missing, class_0, class_1, button_, slider_, multiselect_, number_input_, selectbox_, multiselect

def main_text_and_data_upload():
    st.title("DEV | Proto Learn — Clinical Proteomics Machine Learning Tool")
    st.info(""" 
        * Upload your excel / csv file here. Maximum size is 200 Mb.
        * Each row corresponds to a sample, each column to a feature
        * Protein names should be uppercase
        * Additional features should be marked with a leading '_'
    """)
    st.subheader("Dataset")
    file_buffer = st.file_uploader("Upload your dataset below", type=["csv", "xlsx"])
    delimiter = st.selectbox("Determine the delimiter in your dataset", ["Excel File", "Comma (,)", "Semicolon (;)"])
    sample_file = st.selectbox("Or select sample file here:", ["None", "Sample"])
    df = load_data(file_buffer, delimiter)
    return sample_file, df

def checkpoint_for_data_upload(sample_file, df, class_0, class_1, n_missing, multiselect):
    if (sample_file != 'None') and (len(df) > 0):
        st.warning("Please, either choose a sample file or set it as `None` to work on your file")
        df = pd.DataFrame()
    elif sample_file != 'None':
        st.text("Here is the sample dataset:")
        df = pd.read_excel('data/'+ sample_file + '.xlsx')
        st.write(df)
    elif len(df) > 0:
        st.text("Here is your dataset:")
        st.write(df)
    else:
        st.error('No dataset uploaded or selected.')

    n_missing = df.isnull().sum().sum()

    if len(df) > 0:
        if n_missing > 0:
            st.warning('Found {} missing values. Use missing value imputation or xgboost classifier.'.format(n_missing))

        # Distinguish the proteins from others
        proteins = [_ for _ in df.columns.to_list() if _[0] != '_']
        not_proteins = [_ for _ in df.columns.to_list() if _[0] == '_']

        st.subheader("Subset")
        st.text('Create a subset based on values in the selected column')
        subset_column = st.selectbox("Select subset column:", ['None']+not_proteins)

        if subset_column != 'None':
            subset_options = df[subset_column].value_counts().index.tolist()
            subset_class = multiselect("Select values to keep:", subset_options, default=subset_options)
            df_sub = df[df[subset_column].isin(subset_class)].copy()
        else:
            df_sub = df.copy()

        st.subheader("Features")
        option = st.selectbox("Select target column:", not_proteins)
        st.markdown("Unique elements in `{}` column.".format(option))
        unique_elements = df_sub[option].value_counts()
        st.write(unique_elements)
        unique_elements_lst = unique_elements.index.tolist()

        # Define classes
        st.subheader("Define classes".format(option))
        class_0 = multiselect("Select Class 0:", unique_elements_lst, default=None)
        class_1 = multiselect("Select Class 1:", [_ for _ in unique_elements_lst if _ not in class_0], default=None)
        remainder = [_ for _ in not_proteins if _ is not option]

        # Define `exclude_features` and `additional_features` as empty if the classes are not defined
        exclude_features, additional_features = "", ""
        if class_0 and class_1:
            st.subheader("Additional features")
            st.text("Select additional features. All non numerical values will be encoded (e.g. M/F -> 0,1)")
            additional_features = st.multiselect("Select additional features for trainig:", remainder, default=None)
            #Todo: Check if we need additional features
            st.subheader("Exclude proteins")
            exclude_features = st.multiselect("Select proteins that should be excluded:", proteins, default=None)

        st.subheader("Cohort comparison")
        st.text('Select cohort column to train on one and predict on another:')
        not_proteins_excluded_target_option = not_proteins
        not_proteins_excluded_target_option.remove(option)
        cohort_column = st.selectbox("Select cohort column:", ['None'] + not_proteins_excluded_target_option)

        return class_0, class_1, df, unique_elements_lst, cohort_column, exclude_features, remainder, proteins, not_proteins, option, df_sub, additional_features, n_missing, subset_column

def generate_sidebar_elements(multiselect_, slider_, selectbox_, number_input_, n_missing, additional_features, proteins):
    st.sidebar.image(icon, use_column_width=True, caption="Proto Learn " + version)
    st.sidebar.title("Options")
    random_state = slider_("Random State:", min_value = 0, max_value = 99, value=23)
    st.sidebar.markdown('## [Preprocessing](https://github.com/OmicEra/proto_learn/wiki/METHODS-%7C-1.-Preprocessing)')
    normalizations = ['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer']
    normalization = selectbox_("Normalization method:", normalizations)

    # Define these two variables if normalization is not these two:
    normalization_detail, n_quantiles = "", ""
    
    if normalization == "PowerTransformer":
        normalization_detail = selectbox_("Power transformation method:", ["Yeo-Johnson", "Box-Cox"])
    elif normalization == "QuantileTransformer":
        n_quantiles = number_input_("Number of quantiles:", value = 100, min_value = 1, max_value = 2000)
        normalization_detail = selectbox_("Output distribution method:", ["Uniform", "Normal"])

    if n_missing > 0:
        st.sidebar.markdown('## [Missing value imputation](https://github.com/OmicEra/proto_learn/wiki/METHODS-%7C-1.-Preprocessing#1-2-imputation-of-missing-values)')
        missing_values = ['Zero', 'Mean', 'Median', 'IterativeImputer', 'KNNImputer', 'None']
        missing_value = selectbox_("Missing value imputation", missing_values)
    else:
        missing_value = 'None'

    st.sidebar.markdown('## [Feature selection](https://github.com/OmicEra/proto_learn/wiki/METHODS-%7C-2.-Feature-selection)')
    feature_methods = ['ExtraTrees', 'k-best (mutual_info_classif)','k-best (f_classif)', 'k-best (chi2)', 'Manual']
    feature_method = selectbox_("Feature selection method:", feature_methods)

    if feature_method != 'Manual':
        max_features = number_input_('Maximum number of features:', value = 20, min_value = 1, max_value = 2000)
    else:
        # Define `max_features` as 0 if `feature_method` is `Manual` 
        max_features = 0

    if feature_method == "ExtraTrees":
        n_trees = number_input_('Number of trees in the forest:', value = 100, min_value = 1, max_value = 2000)
    else:
        n_trees = 0

    st.sidebar.markdown('## [Classification](https://github.com/OmicEra/proto_learn/wiki/METHODS-%7C-3.-Classification#3-classification)')

    if xgboost_installed:
        classifiers = ['AdaBoost','LogisticRegression','KNeighborsClassifier','RandomForest','DecisionTree','LinearSVC','XGBoost']
    else:
        classifiers = ['AdaBoost','LogisticRegression','KNeighborsClassifier','RandomForest','DecisionTree','LinearSVC']

    if n_missing > 0:
        if missing_value == 'None':
            classifiers = ['XGBoost']

    classifier = selectbox_("Classifier", classifiers)

    # Define variables as 0 if classifier not those:
    n_estimators, learning_rate, n_neighbors, penalty, solver, max_iter, c_val = 0, 0, 0, "", "", 0, 0

    if classifier == 'AdaBoost':
        n_estimators = number_input_('Number of estimators:', value = 100, min_value = 1, max_value = 2000)
        learning_rate = number_input_('Learning rate:', value = 1, min_value = 1, max_value = 100)

    if classifier == 'KNeighborsClassifier':
        n_neighbors = number_input_('Number of neighbors:', value = 100, min_value = 1, max_value = 2000)

    if classifier == 'LogisticRegression':
        penalty = selectbox_("Specify norm in the penalization:", ["l2", "l1", "ElasticNet", "None"])
        solver = selectbox_("Select the algorithm for optimization:", ["lbfgs", "newton-cg", "liblinear", "sag", "saga"])
        max_iter = number_input_('Maximum number of iteration:', value = 100, min_value = 1, max_value = 2000)
        c_val = number_input_('C parameter:', value = 1, min_value = 1, max_value = 100)

    st.sidebar.markdown('## [Cross Validation](https://github.com/OmicEra/proto_learn/wiki/METHODS-%7C-4.-Cross-Validation)')
    cv_splits = number_input_('CV Splits:', min_value = 2, max_value = 10, value=5)
    cv_repeats = number_input_('CV Repeats:', min_value = 1, max_value = 50, value=10)

    features_selected = False

    # Define manual_features and features as empty if method is not Manual
    manual_features, features = "", []

    if feature_method == 'Manual':
        st.sidebar.subheader("Manually select proteins")
        manual_features = multiselect_("Select your proteins manually:", proteins, default=None)
        features = manual_features +  additional_features
        
    return random_state, normalization, normalization_detail, n_quantiles, missing_value, feature_method, max_features, n_trees, classifiers, n_estimators, learning_rate, n_neighbors, penalty, solver, max_iter, c_val,  cv_splits, cv_repeats, features_selected, classifier, manual_features, features

def feature_selection(df, option, class_0, class_1, df_sub, features, manual_features, additional_features, proteins, normalization, normalization_detail, n_quantiles, feature_method, max_features, n_trees, random_state):
    st.subheader("Feature selection")
    class_names = [df[option].value_counts().index[0], df_sub[option].value_counts().index[1]]
    st.markdown("Using the following identifiers: Class 0 `{}`, Class 1 `{}`".format(class_0, class_1))
    subset = df_sub[df_sub[option].isin(class_0) | df_sub[option].isin(class_1)].copy()

    st.write(subset[option].value_counts())
    y = subset[option].isin(class_0) #is class 0 will be 1!
    X = transform_dataset(subset, additional_features, proteins)
    X = normalize_dataset(X, normalization, normalization_detail, n_quantiles, random_state)

    if feature_method == 'Manual':
        features = manual_features +  additional_features
        pass
    else:
        features, feature_importance, p_values = select_features(feature_method, X, y, max_features, n_trees, random_state)
        p, feature_df = plot_feature_importance(features, feature_importance, p_values)
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_pdf_download_link(p, 'feature_importance.pdf')
            get_svg_download_link(p, 'feature_importance.svg')
        # st.dataframe(feature_df)
    
    return class_names, subset, X, y, features

def all_plotting_and_results(X, y, subset, cohort_column, classifier, random_state, n_estimators, learning_rate, n_neighbors, penalty, solver, max_iter, c_val, cv_splits, cv_repeats, class_0, class_1):
    
    # Cross-Validation                
    st.markdown("Running Cross-Validation")
    _cv_results, roc_curve_results, pr_curve_results, split_results, y_test = perform_cross_validation(X, y, classifier, cv_splits, cv_repeats, random_state, n_estimators, learning_rate, n_neighbors, penalty, solver, max_iter, c_val,  st.progress(0))
    st.header('Cross-Validation')

    # ROC-AUC
    st.subheader('Receiver operating characteristic')
    p = plot_roc_curve_cv(roc_curve_results)
    st.plotly_chart(p)
    if p:
        get_pdf_download_link(p, 'roc_curve.pdf')
        get_svg_download_link(p, 'roc_curve.svg')

    # Precision-Recall Curve
    st.subheader('Precision-Recall Curve')
    st.text("Precision-Recall (PR) Curve might be used for imbalanced datasets.")
    p = plot_pr_curve_cv(pr_curve_results, y_test)
    st.plotly_chart(p)
    if p:
        get_pdf_download_link(p, 'pr_curve.pdf')
        get_svg_download_link(p, 'pr_curve.svg')

    # Confusion Matrix (CM)
    st.subheader('Confusion matrix')
    names = ['CV_split {}'.format(_+1) for _ in range(len(split_results))]
    names.insert(0, 'Sum of all splits')
    p  = plot_confusion_matrices(class_0, class_1, split_results, names)
    st.plotly_chart(p)
    if p:
        get_pdf_download_link(p, 'cm_cohorts.pdf')
        get_svg_download_link(p, 'cm_cohorts.svg')

    # Results
    st.subheader('Run Results for `{}`'.format(classifier))
    summary = pd.DataFrame(_cv_results).describe()
    summary_df = pd.DataFrame(summary)
    st.write(summary_df)
    get_csv_download_link(summary_df, "run_results.csv")

    if cohort_column != 'None':
        st.header('Cohort comparison')
        _cohort_results, roc_curve_results_cohort, pr_curve_results_cohort, cohort_results, cohort_combos, y_test = perform_cohort_validation(X, y, subset, cohort_column, classifier, random_state, n_estimators, learning_rate, n_neighbors, penalty, solver, max_iter, c_val, st.progress(0))

        # ROC-AUC for Cohorts
        st.subheader('Receiver operating characteristic')
        p = plot_roc_curve_cohort(roc_curve_results_cohort, cohort_combos)
        st.plotly_chart(p)
        if p:
            get_pdf_download_link(p, 'roc_curve_cohort.pdf')
            get_svg_download_link(p, 'roc_curve_cohort.svg')

        # PR Curve for Cohorts
        st.subheader('Precision-Recall Curve')
        st.text("Precision-Recall (PR) Curve might be used for imbalanced datasets.")
        p = plot_pr_curve_cohort(pr_curve_results_cohort, cohort_combos, y_test)
        st.plotly_chart(p)
        if p:
            get_pdf_download_link(p, 'pr_curve_cohort.pdf')
            get_svg_download_link(p, 'pr_curve_cohort.svg')

        st.subheader('Confusion matrix')
        names = ['Train on {}, Test on {}'.format(_[0], _[1]) for _ in cohort_combos]
        names.insert(0, 'Sum of cohort comparisons')
        
        # Confusion Matrix (CM) for Cohorts
        p = plot_confusion_matrices(class_0, class_1, cohort_results, names)
        st.plotly_chart(p)
        if p:
            get_pdf_download_link(p, 'cm.pdf')
            get_svg_download_link(p, 'cm.svg')

        st.subheader('Run Results for `{}`'.format(classifier))
        summary = pd.DataFrame(_cohort_results).describe()
        st.write(pd.DataFrame(summary))
    else:
        # Set these values as empty if cohort_column is `None`
        _cohort_results, roc_curve_results_cohort, cohort_results, cohort_combos = "", "", "", ""

    return summary, _cohort_results, roc_curve_results_cohort, cohort_results, cohort_combos

def generate_text(normalization, normalization_detail, n_quantiles, proteins, feature_method, n_trees, classifier, cohort_column, cv_repeats, cv_splits, class_0, class_1, summary, _cohort_results, cohort_combos):
    st.write("## Summary")
    text ="```"
    
    # Packages
    text += "Proto Learn ({proto_learn_version}) was utilized for performing the data analysis, model execution and generating the plots and charts. Machine learning was done in Python ({python_version}). Protein tables were imported via the Pandas package ({pandas_version}) together with Numpy package ({numpy_version}). The machine learning pipeline was employed using the scikit-learn package ({sklearn_version}). For generating the plots and charts, Plotly ({plotly_version}) library was used. ".format(**report)
    
    # Normalization
    if normalization == 'None':
        text += 'After importing, no further normalization was performed. '
    else:
        if n_quantiles != "":
            text += 'After importing, features were normalized using a {} ({} as output distribution method and n_quantiles={}) approach. '.format(normalization, normalization_detail, n_quantiles)
        elif normalization_detail != "":
            text += 'After importing, features were normalized using a {} ({}) approach. '.format(normalization, normalization_detail)
        else:
            text += 'After importing, features were normalized using a {} approach. '.format(normalization)

    # Feature
    if feature_method == 'Manual':
        text += 'A total of {} proteins were manually selected. '.format(len(proteins))
    elif feature_method == 'ExtraTrees':
        text += 'Proteins were selected using a {} (n_trees={}) strategy. '.format(feature_method, n_trees)
    else:
        text += 'Proteins were selected using a {} strategy. '.format(feature_method)

    # Classifier
    if classifier is not 'XGBoost':
        text += 'For classification, we used a {}-Classifier. '.format(classifier)
    else:
        text += 'For classification, we used a {}-Classifier ({}). '.format(classifier, xgboost.__version__ )

    # Cross-Validation
    text += 'When using a repeated (n_repeats={}), stratified cross-validation (n_splits={}) approach to classify {} vs. {}, we achieved a receiver operating characteristic (ROC) with an average AUC (area under the curve) of {:.2f} ({:.2f} std). '.format(cv_repeats, cv_splits, ''.join(class_0), ''.join(class_1), summary.loc['mean']['roc_auc'], summary.loc['std']['roc_auc'])

    if cohort_column is not 'None':
        text += 'When training on one cohort and predicting on another to classify {} vs. {}, we achieved the following AUCs: '.format(''.join(class_0), ''.join(class_1))
        for i, cohort_combo in enumerate(cohort_combos):
            text+= '{:.2f} when training on {} and predicting on {}. '.format(pd.DataFrame(_cohort_results).iloc[i]['roc_auc'], cohort_combo[0], cohort_combo[1])
    text +="```"
    st.markdown(text)


# Saving session info
@st.cache(allow_output_mutation=True)
def get_sessions():
    return [], {}

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
    st.dataframe(sessions_df)
    get_csv_download_link(sessions_df, "session_history.csv")

# Main Function
def ProtoLearn_Main():
    # Main components
    widget_values, n_missing, class_0, class_1, button_, slider_, multiselect_, number_input_, selectbox_, multiselect = main_components()

    # Welcome text and Data uploading 
    sample_file, df = main_text_and_data_upload()

    # Checkpoint for whether data uploaded/selected
    class_0, class_1, df, unique_elements_lst, cohort_column, exclude_features, \
    remainder, proteins, not_proteins, option, df_sub, additional_features, \
    n_missing, subset_column = checkpoint_for_data_upload(sample_file, df, class_0, class_1, n_missing, multiselect)

    # Proteins selection
    proteins = [_ for _ in proteins if _ not in exclude_features]

    # Sidebar widgets
    random_state, normalization, normalization_detail, n_quantiles, missing_value, feature_method, max_features, n_trees, classifiers, \
    n_estimators, learning_rate, n_neighbors, penalty, solver, max_iter, c_val, cv_splits, cv_repeats, features_selected, classifier, \
    manual_features, features = generate_sidebar_elements(multiselect_, slider_, selectbox_, number_input_, n_missing, additional_features, proteins)

    # Analysis Part
    if (df is not None) and (class_0 and class_1) and (st.button('Run Analysis', key='run')):

        # Feature Selection
        class_names, subset, X, y, features = feature_selection(df, option, class_0, class_1, df_sub, features, manual_features, additional_features, proteins, normalization, normalization_detail, n_quantiles, feature_method, max_features, n_trees, random_state)
        st.markdown('Using classifier `{}`.'.format(classifier))
        st.markdown('Using features `{}`.'.format(features))
        # result = cross_validate(model, X=_X, y=_y, groups=_y, cv=RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=0) , scoring=metrics, n_jobs=-1)

        # Define X vector and impute the NaN values
        X = X[features]
        X = impute_nan(X, missing_value, random_state)

        # Plotting and Get the results
        summary, _cohort_results, roc_curve_results_cohort, \
        cohort_results, cohort_combos = all_plotting_and_results(X, y, subset, cohort_column, classifier, random_state, n_estimators, learning_rate, n_neighbors, penalty, solver, max_iter, c_val, cv_splits, cv_repeats, class_0, class_1)

        # Generate summary text
        generate_text(normalization, normalization_detail, n_quantiles, proteins, feature_method, n_trees, classifier, cohort_column, cv_repeats, cv_splits, class_0, class_1, summary, _cohort_results, cohort_combos)
        
        # Session and Run info
        widget_values["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " (UTC)"
        widget_values["ROC AUC Mean"] = summary.loc['mean']['roc_auc']
        widget_values["ROC AUC Std"] = summary.loc['std']['roc_auc']
        widget_values["Precision Mean"] = summary.loc['mean']['precision']
        widget_values["Precision Std"] = summary.loc['std']['precision']
        widget_values["Recall Mean"] = summary.loc['mean']['recall']
        widget_values["Recall Std"] = summary.loc['std']['recall']
        widget_values["F1 Score Mean"] = summary.loc['mean']['f1']
        widget_values["F1 Score Std"] = summary.loc['std']['f1']
        widget_values["Balanced Accuracy Mean"] = summary.loc['mean']['balanced_accuracy']
        widget_values["Balanced Accuracy Std"] = summary.loc['std']['balanced_accuracy']
        user_name = str(random.randint(0,10000)) + "ProtoLearn"
        session_state = session_states.get(user_name=user_name)
        widget_values["user"] = session_state.user_name
        save_sessions(widget_values, session_state.user_name)

# Run the Proto Learn
if __name__ == '__main__':
    try:
        ProtoLearn_Main()
    except (ValueError, IndexError) as val_ind_error:
        st.error("There is a problem with values/parameters or dataset due to {}.".format(val_ind_error))
    except TypeError as e:
        # st.warning("TypeError exists in {}".format(e))
        pass
