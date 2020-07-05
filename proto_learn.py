import streamlit as st
import pandas as pd

try:
    from xgboost import XGBClassifier
    import xgboost
    xgboost_installed = True
except:
    xgboost_installed = False


from helper import make_recording_widget, load_data, transform_dataset, normalize_dataset
from helper import select_features, plot_feature_importance, impute_nan, perform_cross_validation, plot_confusion_matrices
from helper import perform_cohort_validation, plot_roc_curve_cv, plot_roc_curve_cohort, get_system_report

from helper import get_svg_download_link

from PIL import Image
icon = Image.open('proto_learn.png')

with open("__version__.py") as version_file:
    version = version_file.read().strip()

svg_export = True
try:
    True #Todo include argument
except:
    raise
    svg_export = False

#Todo: Parser for MaxQuant default files
#Check if proteins are really float
#ToDo: Include hyperparameters for other optimiziers
#Better import for excel sheets #
#SVG Export  Plot.background_fill_color and Plot.border_fill_color


blue_color = '#0068c9'
red_color = '#f63366'
gray_color ='#f3f4f7'




def main():

    plots = {}

    st.sidebar.image(icon, use_column_width=True)
    st.sidebar.text(version)

    widget_values = {}
    n_missing = 0

    class_0, class_1 = None, None

    #Sidebar widgets
    button_ = make_recording_widget(st.sidebar.button, widget_values)
    slider_ = make_recording_widget(st.sidebar.slider, widget_values)
    multiselect_ = make_recording_widget(st.sidebar.multiselect, widget_values)
    number_input_ = make_recording_widget(st.sidebar.number_input, widget_values)
    selectbox_ = make_recording_widget(st.sidebar.selectbox, widget_values)

    multiselect = make_recording_widget(st.multiselect, widget_values)

    st.title("Clinical Proteomics Machine Learning Tool")
    st.text("* Upload your excel / csv file here.")
    st.text("* Each row corresponds to a sample, each column to a feature")
    st.text("* Protein names should be uppercase")
    st.text("* Additional features should be marked with a leading '_'")

    st.sidebar.title("Options")
    st.subheader("Dataset")
    file_buffer = st.file_uploader("Upload dataset below", type=["csv", "xlsx"])
    sample_file = st.selectbox("Or select sample file here:", ['None','Sample'])

    df = load_data(file_buffer)

    if len(df) == 0:
        if sample_file != 'None':
            df = pd.read_excel('sample_data.xlsx')
            st.write(df)
        else:
            st.text('No dataset uploaded.')
    else:
        st.write(df)

    n_missing = df.isnull().sum().sum()

    if len(df) > 0:
        if n_missing > 0:
            st.text('Found {} missing values. Use missing value imputation or xgboost classifier.'.format(n_missing))

        proteins = [_ for _ in df.columns.to_list() if _[0] != '_']
        not_proteins = [_ for _ in df.columns.to_list() if _[0] == '_']

        st.subheader("Subset")
        st.text('Create a subset based on values in the selected column')
        subset_column = st.selectbox("Select subset column", ['None']+not_proteins)

        if subset_column != 'None':
            subset_options = df[subset_column].value_counts().index.tolist()
            subset_class = multiselect("Select values to keep", subset_options, default=subset_options)
            df_sub = df[df[subset_column].isin(subset_class)].copy()
        else:
            df_sub = df.copy()

        st.subheader("Features")
        option = st.selectbox("Select target column", not_proteins)
        st.text("Unique elements in {}".format(option))

        unique_elements = df_sub[option].value_counts()

        st.write(unique_elements)

        unique_elements_lst = unique_elements.index.tolist()

        st.subheader("Define classes".format(option))


        class_0 = multiselect("Class 0", unique_elements_lst, default=None)
        class_1 = multiselect("Class 1", [_ for _ in unique_elements_lst if _ not in class_0], default=None)

        remainder = [_ for _ in not_proteins if _ is not option]

        if class_0 and class_1:

            st.subheader("Additional features")
            st.text("Select additional Features. All non numerical values will be encoded (e.g. M/F -> 0,1)")

            additional_features = st.multiselect("Additional features for trainig", remainder, default=None)

            #Todo: Check if we need additional features

            st.subheader("Exclude proteins")
            exclude_features = st.multiselect("Select proteins that should be excluded", proteins, default=None)

        st.subheader("Cohort comparison")
        st.text('Select cohort column to train on one and predict on another')
        cohort_column = st.selectbox("Select cohort column", ['None']+not_proteins)


    random_state = st.sidebar.slider("RandomState", min_value = 0, max_value = 99, value=23)

    st.sidebar.markdown('## [Preprocessing](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)')
    normalizations = ['None', 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer(Gaussian)','QuantileTransformer(uniform)','Normalizer']
    normalization = selectbox_("Normalization", normalizations)

    if n_missing > 0:
        st.sidebar.markdown('## [Missing value imputation](https://scikit-learn.org/stable/modules/impute.html)')
        #missing_values = ['Zero', 'Mean', 'Median', 'IterativeImputer', 'KNNImputer', 'None']
        missing_values = ['Zero', 'Mean', 'Median', 'IterativeImputer', 'KNNImputer','None']
        missing_value = selectbox_("Missing value imputation", missing_values)
    else:
        missing_value = 'None'

    st.sidebar.markdown('## [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)')
    feature_methods = ['DecisionTree', 'k-best (mutual_info)','k-best (f_classif)', 'Manual']
    feature_method = selectbox_("Feature selection method", feature_methods)

    if feature_method != 'Manual':
        max_features = number_input_('Maximum number of features', value = 20, min_value = 1, max_value = 2000)

    st.sidebar.markdown('## [Classification](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html?highlight=classifiers)')

    if xgboost_installed:
        classifiers = ['AdaBoost','LogisticRegression','RandomForest','XGBoost','DecisionTree']
    else:
        classifiers = ['AdaBoost','LogisticRegression','RandomForest','DecisionTree']

    if n_missing > 0:
        if missing_value == 'None':
            classifiers = ['XGBoost']

    classifier = selectbox_("Classifier", classifiers)

    if classifier == 'AdaBoost':
        n_estimators = number_input_('number of estimators', value = 100, min_value = 1, max_value = 2000)


    st.sidebar.markdown('## [Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html)')
    cv_splits = number_input_('CV Splits', min_value = 2, max_value = 10, value=5)
    cv_repeats = number_input_('CV Repeats', min_value = 1, max_value = 50, value=10)

    features_selected = False

    if feature_method == 'Manual':
        manual_features = st.multiselect("Manually select proteins", proteins, default=None)
        features = manual_features +  additional_features

    if df is not None:
        if class_0 and class_1:
            if st.button('Run Analysis',key='run'):
                proteins = [_ for _ in proteins if _ not in exclude_features]

                st.subheader("Feature selection")

                class_names = [df[option].value_counts().index[0], df_sub[option].value_counts().index[1]]

                st.markdown("Using the following identifiers: Class 0 {}, Class 1 {}".format(class_0, class_1))
                subset = df_sub[df_sub[option].isin(class_0) | df_sub[option].isin(class_1)].copy()

                st.write(subset[option].value_counts())

                y = subset[option].isin(class_0) #is class 0 will be 1!
                X = transform_dataset(subset, additional_features, proteins)
                X = normalize_dataset(X, normalization)

                if feature_method == 'Manual':
                    pass
                else:
                    features, feature_importance, p_values = select_features(feature_method, X, y, max_features)
                    p = plot_feature_importance(features, feature_importance, p_values)
                    st.bokeh_chart(p, use_container_width=True)

                    if svg_export:
                        get_svg_download_link(p, 'features.svg')

                st.markdown('Using classifier {}'.format(classifier))
                #result = cross_validate(model, X=_X, y=_y, groups=_y, cv=RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=0) , scoring=metrics, n_jobs=-1)
                st.markdown('Using features {}'.format(features))

                X = X[features]
                X = impute_nan(X, missing_value, random_state)

                st.markdown("Running Cross-Validation")
                _cv_results, roc_curve_results, split_results = perform_cross_validation(X, y, classifier, cv_splits, cv_repeats, random_state, st.progress(0))

                st.header('Cross-Validation')
                st.subheader('Receiver operating characteristic')

                p = plot_roc_curve_cv(roc_curve_results)
                st.bokeh_chart(p)
                if svg_export:
                    get_svg_download_link(p, 'roc_curve.svg')


                st.subheader('Confusion matrix')
                #st.text('Performed on the last CV split')

                names = ['CV_split {}'.format(_+1) for _ in range(len(split_results))]

                st.bokeh_chart(plot_confusion_matrices(class_0, class_1, split_results, names))

                st.subheader('Run Results for {}'.format(classifier))

                summary = pd.DataFrame(_cv_results).describe()
                st.write(pd.DataFrame(summary))

                if cohort_column != 'None':
                    st.header('Cohort comparison')
                    st.subheader('Receiver operating characteristic',)

                    _cohort_results, roc_curve_results_cohort, cohort_results, cohort_combos = perform_cohort_validation(X, y, subset, cohort_column, classifier, random_state, st.progress(0))

                    p = plot_roc_curve_cohort(roc_curve_results_cohort, cohort_combos)
                    st.bokeh_chart(p)
                    if svg_export:
                        get_svg_download_link(p, 'roc_curve_cohort.svg')

                    st.subheader('Confusion matrix')

                    names = ['Train on {}, Test on {}'.format(_[0], _[1]) for _ in cohort_combos]

                    st.bokeh_chart(plot_confusion_matrices(class_0, class_1, cohort_results, names))

                    st.subheader('Run Results for {}'.format(classifier))

                    summary = pd.DataFrame(_cohort_results).describe()
                    st.write(pd.DataFrame(summary))


                st.write("## Summary")

                report = get_system_report()

                text ="```"

                # Packages

                text += "Machine learning was done in Python ({python_version}). Protein tables were imported via the pandas package ({pandas_version}). The machine learning pipeline was employed using the scikit-learn package ({sklearn_version}). ".format(**report)

                if normalization == 'None':
                    text += 'After importing, no further normalization was performed. '
                else:
                    text += 'After importing, features were normalized using a {} approach. '.format(normalization)

                if feature_method == 'Manual':
                    text += 'A total of {} proteins were manually selected. '.format(len(proteins))
                else:
                    text += 'Proteins were selected using a {} strategy. '.format(feature_method)

                # Classifier
                if classifier is not 'XGBoost':
                    text += 'For classification, we used a {}-Classifier. '.format(classifier)
                else:
                    text += 'For classification, we used a {}-Classifier ({}). '.format(classifier, xgboost.__version__ )

                # Cross-Validation

                text += 'When using a repeated (n_repeats={}), stratified cross-validation (n_splits={}) approach to classify {} vs. {}, we achieved a receiver operating characteristic (ROC) with an average AUC (area under the curve) of {:.2f} ({:.2f} std). '.format(cv_repeats, cv_splits, ''.join(class_0), ''.join(class_1), summary.loc['mean']['roc_auc'], summary.loc['std']['roc_auc'])
                #if len()

                if cohort_column is not 'None':

                    text += 'When training on one cohort and predicting on another to classify {} vs. {}, we achieved the following AUCs: '.format(''.join(class_0), ''.join(class_1))

                    for i, cohort_combo in enumerate(cohort_combos):
                        text+= '{:.2f} when training on {} and predicting on {}. '.format(pd.DataFrame(_cohort_results).iloc[i]['roc_auc'], cohort_combo[0], cohort_combo[1])

                text +="```"

                st.markdown(text)

if __name__ == '__main__':
    main()
