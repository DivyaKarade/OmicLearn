## **Table of Contents**
- [**Table of Contents**](#table-of-contents)
- [Using Proto Learn](#using-proto-learn)
- [Uploading data](#uploading-data)
  - [Sample Datasets](#sample-datasets)
- [Sidebar: Selecting Parameters](#sidebar-selecting-parameters)
- [Main Window: Selecting data, define workflow, and explore results](#main-window-selecting-data-define-workflow-and-explore-results)
  - [Data Selection](#data-selection)
  - [Running the Workflow](#running-the-workflow)
  - [Analysis results and plots](#analysis-results-and-plots)
  - [Checking the Session History](#checking-the-session-history)

---

## Using Proto Learn
Proto Learn enables researchers and scientists to explore the latest algorithms in machine learning (ML) for their usage in clinical proteomics.

The core steps of the pipeline are  `Preprocessing`, `Missing Value Imputation`, `Feature Selection`, `Classification`, and `Validation` of selected method/algorithms and are presented in the flowchart below:

![Proto Learn Workflow](https://user-images.githubusercontent.com/49681382/91734594-cb421380-ebb3-11ea-91fa-8acc8826ae7b.png)

_**Figure 1:** Main steps for the workflow of Proto Learn at a glance_

## Uploading data

Own data can be uploaded via dragging and dropping on the file menu or clicking the link.
The data should be formatted according to the following conventions:

> - The file format should be `.xlsx (Excel)` or `.csv (Comma-separated values)`.  For `.csv`, the separator should be either `comma (,)` or `semicolon (;)`.
>
> - Maximum file size is 200 Mb.
>
> - Protein names should be uppercase.
>
> - Each row corresponds to a sample, each column to a feature.
>
> - Additional features should be marked with a leading `'_'`.

![DATA_UPLOAD/SELECTION](https://user-images.githubusercontent.com/49681382/95564530-a0a37000-0a27-11eb-958a-41bc2f613915.png)

_**Figure 2:** Uploading a dataset or selecting a sample file_

The data will be checked for consistency, and if your dataset contains missing values (`NaNs`), a notification will appear.
![NAN_WARNING](https://user-images.githubusercontent.com/49681382/95565283-9b92f080-0a28-11eb-9ba0-61fcf94f5115.png)

_**Figure 3:** Missing value warning_


### Sample Datasets

- Proto Learn has several sample [datasets](https://github.com/OmicEra/proto_learn/tree/master/data) included that can be used for exploring the analysis, which can be selected from the dropdown menu.

Here is the list of sample datasets available:

**`1. Alzheimer Dataset`**
> 📁 **File Name:** Alzheimer.xlsx
>
> 📖 **Description:** Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease
>
> 🔗 **Source:** Bader, J., Geyer, P., Müller, J., Strauss, M., Koch, M., & Leypoldt, F. et al. (2020). Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease. Molecular Systems Biology, 16(6). doi: [10.15252/msb.20199356](http://doi.org/10.15252/msb.20199356).

**`2. Sample Dataset`**
> 📁 **File Name:** Sample.xlsx
>
> 📖 **Description:** Sample dataset for testing the tool
>
> 🔗 **Source:** -

## Sidebar: Selecting Parameters

Proto Learn has a large variety of options to choose from which are detailed in the [methods wiki](https://github.com/OmicEra/proto_learn/wiki/METHODS).  The parameters can be selected in the sidebar.

Moreover, after changing the parameters, you are asked to re-run the analysis. Each analysis result will be stored in the [session history](#checking-the-session-history).

![Proto Learn SideBar](https://user-images.githubusercontent.com/49681382/95566522-54a5fa80-0a2a-11eb-9502-b11b63ed358e.png)


_**Figure 4:** Proto Learn sidebar options_

## Main Window: Selecting data, define workflow, and explore results

### Data Selection

After uploading the data, the data will be displayed within the Proto Learn window and can be explored. The dropdown menu `Subset` allows you to specify a subset of data based on values within a comma. This way, you can exclude data that should not be used at all.

Within `Features,` you should select the target column. This refers to the variable that the classifier should be able to distinguish. As we are performing a binary classification task, there are only two options for the outcome of the classifier. By assigning multiple values to a class, multiple combinations of classifications can be tested.

Furthermore, `Additional Features` can be selected. This refers to columns that are not proteins (not uppercase and have a leading underscore (`_`). 

The option `Cohort comparison` allows comparing results over different cohorts (i.e., train on one cohort and predict on another)

![selections](https://user-images.githubusercontent.com/49681382/95566912-e150b880-0a2a-11eb-8b55-c7397a6e3e42.png)

_**Figure 6:** Selections on the dataset_

### Running the Workflow
After selecting all parameters you are able to execute the workflow by clicking the `Run Analysis` button.

### Analysis results and plots
Once the analysis is completed, Proto Learn automatically generates the plots together with a table showing the results of each validation run. The plots are downloadable as `.pdf` and `.svg` format in addition to the `.png` format provided by Plotly.

![plot](https://user-images.githubusercontent.com/49681382/95567275-62a84b00-0a2b-11eb-873a-1c50db32d9c8.png)

_**Figure 7:** Bar chart for selected features, its table containing links to UniProt and download options_

![CLF_Feature_Imp](https://user-images.githubusercontent.com/49681382/95567884-36d99500-0a2c-11eb-9cdd-4d9df200cb97.png)

_**Figure 8:** Bar chart for feature importance values received from the classifier, its table containing links to UniProt and download options_

![ROC-CURVE](https://user-images.githubusercontent.com/49681382/95567533-be72d400-0a2b-11eb-8646-3b271a7c4ee8.png)

![PR-CURVE](https://user-images.githubusercontent.com/49681382/95567509-b31fa880-0a2b-11eb-99e6-1c6af6ed191e.png)

_**Figure 9:** Receiver operating characteristic (ROC) Curve, Precision-Recall (PR) Curve and download options_

![CONF-MATRIX](https://user-images.githubusercontent.com/49681382/95567699-fe39bb80-0a2b-11eb-9340-4954af364e20.png)

_**Figure 10:** Confusion matrix, slider for looking at the other matrix tables and download options_

Proto Learn generates a `Summary` to describe the method. This can be used for a method section in a publication.

![results](https://user-images.githubusercontent.com/49681382/95567106-25dc5400-0a2b-11eb-8220-1a259c2feab9.png)

_**Figure 11:** Results table of the analysis and summary text_

### Checking the Session History

Each analysis run will be appended to the `Session History` so that you can investigate the different results for different parameter sets.

![session](https://user-images.githubusercontent.com/49681382/95568625-2544bd00-0a2d-11eb-9f13-912f54b4181c.png)

_**Figure 12:** Session history table and download option_
