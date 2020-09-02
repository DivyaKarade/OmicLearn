## **Table of Contents**
- [Using Proto Learn](#using-proto-learn)
- [Uploading data](#uploading-data)
  * [Sample Datasets](#sample-datasets)
- [Sidebar: Selecting Parameters](#sidebar--selecting-parameters)
- [Main Window: Selecting data, define workflow and explore results](#main-window--selecting-data--define-workflow-and-explore-results)
  * [Data Selection](#data-selection)
  * [Running the Workflow](#running-the-workflow)
  * [Analysis results and plots](#analysis-results-and-plots)
  * [Checking the Session History](#checking-the-session-history)

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
> - Protein names should be UPPERCASE.
>
> - Additional features should be marked with a leading `'_'`.

![DATA_UPLOAD/SELECTION](https://user-images.githubusercontent.com/49681382/90772677-8ee1fe00-e2fd-11ea-89c4-5200ae166439.png)

_**Figure 2:** Uploading a Dataset_

The data will be checked for consistency, and if your dataset contains missing values (`NaNs`), a notification will appear.
![NAN_WARNING](https://user-images.githubusercontent.com/49681382/90772680-8ee1fe00-e2fd-11ea-8161-98630d750b31.png)

_**Figure 3:** Missing value warning_

### Sample Datasets

- Proto Learn has several sample [datasets](https://github.com/OmicEra/proto_learn/tree/master/data) included that can be used for exploring the analysis, which can be selected from the dropdown menu.

Here is the list of sample datasets available:

**`Sample Dataset 1`**
> 📁 **File Name:** Sample.xlsx
>
> 📖 **Description:** Some description
>
> 🔗 **Source:** Citation/Link

**`Sample Dataset 2`**
> 📁 **File Name:** Sample2.xlsx
>
> 📖 **Description:** Some description 2
>
> 🔗 **Source:** Citation/Link

**`Sample Dataset 3`**
> 📁 **File Name:** Sample3.xlsx
>
> 📖 **Description:** Some description 3
>
> 🔗 **Source:** Citation/Link

## Sidebar: Selecting Parameters

Proto Learn has a large variety of options to choose from which are detailed in the [methods wiki](https://github.com/OmicEra/proto_learn/wiki/METHODS).  The parameters can be selected in the sidebar.

![Proto Learn SideBar](https://user-images.githubusercontent.com/49681382/90772676-8e496780-e2fd-11ea-8b61-9ac920959574.png)

After changing parameters, you are asked to re-run the analysis. Each analysis result will be stored in the [session history](#checking-the-session-history).

_**Figure 4:** Proto Learn Side Bar Options_

## Main Window: Selecting data, define workflow and explore results

### Data Selection

After uploading the data, the data will be displayed within the Proto Learn window and can be explored. The dropdown menu `Subset` allows you to specify a subset of data based on values within a comma. This way, you can exclude data that should not be used at all.

Within `Features,` you should select the target column. This refers to the variable that the classifier should be able to distinguish. As we are performing a binary classification task, there are only two options for the outcome of the classifier. By assigning multiple values to a class, multiple combinations of classifications can be tested.

Furthermore, `Additional Features` can be selected. This refers to columns that are not proteins (not uppercase and have a leading underscore (`_`). 

The option `Cohort comparison` allows comparing results over different cohorts (i.e., train on one cohort and predict on another)

![selections](https://user-images.githubusercontent.com/49681382/90772670-8d183a80-e2fd-11ea-81b9-ee72c3744e05.png)

_**Figure 6:** Selections on the Dataset_

### Running the Workflow
After selecting all parameters you are able to execute the workflow by clicking the `Run Analysis` button.

### Analysis results and plots
Once the analysis is completed, Proto Learn automatically generates the plots together with a table showing the results of each validation run. The plots are downloadable as `.pdf` and `.svg` format.

![plot](https://user-images.githubusercontent.com/49681382/90772681-8f7a9480-e2fd-11ea-878e-18848c85af15.png)

_**Figure 7:** Example Plot (ROC Curve - AUC) and Download Options_

Proto Learn generates a `Summary` to describe the method. This can be used for a method section in a publication.

![results](https://user-images.githubusercontent.com/49681382/90772684-8f7a9480-e2fd-11ea-8f5e-01fcf16b61e3.png)

_**Figure 8:** Results of Analysis_

### Checking the Session History

Each analysis run will be appended to the `Session History` so that you can investigate the different results for different parameter sets.

![session](https://user-images.githubusercontent.com/49681382/90772672-8e496780-e2fd-11ea-8515-2d3eace637d5.png)

_**Figure 9:** Session History Table_