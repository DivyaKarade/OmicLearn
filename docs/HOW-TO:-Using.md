## **Table of Contents**
- [Using OmicLearn](#using-omiclearn)
- [Uploading data](#uploading-data)
  - [Sample Datasets](#sample-datasets)
- [Sidebar: Selecting Parameters](#sidebar-selecting-parameters)
- [Main Window: Selecting data, define workflow and explore results](#main-window-selecting-data-define-workflow-and-explore-results)
  - [Data Selection](#data-selection)
  - [Running the Workflow](#running-the-workflow)
  - [Analysis results and plots](#analysis-results-and-plots)
  - [Checking the Session History](#checking-the-session-history)
- [Cite us and Report bugs](#cite-us--report-bugs)

---

## Using OmicLearn
[OmicLearn](http://omiclearn.com) enables researchers and scientists to explore the latest algorithms in machine learning (ML) for their usage in clinical proteomics.

The core steps of the pipeline are  `Preprocessing`, `Missing Value Imputation`, `Feature Selection`, `Classification`, and `Validation` of selected method/algorithms and are presented in the flowchart below:

![OmicLearn Workflow](https://user-images.githubusercontent.com/49681382/91734594-cb421380-ebb3-11ea-91fa-8acc8826ae7b.png)

_**Figure 1:** Main steps for the workflow of OmicLearn at a glance_

## Uploading data

Own data can be uploaded via dragging and dropping on the file menu or clicking the link.
The data should be formatted according to the following conventions:

> - The file format should be `.xlsx (Excel)` or `.csv (Comma-separated values)`.  For `.csv`, the separator should be either `comma (,)` or `semicolon (;)`.
>
> - Maximum file size is 200 Mb.
>
> - 'Identifiers' such as protein IDs, gene names, lipids or miRNA IDs should be uppercase.
>
> - Each row corresponds to a sample, each column to a feature.
>
> - Additional features should be marked with a leading underscore (`_`).

![DATA_UPLOAD/SELECTION](https://user-images.githubusercontent.com/49681382/95564530-a0a37000-0a27-11eb-958a-41bc2f613915.png)

_**Figure 2:** Uploading a dataset or selecting a sample file_

The data will be checked for consistency, and if your dataset contains missing values (`NaNs`), a notification will appear.
Then, you might consider using the methods listed on the left sidebar for the imputation of missing values.

![NAN_WARNING](https://user-images.githubusercontent.com/49681382/95565283-9b92f080-0a28-11eb-9ba0-61fcf94f5115.png)

_**Figure 3:** Missing value warning_


### Sample Datasets

OmicLearn has several sample [datasets](https://github.com/OmicEra/OmicLearn/tree/master/data) included that can be used for exploring the analysis, which can be selected from the dropdown menu.

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

OmicLearn has a large variety of options to choose from which are detailed in the [methods wiki](https://github.com/OmicEra/OmicLearn/wiki/METHODS).  The parameters can be selected in the sidebar.

Moreover, after changing the parameters, you are asked to re-run the analysis. Each analysis result will be stored in the [`Session History` section](#checking-the-session-history).

![OmicLearn SideBar](https://user-images.githubusercontent.com/49681382/106890995-2c3dbc80-66fb-11eb-901a-f44fec38d69c.png)

_**Figure 4:** OmicLearn sidebar options_

## Main Window: Selecting data, define workflow, and explore results

### Data Selection

After uploading the data, the data will be displayed within the OmicLearn window and can be explored. The dropdown menu `Subset` allows you to specify a subset of data based on values within a comma. This way, you can exclude data that should not be used at all.

![Subset](https://user-images.githubusercontent.com/49681382/106892179-c0f4ea00-66fc-11eb-90bb-53595e8dc124.png)

_**Figure 5:** Example usage for `Subset` section_

Within `Features,` you should select the target column. This refers to the variable that the classifier should be able to distinguish. As we are performing a binary classification task, there are only two options for the outcome of the classifier. By assigning multiple values to a class, multiple combinations of classifications can be tested.

![Classification target](https://user-images.githubusercontent.com/49681382/106891533-dd445700-66fb-11eb-8c42-322bdcbea432.png)

_**Figure 6:** `Classification target` section for selecting the target columns and `Define classes` section for assigning the classes_

Furthermore, `Additional Features` can be selected. This refers to columns that are not your identifiers such as protein IDs, gene names, lipids or miRNA IDs (not uppercase and have a leading underscore (`_`). 

![Add Features](https://user-images.githubusercontent.com/49681382/106891702-1f6d9880-66fc-11eb-9f65-d1623a278103.png)

_**Figure 7:** Sample case for `Additional Features` option_

The section `Exclude identifiers` enables users to exclude the identifiers. Also, the file uploading menu allows avoiding to do this job manually.

> To utilize this option, you should upload a CSV (comma `,` separated) file where each row corresponds to a feature to be excluded. Also, the file should include a header (title row).

![exclude_identifiers](https://user-images.githubusercontent.com/49681382/101819569-7e68c400-3b36-11eb-9fa3-a02dbc00207b.png)

_**Figure 8:** Selections on the dataset_

The option `Cohort comparison` allows comparing results over different cohorts (i.e., train on one cohort and predict on another)

![dataselections](https://user-images.githubusercontent.com/49681382/106892120-ae7ab080-66fc-11eb-9dac-2284b5a75296.png)

_**Figure 9:** Selections on the dataset_

### Running the Workflow
After selecting all parameters you are able to execute the workflow by clicking the `Run Analysis` button.

### Analysis results and plots
Once the analysis is completed, OmicLearn automatically generates the plots together with a table showing the results of each validation run. The plots are downloadable as `.pdf` and `.svg` format in addition to the `.png` format provided by Plotly.

![FeatAtt_Chart](https://user-images.githubusercontent.com/49681382/106892562-48daf400-66fd-11eb-9b05-5d765c283267.png)

![FeatAtt_Table](https://user-images.githubusercontent.com/49681382/106892621-5ee8b480-66fd-11eb-81e6-323f5da7eb3f.png)

_**Figure 10:** Bar chart for feature importance values received from the classifier after all cross-validation runs, its table containing links to NCBI search and download options_

![ROC Curve](https://user-images.githubusercontent.com/49681382/106893284-6197d980-66fe-11eb-9a78-d1891b32aacd.png)

![PR Curve](https://user-images.githubusercontent.com/49681382/106892788-a1aa8c80-66fd-11eb-9ff7-093dfc1b643b.png)

_**Figure 11:** Receiver operating characteristic (ROC) Curve, Precision-Recall (PR) Curve and download options_

![CONF-MATRIX](https://user-images.githubusercontent.com/49681382/106892883-c56dd280-66fd-11eb-9d37-40ddc16d1e67.png)

_**Figure 12:** Confusion matrix, slider for looking at the other matrix tables and download options_

OmicLearn generates a `Summary` to describe the method. This can be used for a method section in a publication.

![Results table](https://user-images.githubusercontent.com/49681382/106892958-e3d3ce00-66fd-11eb-8896-125a726cc817.png)

![summary text](https://user-images.githubusercontent.com/49681382/106893033-0665e700-66fe-11eb-9395-3eac604aa455.png)

_**Figure 13:** Results table of the analysis, its download option, and auto-generated `Summary` text_

### Checking the Session History

Each analysis run will be appended to the `Session History` so that you can investigate the different results for different parameter sets.

![session](https://user-images.githubusercontent.com/49681382/95568625-2544bd00-0a2d-11eb-9f13-912f54b4181c.png)

_**Figure 14:** Session history table and download option_

## Cite us & Report bugs

At the end of the analysis, you find a footnote for reporting bugs. Also, there is information on how to cite OmicLearn in your work if you find it useful.

![bug_report](https://user-images.githubusercontent.com/49681382/98796034-fb136000-241b-11eb-8c12-1fe3f8b053e0.png)

_**Figure 15:** Tabs for Citation and Bug Reporting_