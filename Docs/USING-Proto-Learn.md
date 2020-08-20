## **Table of Contents**
- [Using Proto Learn](#using-proto-learn)
  * [Video for Using Proto Learn with Sample Dataset](#video-for-using-proto-learn-with-sample-dataset)
  * [Uploading or Selecting Dataset](#uploading-or-selecting-dataset)
  * [Configuring the Options and Tuning the Parameters](#configuring-the-options-and-tuning-the-parameters)
  * [Making Selections on the Dataset](#making-selections-on-the-dataset)
  * [Running the Workflow](#running-the-workflow)
  * [Getting Results & Plots](#getting-results---plots)
  * [Checking the Session History](#checking-the-session-history)

---

## Using Proto Learn
This documentation aims to provide detailed information for getting started with Proto Learn.

Here, in the figure below, the main steps for the workflow of Proto Learn are represented:

![Proto Learn Workflow](https://user-images.githubusercontent.com/49681382/90739663-62b38680-e2d7-11ea-83f0-3a9cf91e3374.png)

On this page, you can click on the titles listed above to see the detailed documentation for the section that explains the details. 

### Video for Using Proto Learn with Sample Dataset
Coming soon!

### Uploading or Selecting Dataset

- Proto Learn provides sample datasets in [`data/`](https://github.com/OmicEra/proto_learn/tree/master/data) folder to be performed analysis. So, you can easily select a sample file from the dropdown menu. 

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

- Also, within Proto Learn, you are able to upload the dataset directly via dropping or browsing your `.xlsx (Excel)` or `.csv (Comma-separated values)` file.

Here, there are some requirements and notes for data uploading:

> - The file format should be `.xlsx (Excel)` or `.csv (Comma-separated values)`. 
Here, for `.csv` file format, either `comma (,)` or `semicolon (;)` should be used as field seperator.
>
> - Maximum file size is 200 Mb.
>
> - Protein names should be UPPERCASE.
>
> - Additional features should be marked with a leading `'_'`.

![DATA_UPLOAD/SELECTION](https://user-images.githubusercontent.com/49681382/90772677-8ee1fe00-e2fd-11ea-89c4-5200ae166439.png)

_Figure 1: Uploading or Selecting Dataset_

Additionally, if your dataset contain missing values (`NaNs`), a warning will appear that you warn and want you to select a method for filling them.

![NAN_WARNING](https://user-images.githubusercontent.com/49681382/90772680-8ee1fe00-e2fd-11ea-8161-98630d750b31.png)

_Figure 2: Missing value warning_


### Configuring the Options and Tuning the Parameters
Proto Learn enables researchers and scientists to use the latest techniques in machine learning (ML) for clinical proteomics datasets.

A common ML pipeline includes `Preprocessing`, `Missing Value Imputation`, `Feature Selection`, `Classification`, and `Validation` of selected method/algorithms.

So, in Proto Learn, you have the possibility to select a method among several different choices and explore their effect on your data.

Also, you are able to set your own parameters for the methods/algorithms. For this purpose, [special pages on Wiki called `METHODS`](https://github.com/OmicEra/proto_learn/wiki/METHODS) are prepared to provide more detailed information about these methods and parameters.

As a note, once you make a change in any parameter or method, it asks you to re-run it. In addition, all these parameters that you previously selected are recorded. Please, have a look at [Checking the Session History](#checking-the-session-history) headline.

![Proto Learn SideBar](https://user-images.githubusercontent.com/49681382/90772676-8e496780-e2fd-11ea-8b61-9ac920959574.png)

_Figure 3: Proto Learn Side Bar Options_

### Making Selections on the Dataset

Here, you can choose the Features you would like work on or adjust the dataset for your goal.

![selections](https://user-images.githubusercontent.com/49681382/90772670-8d183a80-e2fd-11ea-81b9-ee72c3744e05.png)

_Figure 4: Selections on the Dataset_

### Running the Workflow
After tuning your hyperparameters and selecting your methods/algorithms together with making selections on your dataset, you are able to execute the workflow by clicking the `Run Analysis` button.

As a note, when you alter any parameter on the sidebar or change the selections on your dataset, it will ask you to re-run by clicking the same button.

### Getting Results & Plots
Once the analysis is completed, Proto Learn automatically generates the plots together with the result of analysis. 

For the plots, you are able to download them as `.pdf` and `.svg` format.

![plot](https://user-images.githubusercontent.com/49681382/90772681-8f7a9480-e2fd-11ea-878e-18848c85af15.png)

_Figure 5: Example Plot (ROC Curve - AUC) and Download Options_

Also, importantly, as seen in the figure below, Proto Learn generates a `Summary` text that can be used for `Methods` part in the papers.

![results](https://user-images.githubusercontent.com/49681382/90772684-8f7a9480-e2fd-11ea-8f5e-01fcf16b61e3.png)

_Figure 6: Results of Analysis_

### Checking the Session History
When you run the workflow and get some results, you can focus on your hyperparameters again and change some features or excluding some columns from the selection part for your dataset. 

So, at the end of each analysis, you can check the `Session History` table that keeps your previous parameters, settings, selections, and results of your analysis together with the date and time.

Hence, you have a chance to compare and contrast your previous runs with the current one and then decide which one to be used.

![session](https://user-images.githubusercontent.com/49681382/90772672-8e496780-e2fd-11ea-8515-2d3eace637d5.png)

_Figure 7: Session History Table_

