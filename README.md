<p align="center"> <img src="https://user-images.githubusercontent.com/49681382/98436689-68f31b00-20ee-11eb-8fa4-f9836a1e7d4d.png" height="270" width="277" /> </p>
<h2 align="center"> 📰 Manual and Documentation is available at: <a href="https://github.com/OmicEra/OmicLearn/wiki" target="_blank">Omic Learn Wiki Page </a> </h2>
<h2 align="center"> 🟢 Omic Learn is now accessible through the website: <a href="http://omiclearn.com/" target="_blank">omiclearn.com</a> </h2>

![Omic Learn Tests](https://github.com/OmicEra/OmicLearn/workflows/Omic%20Learn%20Tests/badge.svg)
![Python Badges](https://img.shields.io/badge/Tested_with_Python-3.7-blue)
![Omic Learn Version](https://img.shields.io/badge/Release-v1.0.0-orange)
![Omic Learn Release](https://img.shields.io/badge/Release%20Date-November%202020-green)
![Omic Learn Server Status](https://img.shields.io/badge/Server%20Status-up-success)
![Omic Learn License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Omic Learn

Transparent exploration of machine learning approaches for omics datasets.

This Wiki aims to provide general background information for Machine Learning, and it's applicability for clinical proteomics.

On Wiki page, click on the sidebar on the right to get more information about individual processing steps.

## Installation & Running

> More details for [`Installation & Running`](https://github.com/OmicEra/OmicLearn/wiki/HOW-TO:-Installation-&-Running) is available [on Wiki pages](https://github.com/OmicEra/OmicLearn/wiki/HOW-TO:-Installation-&-Running).

- It is strongly recommended to install Omic Learn in its own environment using [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Python distributions.

  1. Open the console and create a new conda environment: `conda create --name omic_learn python=3.7`
  2. Activate the environment: `conda activate omic_learn` for Linux / Mac Os X / Windows
  
  
  > **Note:** Type following command for conda versions prior to `4.6`:
  >
  > `source activate omic_learn` for macOS and Linux
  >
  > `activate omic_learn` for Windows

  3. Redirect to the folder of choice and clone the repository: `git clone https://github.com/OmicEra/OmicLearn`
  4. Install the packages with `pip install -r requirements.txt`
  5. To be able to use Xgboost, install via conda: `conda install py-xgboost`

- After successfull instalattion, type the following command to run Omic Learn:

  `streamlit run omic_learn.py --browser.gatherUsageStats False`
  
  > `Running with Docker` option is also available. Please, check it out from [the Wiki pages](https://github.com/OmicEra/OmicLearn/wiki/INSTALLATION-%26-RUNNING/).
  
 - Then, the Omic Learn page will be accessible via [`http://localhost:8501`](http://localhost:8501).

## Getting Started with Omic Learn

![Omic Learn Workflow](https://user-images.githubusercontent.com/49681382/91734594-cb421380-ebb3-11ea-91fa-8acc8826ae7b.png)

Above, you can see the main steps for workflow of Omic Learn at a glance. 

To get started with Omic Learn, [a special page](https://github.com/OmicEra/OmicLearn/wiki/HOW-TO:-Using) is prepared for [`Using Omic Learn`](https://github.com/OmicEra/OmicLearn/wiki/HOW-TO:-Using). 

On this page, you can click on the titles listed in *Table of Contents* for jumping into to the detailed documentation for each section explaning them and allowing you to give a try for the steps in the workflow using the example dataset. 

## Contributing
Firstly, thank you very much for taking your time and we appreciate all contributions. 👍

📰 For the details, please check our [`CONTRIBUTING`](https://github.com/OmicEra/OmicLearn/blob/master/CONTRIBUTING.md) guideline out. 

When contributing to **Omic Learn**, please [open a new issue](https://github.com/OmicEra/OmicLearn/issues/new/choose) to report the bug or discuss the changes you plan before sending a PR (pull request).

Also, be aware that you agree to the `OmicEra Individual Contributor License Agreement` by submitting your code. 🤝
