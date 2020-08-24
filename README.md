<p align="center"> <img src="https://user-images.githubusercontent.com/49681382/88778270-f9859b00-d190-11ea-8c55-eaa2f683aa78.png" height="240" width="277" /> </p>
<h2 align="center"> 📰 Manual and Documentation is available at: <a href="https://github.com/OmicEra/proto_learn/wiki" target="_blank">Proto Learn Wiki Page </a> </h2>
<h2 align="center"> 🟢 Proto Learn is now accessible through the website: <a href="http://proto-learn.com/" target="_blank">proto-learn.com</a> </h2>

![Proto Learn CI/CD](https://github.com/OmicEra/QC_Dashboard/workflows/QC_Dashboard_Workflow/badge.svg)
![Python Badges](https://img.shields.io/badge/Tested_with_Python-3.7-blue)
![Proto Learn Version](https://img.shields.io/badge/Release-v1.0.0-orange)
![Proto Learn Release](https://img.shields.io/badge/Release%20Date-September%202020-green)
![Proto Learn Views](https://img.shields.io/badge/Views-20k-blueviolet.svg)
![Proto Learn Server Status](https://img.shields.io/badge/Server%20Status-up-success)
![Proto Learn License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Proto Learn

Transparent exploration of machine learning approaches for clinical proteomics data sets.

This Wiki aims to provide general background information for Machine Learning, and it's applicability for clinical proteomics.

Click on the sidebar on the right to get more information about individual processing steps.

## Installation & Running

> More details for [`INSTALLATION & RUNNING`](https://github.com/OmicEra/proto_learn/wiki/INSTALLATION-%26-RUNNING/) is available [on Wiki pages](https://github.com/OmicEra/proto_learn/wiki/INSTALLATION-%26-RUNNING/).

- It is strongly recommended to install Proto Learn in its own environment using [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Python distributions.

  1. Open the console and create a new conda environment: `conda create --name proto_learn python=3.7`
  2. Activate the environment: `source activate proto_learn` for Linux / Mac Os X or `activate proto_learn` for Windows
  3. Redirect to the folder of choice and clone the repository: `git clone https://github.com/OmicEra/proto_learn`
  4. Install the packages with `pip install -r requirements.txt`
  5. To be able to use Xgboost, install via conda: `conda install py-xgboost`

- After successfull instalattion, type the following command to run Proto Learn:

  `streamlit run proto_learn.py --browser.gatherUsageStats False`
  
  > `Running with Docker` option is also available. Please, check it out from [the Wiki pages](https://github.com/OmicEra/proto_learn/wiki/INSTALLATION-%26-RUNNING/).
  
 - Then, the Proto Learn page will be accessible via `http://localhost:8501`.

## Getting Started with Proto Learn

![Proto Learn Workflow](https://user-images.githubusercontent.com/49681382/90739663-62b38680-e2d7-11ea-83f0-3a9cf91e3374.png)

Above, you can see the main steps for workflow of Proto Learn at a glance. 

To get started with Proto Learn, [a special page](https://github.com/OmicEra/proto_learn/wiki/USING-Proto-Learn) is prepared for [`Using Proto Learn`](https://github.com/OmicEra/proto_learn/wiki/USING-Proto-Learn). 

On this page, you can click on the titles listed in *Table of Contents* for jumping into to the detailed documentation for each section explaning them and allowing you to give a try for the steps in the workflow using the example dataset. 

## Contributing
