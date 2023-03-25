# Predict Customer Churn

## Project Description
This project is part of the Udacity Machine Learning DevOps Engineer Nanodegree. The goal of this project is to write clean code to predict customer churn of a telecom company. The data set was originally provided by IBM. It contains information about customers who left the company within the last month. The data set includes information about:

* Customers who left within the last month – the column is called Churn

* Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies

* Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges

Find more information about the original data set on Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)


## Files and data description
Overview of the files and data present in the root directory. 


### project strcutre

```bash
├── Guide.ipynb          # Project guide
├── churn_notebook.ipynb # Contains the original code that was refactored
├── churn_library.py     # Contains data cleaning, EDA, and model training functions
├── churn_script_logging_and_tests.py # Tests the functions in churn_library.py
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Data file used for the project
│   └── bank_data.csv
├── images               # EDA & model Evaluation results
│   ├── eda
│   └── results
├── logs				 # Test logs
└── models               # Test & trained models
```

## Running Files

### Requirements
* Python 3.8
* Libraries: pandas, numpy, matplotlib, seaborn, sklearn, joblib
* All libraries are listed in the [requirements.txt](requirements.txt) file.

### Instructions
#### 1. Clone the repository and navigate to the downloaded folder.
```
git clone https://github.com/sagecodes/customer-churn
cd customer-churn
```
####  2. Create (and activate) a new environment, named `churn` with Python 3.8. 

Use conda, venv, or preferred environment manager.

```bash
conda create -n churn python=3.8
conda activate churn
```

#### 3. Install the required packages.
```
pip install -r requirements.txt
```

#### 4. Run the functions in churn_library.py to clean, explore, and train the data.
To run the churn library without tests you can either execute code blocks in the run.py file with the VS Code Python extension or run the following command in the terminal:

```bash
python run.py
```

This will run the functions in churn_library.py and save the results in the `logs`, `images`, and `models` folders. If running with the VS Code Python extension, some of the results will show up in the ineractive window.

#### 5. (optional) Run the churn_script_logging_and_tests.py file to test the functions in churn_library.py

```bash
python churn_script_logging_and_tests.py
```
This will run the functions in churn_library.py and save the results in the logs and models folders. Some of the results will be printed in the terminal.




--
A few other libraries used from experience to help make the code cleaner:
- [black](https://github.com/psf/black)
- [isort](https://pycqa.github.io/isort/)




