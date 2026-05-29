# Lithium ion battery state of health prediction:battery:

This project implements machine learning and deep learning models to predict the state of health of a Li-ion battery based on NASA's battery cycling data.

---- 
# Project overview: 
The project contains a thorough description of the process of extracting the relevant information about battery health descriptors. The careful selection of parameters were used to train two machine learning models: XGboost (XGB) and Support vector regression (SVR) and one neural network, which was based on long-short-term-memory (LSTM) layers. This is a great excercise to understand how batery management systems can benefit from data-driven approaches to ensure optimal feedback on the state of the battery. 

---

# Data Collection: 
To obtain the data, please visit the following sites: 
    - Kaggle: 
        - NASA- Battery Dataset: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset

# Repository structure: 
battery-prediction/
│
├── data/                    # Data directory
│   └── README.md            # Instructions for downloading data
│
├── notebooks/               # Jupyter notebooks
│   ├── battery_data_analysis.ipynb   # Data exploration and feature engineering
│   └── battery_prediction.ipynb      # Model training and evaluation
│
├── models/                  # Saved model files
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── data_loader.py       # Functions to load and preprocess data
│   └── plotting.py          # Visualization functions
│
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── LICENSE                  # License file

* Goal: Data-driven prediction of battery failures for safety managment.

* Steps:
    - Exploration of data and related literature
    - Selection of features and targe
    - Exploration of benchmarked models
    - Use of LSTM for prediction.



(could it also be done for fuel cells??--> future outlook)
