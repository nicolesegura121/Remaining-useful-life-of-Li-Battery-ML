# Lithium ion battery state of health prediction :battery:

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
```
battery-prediction/
│
├── data/                    # Data directory
│   └── README.md            # Instructions for downloading data
│
├── notebooks/               # Jupyter notebooks
│   ├── EDA.ipynb   # Data exploration feature selection
│   └── Models.ipynb      # Model: LSTM, XGboost, ADA boosting and SVR. Training and prediction results
│
├── data_info/                  # Description of the experiments present in the dataset.
│
├── utils/                   # Utility functions
│
├── environment.yml         # Project dependencies
├── README.md                # Project documentation
└── LICENSE                  # License file
```

# Results 
Feature engineering was the most relevant aspect throughout this project. Making careful decisions about the type of features included in the models was critical to achieve good performance. Future work will be performed in improving the model architecture, specially for LSTM. 
