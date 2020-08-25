# House Price Prediction Kaggle Challenge
This is a machine learning project for a house price prediction challenge on Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/

## Data
The files in the `data` folder were downloaded straight from the Kaggle challenge. `train.csv` contains all the training samples, including the labels (the `SalePrice` column). `test.csv` contains the samples meant to be predicted, so there's no `SalePrice` column. The goal for the challenge is to generate a csv file containing the predicted `SalePrice` for the samples in `test.csv`.

## Required libraries
The script uses the following libraries (for Python 3):
### Standard libraries
- `csv`
- `sys`
### Third-party libraries
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`

## Running the prediction algorithm
To run a validation check, which splits the training data into 80% training and 20% validation and then evaluates and prints the RMSLE for the validation results, run the following:
`python3 house_pred.py validate`
Running the validation check does not use the test data.

To evaluate the test data and generate the csv file for submission to Kaggle, run the following:
`python3 house_pred.py test`
This will generate `output.csv` in the `data/` directory.

## EDA script
The script `eda.py` contains some functions for analyzing/interpreting the data. It can be run by commenting or uncommenting the function calls at the bottom of the script and running:
`python3 eda.py`
