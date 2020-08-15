import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

# Read training data, break into training/validation
housing = pd.read_csv("data/train.csv")
train_set, valid_set = train_test_split(housing, test_size=0.2, random_state=69420)

# Break datasets into data and labels
# Only using 2 attributes to start
train_data = train_set[["LotArea", "1stFlrSF"]].copy()
train_labels = train_set["SalePrice"].copy()
valid_data = valid_set[["LotArea", "1stFlrSF"]].copy()
valid_labels = valid_set["SalePrice"].copy()

# Train a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(train_data, train_labels)

# Test on validation set
predictions = lin_reg.predict(valid_data)

# Evaluate root-mean-squared-log-error
lin_msle = mean_squared_log_error(valid_labels, predictions)
lin_rmsle = np.sqrt(lin_msle)
print("RMSLE: ", lin_rmsle)
