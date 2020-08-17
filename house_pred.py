import csv
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

def fix_na_values(housing):
    h = housing.copy()
    h = h.fillna(0)
    h["Alley"] = h["Alley"].replace([0], "None")
    h["MasVnrType"] = h["MasVnrType"].replace([0], "None")
    h["BsmtExposure"] = h["BsmtExposure"].replace([0], "None")
    h["BsmtFinType1"] = h["BsmtFinType1"].replace([0], "None")
    h["BsmtFinType2"] = h["BsmtFinType2"].replace([0], "None")
    h["Electrical"] = h["Electrical"].replace([0], "None")
    h["GarageType"] = h["Electrical"].replace([0], "None")
    h["MiscFeature"] = h["MiscFeature"].replace([0], "None")
    return h

def quantify_ordinals(housing):
    h = housing.copy()
    h["ExterQual"] = h["ExterQual"].replace(["Ex"], 5)\
                                               .replace(["Gd"], 4)\
                                               .replace(["TA"], 3)\
                                               .replace(["Fa"], 2)\
                                               .replace(["Po"], 1)
    h["ExterCond"] = h["ExterCond"].replace(["Ex"], 5)\
                                               .replace(["Gd"], 4)\
                                               .replace(["TA"], 3)\
                                               .replace(["Fa"], 2)\
                                               .replace(["Po"], 1)
    h["BsmtQual"] = h["BsmtQual"].replace(["Ex"], 5)\
                                             .replace(["Gd"], 4)\
                                             .replace(["TA"], 3)\
                                             .replace(["Fa"], 2)\
                                             .replace(["Po"], 1)
    h["BsmtCond"] = h["BsmtCond"].replace(["Ex"], 5)\
                                 .replace(["Gd"], 4)\
                                 .replace(["TA"], 3)\
                                 .replace(["Fa"], 2)\
                                 .replace(["Po"], 1)
    h["HeatingQC"] = h["HeatingQC"].replace(["Ex"], 5)\
                                   .replace(["Gd"], 4)\
                                   .replace(["TA"], 3)\
                                   .replace(["Fa"], 2)\
                                   .replace(["Po"], 1)
    h["CentralAir"] = h["CentralAir"].replace(["N"], 0)\
                                     .replace(["Y"], 1)
    h["KitchenQual"] = h["KitchenQual"].replace(["Ex"], 5)\
                                       .replace(["Gd"], 4)\
                                       .replace(["TA"], 3)\
                                       .replace(["Fa"], 2)\
                                       .replace(["Po"], 1)
    h["FireplaceQu"] = h["FireplaceQu"].replace(["Ex"], 5)\
                                       .replace(["Gd"], 4)\
                                       .replace(["TA"], 3)\
                                       .replace(["Fa"], 2)\
                                       .replace(["Po"], 1)
    h["GarageFinish"] = h["GarageFinish"].replace(["Fin"], 3)\
                                         .replace(["RFn"], 2)\
                                         .replace(["Unf"], 1)
    h["GarageQual"] = h["GarageQual"].replace(["Ex"], 5)\
                                     .replace(["Gd"], 4)\
                                     .replace(["TA"], 3)\
                                     .replace(["Fa"], 2)\
                                     .replace(["Po"], 1)
    h["GarageCond"] = h["GarageCond"].replace(["Ex"], 5)\
                                     .replace(["Gd"], 4)\
                                     .replace(["TA"], 3)\
                                     .replace(["Fa"], 2)\
                                     .replace(["Po"], 1)
    h["PavedDrive"] = h["PavedDrive"].replace(["Y"], 3)\
                                     .replace(["P"], 2)\
                                     .replace(["N"], 1)
    h["PoolQC"] = h["PoolQC"].replace(["Ex"], 5)\
                             .replace(["Gd"], 4)\
                             .replace(["TA"], 3)\
                             .replace(["Fa"], 2)\
                             .replace(["Po"], 1)
    return h


# Read training data, deal with all "NA" values appropriately
housing = pd.read_csv("data/train.csv")
housing = fix_na_values(housing)

# Change ordinal categories to numbers
housing = quantify_ordinals(housing)

# Break into training/validation
train_set, valid_set = train_test_split(housing, test_size=0.2, random_state=69420)

# Numerical attributes and categorical attributes
num_attrs = ["LotFrontage",
             "LotArea",
             "MasVnrArea",
             "BsmtFinSF1",
             "BsmtFinSF2",
             "BsmtUnfSF",
             "TotalBsmtSF",
             "1stFlrSF",
             "2ndFlrSF",
             "LowQualFinSF",
             "GrLivArea",
             "BsmtFullBath",
             "BsmtHalfBath",
             "FullBath",
             "HalfBath",
             "BedroomAbvGr",
             "KitchenAbvGr",
             "TotRmsAbvGrd",
             "Fireplaces",
             "GarageCars",
             "GarageArea",
             "WoodDeckSF",
             "OpenPorchSF",
             "EnclosedPorch",
             "3SsnPorch",
             "ScreenPorch",
             "PoolArea",
             "MiscVal"]
ord_attrs = ["OverallQual",
             "OverallCond",
             "YearBuilt",
             "YearRemodAdd",
             "ExterQual",
             "ExterCond",
             "BsmtQual",
             "BsmtCond",
             "HeatingQC",
             "CentralAir",
             "KitchenQual",
             "FireplaceQu",
             "GarageYrBlt",
             "GarageFinish",
             "GarageQual",
             "GarageCond",
             "PavedDrive",
             "PoolQC",
             "YrSold"]
cat_attrs = ["Neighborhood",
             "MSSubClass",
             "MSZoning",
             "Street",
             "Alley",
             "LotShape",
             "LandContour",
             "BldgType",
             "HouseStyle",
             "RoofStyle",
             "RoofMatl",
             "Exterior1st",
             "Exterior2nd",
             "MasVnrType",
             "Foundation",
             "BsmtExposure",
             "BsmtFinType1",
             "BsmtFinType2",
             "Heating",
             "Electrical",
             "Functional",
             "GarageType",
             "MiscFeature",
             "MoSold",
             "SaleType",
             "SaleCondition"]
all_attrs = num_attrs + ord_attrs + cat_attrs

# Break datasets into data and labels
housing_data = housing[all_attrs].copy()
train_data = train_set[all_attrs].copy()
train_labels = train_set["SalePrice"].copy()
valid_data = valid_set[all_attrs].copy()
valid_labels = valid_set["SalePrice"].copy()

class CombineAttribute (BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        total_sqft = X[:,2] + X[:,6] + X[:,10] + X[:,20] + X[:,21] + X[:,22] + X[:,23] + X[:,24] + X[:,25] + X[:,26]
        X_cp = X.copy()
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 1, 1)
        X_cp = np.delete(X_cp, 10, 1)
        X_cp = np.delete(X_cp, 10, 1)
        X_cp = np.delete(X_cp, 10, 1)
        X_cp = np.delete(X_cp, 10, 1)
        X_cp = np.delete(X_cp, 10, 1)
        X_cp = np.delete(X_cp, 10, 1)
        X_cp = np.delete(X_cp, 10, 1)
        return np.c_[X_cp, total_sqft]

# Apply transformation pipelines
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("std_scaler", StandardScaler())])
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore"))])
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attrs + ord_attrs),
                                   ("cat", cat_pipeline, cat_attrs)])
full_pipeline.fit_transform(housing_data)
train_data_prepared = full_pipeline.transform(train_data)
valid_data_prepared = full_pipeline.transform(valid_data)

# Train a random forest
forest_reg = RandomForestRegressor()
forest_reg.fit(train_data_prepared, train_labels)

# Test on validation set
predictions = forest_reg.predict(valid_data_prepared)

# MSLE throws an error if there are negative values
for i in range(len(predictions)):
    if predictions[i] < 0:
        predictions[i] = 0

# Evaluate root-mean-squared-log-error
lin_msle = mean_squared_log_error(valid_labels, predictions)
lin_rmsle = np.sqrt(lin_msle)
print("RMSLE for validation data: ", lin_rmsle)

# Load and transform the test data
test_housing = pd.read_csv("data/test.csv")
test_housing = fix_na_values(test_housing)
test_housing = quantify_ordinals(test_housing)
test_data = test_housing[all_attrs].copy()
test_data_prepared = full_pipeline.transform(test_data)
predictions = forest_reg.predict(test_data_prepared)
with open("data/output.csv", "w", newline='') as output_file:
    csv_writer = csv.writer(output_file, delimiter=",")
    csv_writer.writerow(["Id", "SalePrice"])
    house_id = 1461
    for p in predictions:
        row = [house_id, float(p)]
        csv_writer.writerow(row)
        house_id += 1
