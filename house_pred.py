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


# Read training data, deal with all "NA" values appropriately
housing = pd.read_csv("data/train.csv")
housing = housing.fillna(0)
housing["Alley"] = housing["Alley"].replace([0], "None")
housing["MasVnrType"] = housing["MasVnrType"].replace([0], "None")
housing["BsmtExposure"] = housing["BsmtExposure"].replace([0], "None")
housing["BsmtFinType1"] = housing["BsmtFinType1"].replace([0], "None")
housing["BsmtFinType2"] = housing["BsmtFinType2"].replace([0], "None")
housing["Electrical"] = housing["Electrical"].replace([0], "None")
housing["GarageType"] = housing["Electrical"].replace([0], "None")
housing["MiscFeature"] = housing["MiscFeature"].replace([0], "None")

# Change ordinal categories to numbers
housing["ExterQual"] = housing["ExterQual"].replace(["Ex"], 5)\
                                           .replace(["Gd"], 4)\
                                           .replace(["TA"], 3)\
                                           .replace(["Fa"], 2)\
                                           .replace(["Po"], 1)
housing["ExterCond"] = housing["ExterCond"].replace(["Ex"], 5)\
                                           .replace(["Gd"], 4)\
                                           .replace(["TA"], 3)\
                                           .replace(["Fa"], 2)\
                                           .replace(["Po"], 1)
housing["BsmtQual"] = housing["BsmtQual"].replace(["Ex"], 5)\
                                         .replace(["Gd"], 4)\
                                         .replace(["TA"], 3)\
                                         .replace(["Fa"], 2)\
                                         .replace(["Po"], 1)
housing["BsmtCond"] = housing["BsmtCond"].replace(["Ex"], 5)\
                                         .replace(["Gd"], 4)\
                                         .replace(["TA"], 3)\
                                         .replace(["Fa"], 2)\
                                         .replace(["Po"], 1)
housing["HeatingQC"] = housing["HeatingQC"].replace(["Ex"], 5)\
                                           .replace(["Gd"], 4)\
                                           .replace(["TA"], 3)\
                                           .replace(["Fa"], 2)\
                                           .replace(["Po"], 1)
housing["CentralAir"] = housing["CentralAir"].replace(["N"], 0)\
                                             .replace(["Y"], 1)
housing["KitchenQual"] = housing["KitchenQual"].replace(["Ex"], 5)\
                                               .replace(["Gd"], 4)\
                                               .replace(["TA"], 3)\
                                               .replace(["Fa"], 2)\
                                               .replace(["Po"], 1)
housing["FireplaceQu"] = housing["FireplaceQu"].replace(["Ex"], 5)\
                                               .replace(["Gd"], 4)\
                                               .replace(["TA"], 3)\
                                               .replace(["Fa"], 2)\
                                               .replace(["Po"], 1)
housing["GarageFinish"] = housing["GarageFinish"].replace(["Fin"], 3)\
                                                 .replace(["RFn"], 2)\
                                                 .replace(["Unf"], 1)
housing["GarageQual"] = housing["GarageQual"].replace(["Ex"], 5)\
                                             .replace(["Gd"], 4)\
                                             .replace(["TA"], 3)\
                                             .replace(["Fa"], 2)\
                                             .replace(["Po"], 1)
housing["GarageCond"] = housing["GarageCond"].replace(["Ex"], 5)\
                                             .replace(["Gd"], 4)\
                                             .replace(["TA"], 3)\
                                             .replace(["Fa"], 2)\
                                             .replace(["Po"], 1)
housing["PavedDrive"] = housing["PavedDrive"].replace(["Y"], 3)\
                                             .replace(["P"], 2)\
                                             .replace(["N"], 1)
housing["PoolQC"] = housing["PoolQC"].replace(["Ex"], 5)\
                                     .replace(["Gd"], 4)\
                                     .replace(["TA"], 3)\
                                     .replace(["Fa"], 2)\
                                     .replace(["Po"], 1)

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

# TODO: Add class for combining square-footage attributes

# Apply transformation pipelines
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("std_scaler", StandardScaler())])
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder())])
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
print("RMSLE: ", lin_rmsle)

# TODO: Run on test data, generate output csv file
