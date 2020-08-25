"""
Exploratory Data Analysis.
Functions for analyzing and plotting the housing data.
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_prices_by_neighborhood(housing):
    nh_counts = dict()
    nh_price_totals = dict()
    nh_max_prices = dict()
    nh_min_prices = dict()
    nh_avg_prices = dict()

    # Count the number of houses, sum of prices, and max/min price for each neighborhood
    for idx, row in housing.iterrows():
        nh_name = row["Neighborhood"]
        sale_price = row["SalePrice"]
        if nh_name not in nh_counts:
            nh_counts[nh_name] = 1
            nh_price_totals[nh_name] = sale_price
            nh_min_prices[nh_name] = sale_price
            nh_max_prices[nh_name] = sale_price
        else:
            nh_counts[nh_name] += 1
            nh_price_totals[nh_name] += sale_price
            if sale_price < nh_min_prices[nh_name]:
                nh_min_prices[nh_name] = sale_price
            if sale_price > nh_max_prices[nh_name]:
                nh_max_prices[nh_name] = sale_price

    # Calculate avg house price for each neighborhood
    for nh_name in nh_counts:
        avg_price = float(nh_price_totals[nh_name]) / float(nh_counts[nh_name])
        nh_avg_prices[nh_name] = avg_price

    # Plot bar graphs
    x = range(len(nh_counts))
    fig, axs = plt.subplots(2)
    axs[0].bar(x, list(nh_max_prices.values()), align="center", color="r")
    axs[0].bar(x, list(nh_avg_prices.values()), align="center", color="g")
    axs[0].bar(x, list(nh_min_prices.values()), align="center", color="b")
    axs[1].bar(x, list(nh_counts.values()), align="center", color="b")
    axs[0].set_title("Min (blue), avg (green), max (red) house price by neighborhood")
    axs[1].set_title("Number of houses by neighborhood")
    plt.sca(axs[0])
    plt.xticks(x, list(nh_avg_prices.keys()), rotation="vertical")
    plt.sca(axs[1])
    plt.xticks(x, list(nh_avg_prices.keys()), rotation="vertical")
    fig.tight_layout()
    plt.show()


def plot_prices_by_month_sold(housing):
    nh_counts = dict()
    nh_price_totals = dict()
    nh_max_prices = dict()
    nh_min_prices = dict()
    nh_avg_prices = dict()

    # Count the number of houses, sum of prices, and max/min price for each neighborhood
    for idx, row in housing.iterrows():
        nh_name = row["MoSold"]
        sale_price = row["SalePrice"]
        if nh_name not in nh_counts:
            nh_counts[nh_name] = 1
            nh_price_totals[nh_name] = sale_price
            nh_min_prices[nh_name] = sale_price
            nh_max_prices[nh_name] = sale_price
        else:
            nh_counts[nh_name] += 1
            nh_price_totals[nh_name] += sale_price
            if sale_price < nh_min_prices[nh_name]:
                nh_min_prices[nh_name] = sale_price
            if sale_price > nh_max_prices[nh_name]:
                nh_max_prices[nh_name] = sale_price

    # Calculate avg house price for each neighborhood
    for nh_name in nh_counts:
        avg_price = float(nh_price_totals[nh_name]) / float(nh_counts[nh_name])
        nh_avg_prices[nh_name] = avg_price

    # Plot bar graphs
    x = range(len(nh_counts))
    fig, axs = plt.subplots(2)
    axs[0].bar(x, list(nh_max_prices.values()), align="center", color="r")
    axs[0].bar(x, list(nh_avg_prices.values()), align="center", color="g")
    axs[0].bar(x, list(nh_min_prices.values()), align="center", color="b")
    axs[1].bar(x, list(nh_counts.values()), align="center", color="b")
    axs[0].set_title("Min (blue), avg (green), max (red) house price by month sold")
    axs[1].set_title("Number of houses by month sold")
    plt.sca(axs[0])
    plt.xticks(x, list(nh_avg_prices.keys()), rotation="vertical")
    plt.sca(axs[1])
    plt.xticks(x, list(nh_avg_prices.keys()), rotation="vertical")
    fig.tight_layout()
    plt.show()


def correlation(housing):
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
                 "MiscVal",
                 "SalePrice"]
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
    housing = housing[num_attrs + ord_attrs]
    corr_matrix = housing.corr()
    print(corr_matrix["SalePrice"].sort_values(ascending=False))


def show_scatter_plot(housing):
    attrs = ["SalePrice", "BsmtHalfBath"]
    pd.plotting.scatter_matrix(housing[attrs])
    plt.show()


housing = pd.read_csv("data/train.csv")
#correlation(housing)
#plot_prices_by_neighborhood(housing)
plot_prices_by_month_sold(housing)
#show_scatter_plot(housing)
#print(housing["BsmtHalfBath"].value_counts())
