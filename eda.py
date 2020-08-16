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


housing = pd.read_csv("data/train.csv")
attrs = ["Neighborhood",
         "SalePrice"]
housing = housing[attrs]
plot_prices_by_neighborhood(housing)
