import pandas as pd
from sklearn.model_selection import train_test_split

housing = pd.read_csv("data/train.csv")
train_set, valid_set = train_test_split(housing, test_size=0.2, random_state=69420)
print(train_set.head())
print(valid_set.head())
