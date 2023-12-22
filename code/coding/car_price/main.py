
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("car_dataset.csv")
# %%
df.head()
# %%
df.drop("Unnamed: 0", axis= 1, inplace=True)
# %%
df.rename(columns= {
    "Property":"V",
    "Power":"CC",
    "Price":"Ücret"
}, inplace=True)
# %%
df.isna().sum()
# %%
df["Ücret"].replace(","," ")
#%%
df.head()
# %%
df.to_csv("car_datasets_new.csv")
# %%
