# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

# %% [markdown]
# # 1.) Pull in Data and Convert ot Monthly

# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from rich import inspect
from icecream import ic

# Elimitate warnings
import warnings
warnings.filterwarnings("ignore")
# %%
apple_data = yf.download('AAPL')
ic(apple_data.head());
# %%
# To revert the order of the data from oldest to newest.
df = apple_data.resample("M").last()[["Adj Close"]]
ic(df.head());
ic(df.index.min());
ic(df.index.max());
# %% [markdown]
# # 2.) Create columns. 
#   - Current Stock Price, Difference in stock price, Whether it went up or down over the next month,  option premium
# %%
df['delta'] = df["Adj Close"].diff()
df['delta'] = df['delta'].shift(-1)
df = df.dropna()
# sign of the delta
df['target'] = np.where(df['delta'] > 0, 1, -1)
# option premium
df['premium'] = df['Adj Close'] * 0.08
ic(df.head());
# %% [markdown]
# # 3.) Pull in X data, normalize and build a LogReg on column 2
# %%
X = pd.read_csv("Xdata.csv", index_col="Date", parse_dates=["Date"])
ic(X.tail());
ic(X.index.min());
ic(X.index.max());
ic(X.describe());
# %%
#y = df.loc[:"2023-09-30","Target"].copy()
min_date = max(X.index.min(), df.index.min())
max_date = min(X.index.max(), df.index.max())
y = df.target.loc[min_date:max_date].copy()
ic(y.tail());
ic(y.index.min());
ic(y.index.max());
ic(y.describe());
# %%
scaler = StandardScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), index=X.index, 
                      columns=X.columns)
X_norm = X_norm.loc[min_date:max_date]
ic(X_norm.head());
ic(X_norm.describe());
# %% [markdown]
# # 4.) Add columns, prediction and profits.
# %%
log_reg = LogisticRegression().fit(X_norm, y)
Y_pred = log_reg.predict(X_norm)
ic(Y_pred[:10]);
# %%
df = df.loc[min_date:max_date]
df['profit'] = 0
# True positive
df.loc[(df['target'] == 1) & (Y_pred == 1), 'profit'] = df['premium']
# False positive
df.loc[(df['target'] == -1) & (Y_pred == 1), 'profit'] = 100*df['delta'] + df['premium']
# %% [markdown]
# # 5.) Plot profits over time
# %%
df['profit'].cumsum().plot()
# This could occur because of the pandemic.
# %% [markdown]
# # 5.5) Your skills from the MQE to help Mr. Luis.
# %% [markdown]
"""
The speaker specializes in the world of blockchain and cryptocurrency. His initial project involved collaboration with a computer science professor, where they developed a more advanced version aimed at enhancing efficiency beyond Ethereum, named Avalanche.

He could contribute significantly to matters concerning network governance by investigating the impacts of various incentive types on network growth, along with other economic aspects of the network.

Given the growth of his platform "The Arena", the accumulation of data presents an opportunity for me to apply my knowledge in the field.
"""
# %% [markdown]
# # 6.) Create a loop that stores total profits over time
# %% [markdown]
# # 7.) What is the optimal threshold and plot the total profits for this model.
