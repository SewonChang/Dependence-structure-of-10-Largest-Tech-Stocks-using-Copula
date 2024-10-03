#!/usr/bin/env python
# coding: utf-8

# # A common way to measure dependence between variables is to calculate the correlation, through something like pearson
# # Correlation is actually directly related to linear regressions, in other words, correlation gives us a linear understanding 
# # of the dependence between variables
# # An alternative way to measure dependence that may be more flexible and accurate is through copula
# 
# # Sklar's Theorem: a way to rewrite the joint distribution by the function of marginal distributions of multivariables

# In[2]:


import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

import copulas.multivariate as cm
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# collect historical data for target tech stocks

tickers = ['AAPL', 'MSFT','NVDA','GOOG','AMZN','META','TSLA','BABA','CRM', 'AMD']

tickers


# In[4]:


# collect adj close price from jan 1st 2022 to jan 1st 2024

basket = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start = "2022-01-01", end = "2024-01-01")
    basket[ticker] = data['Adj Close']

basket.head()


# In[5]:


## compute daily lognormal returns

log_Returns = np.log(basket/basket.shift(1))

## drop missing values

log_Returns = log_Returns.dropna()

log_Returns.head()


# # plot log returns to get a visual idea of their distributions
# 
# num_assets = len(log_Returns.columns)
# 
# plt.figure(figsize=(12, 8))
# fig, axs = plt.subplots(2,5,figsize=(15, 5), sharex=True, sharey=True)
# axs = axs.flatten()
# 
# # Plot histogram for each column
# for i, column in enumerate(log_Returns.columns):
#     axs[i].hist(log_Returns[column], bins=50, density=True, alpha=0.6, color='blue')
#     axs[i].set_title(f'{column} log-returns')
#     axs[i].set_ylabel('Density')

# In[7]:


# Fit log-returns of stocks to student-t distributions and obtain their parameters

pars = {}

for stock in log_Returns:
    loc, scale, dof = stats.t.fit(log_Returns[stock])
    pars[stock] = (loc, scale, dof)

pars


# In[8]:


# Use Kolmogorov-Smirnov Test, test student-t distribution

KS_test = {}

for ticker, (mu, sigma, v) in pars.items():
    statistic, p_value = stats.kstest(log_Returns[ticker], 't', args=pars[ticker])
    KS_test[ticker] = {'Statistic': statistic, 'P-value': p_value}

KS_test

# Notice that the p-values tend to be high, which indicates that the returns strongly follow a student-t distribution with the respective parameters


# In[9]:


# In order to construct the dependence structure of the tech stocks, we require marginal UNIFORM distributions.
# Thence, transform to uniform data

def transform_to_uniform(log_returns, distribution, params):
    cdf_values = stats.t.cdf(log_returns, *params)
    return cdf_values

uniform_data = {}
for ticker, (loc, scale, v) in pars.items():
    uniform_data[ticker] = transform_to_uniform(log_Returns[ticker], stats.t, (loc, scale, v))

uniform_df = pd.DataFrame(uniform_data)

uniform_df.head()


# In[10]:


# Fit the transformed data to gaussian copula
# Generate synthetic data from dependence structure of copula, with same length as original data

copulaG = cm.GaussianMultivariate()
copulaG.fit(uniform_df)

np.random.seed(123)
synthetic_data = copulaG.sample(len(uniform_df))

# Plot histograms of the original and synthetic data to compare

fig, axs = plt.subplots(2,5, figsize=(18, 6))
fig.subplots_adjust(hspace=0.4)


for ax, stock in zip(axs.flatten(), uniform_df.columns):
    ax.hist(uniform_df[stock], bins=30, alpha=0.5, label='Original')
    ax.hist(synthetic_data[stock], bins=30, alpha=0.5, label='Synthetic')
    ax.set_title(stock)
    
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.58,0.05), labelspacing=0.5, ncol=2)

plt.show()


# In[27]:


# The Wasserstein distance is a tool that essentially quantifies how much work is required to transform one distribution to another
# Testing validity of synthetic data against original using this method:

from scipy.stats import wasserstein_distance

wass = {}

for stock in tickers:
    wass[stock] = wasserstein_distance(uniform_df[stock], synthetic_data[stock])
    
table1 = {
  "Wasserstein Distance": wass,
}

pd.DataFrame(table1)

# The listed numbers are the "distances" from one distribution to another, the smaller the number indiactes a closer distribution


# In[12]:


# Further testing validity by analyzing correlation matrices

fig, axes = plt.subplots(ncols=2, figsize=(20, 4))

plt.subplot(1,2,1)
sns.heatmap(uniform_df.corr(), cmap="coolwarm", annot=True, vmin=-1)
plt.title('Original Correlation between Tech Stocks')

plt.subplot(1,2,2)
sns.heatmap(synthetic_data.corr(), cmap="coolwarm", annot=True, vmin=-1)
plt.title('Synthetic Correlation between Tech Stocks from Gaussian Copula')


# # Determine VaR and CVar from many samples of synthesized data
# 
# samples = 100000
# 
# np.random.seed(123)
# synsam = copulaG.sample(samples)
# synthetic_df = pd.DataFrame(synsam)
# 
# conf = 0.95
# VaR = synthetic_df.quantile(1 - conf)
# CVaR = synthetic_df[synthetic_df <= VaR].mean()
# 
# table2 = {
#   "VaR at 95%": VaR,
#   "CVaR at 95%": CVaR
# }
# 
# pd.DataFrame(table2)
