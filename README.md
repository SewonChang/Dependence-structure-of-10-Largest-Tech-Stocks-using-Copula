# Using Gaussian Copula to structure dependence between Tech Stocks

## Description

This project attempts to assume a more accurate function of dependence between the returns of the top 10 tech stocks (in market capitilization).
The Gaussian copula is used to model these correlations between the stocks, which is in turn used to generate synthetic data regarding the 10 stock prices.
Tests are performed to examine validity of the structure dependence and synthetic data, such as Kolmogorov-Smirnov test, correlation matrix comparisons, Wasserstein Distance.

After validating the synthetic data, Monte-Carlo simulations are performed to determine the Value-at-Risk and Conditional-Value-at-Risk.

## Installations

Some key libraries used are the following:
- "numpy" to compute log returns of stocks and set random seeds to reproduce random results
- "pandas" for creation of dataframes
- "yfinance" allowed to retrieve historical data of stock prices
- "scipy" allowed to perform tests such as KS tests and the fitting of distributions
- "copulas.multivariate" which allows to fit the Gaussian Copula to the multivariate dataframe
