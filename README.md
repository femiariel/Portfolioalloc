# Portfolio Optimization in Python with Gradio and PyPortfolioOpt  

## Description  
This program is designed to optimize a stock portfolio using the **PyPortfolioOpt** library. Users can input stock tickers to obtain an optimal allocation that minimizes risk or maximizes the Sharpe ratio. The user interface is built with **Gradio**, making the tool interactive and easy to use.  

## Key Features  
- **Market Data Retrieval**:  
  Historical stock price data is fetched using **pandas_datareader** from the *Stooq* data source.  
- **Portfolio Optimization**:  
  - Calculation of historical average returns.  
  - Estimation of the covariance matrix of returns.  
  - Maximization of the Sharpe ratio (or minimization of volatility if issues occur).  
- **Discrete Allocation**:  
  A discrete allocation of stocks is performed for a $10,000 portfolio, with calculations for expected returns and potential gains.  
- **Graphical Visualization**:  
  A chart is generated to show the initial portfolio value compared to the expected potential value after optimization.  

## Dependencies  
- `matplotlib`  
- `numpy`  
- `pandas`  
- `pandas_datareader`  
- `pypfopt`  
- `gradio`  

### Install Dependencies  
To install the required dependencies, run the following command:  
```bash
pip install -r requirements.txt
