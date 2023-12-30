# Python_ENSAE_2A

Second-year Python project 2023-2024 for **the Python for Data Science course** by Elvis GABA-KPAYEDO, Mohamed Keteb and Aziz Bchir.

The objective of this project was to see how machine learning models behave when faced with a complicated prediction problem. 
So we ventured into time series prediction with our second-year knowledge, which is of course insufficient.
The time series we have chosen are the cryptocurrency stock prices

### The aim of the project : 

-  How can predictive models be used to estimate cryptocurrency closing prices ?

Our task was to integrate technical indicators and compare various approaches, 
such as linear regression, SVR, LSTM, and ARIMA, to obtain stock price predictions. 

### The plan of the project

- I. Data extraction using an API from Twelve Data, allows 800 requests per day
- II. Work on data and enrich data by adding variables relevant to financial series 
- III. Use of machine learning models, models used :
  - linear regression 
  - Support vector regression 
  - Deep learning model with Long Short-Term Memory 
  - Autoregressive integrated moving average (ARIMA)
- IV. Dashbord to makes predictions for day trader with instant data  


Required modules and packages to run all the files :

- in ```crypto_module.py``` :
    
    ```
    matplotlib
    numpy
    seaborn
    pandas
    requests
    ta
    plotly
    sklearn
    tensorflow
    statsmodels
    pmdarima
    datetime
    
    ```

- in ```app.py``` :
  
    ```
    streamlit
    time
    from PIL import Image

    ```

- To run the app use the command : $\texttt{streamlit run app.py}$
