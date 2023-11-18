import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import requests 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import ta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_data(symbol, start_date, end_date, interval):

    api_key = 'de6ee984b4c24a0c9c7a43d7dca8b75e' # The Api key of Twelvedata.com (800 requests per day)
    order = 'asc'
    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&start_date={start_date}&end_date={end_date}&interval={interval}&order={order}&apikey={api_key}'
    
    data = requests.get(api_url).json() # API request to have the data in a json datatyped
    data = pd.DataFrame(data['values'])
    data.dropna()
    
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors = 'coerce')
    
    return data

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def visualize_data(data, symbol, interval):
    '''
    Aim : Visualize the close cotation of  the considered Cryptocurrencie on the considered interval
    
    Input : 
    - data :
        type: pandas DataFrame 
                
    - symbol :
        type: string
        for instance 'BTC/USD'
        
    - interval : 
        type: string
        for instance '1day' or '5min'
        
    Output : Plot

    '''
                   
    sns.set(style = 'darkgrid')
    plt.title('Close cotation of '+ str(symbol))
    plt.xlabel(str(interval))
    plt.ylabel('USD')
    sns.lineplot(x = data.index, y = data['close'], color = 'green')
       
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def finance_visualize(data, symbol, interval):
    ''' 
    Aim : Visualize the candle of the market of the considered Cryptocurrencie
    on the considered interval
    
    Input : 
    - data :
        type: pandas DataFrame 
                
    - symbol :
        type: string
        for instance 'BTC/USD'
        
    - interval : 
        type: string
        for instance '1day' or '5min'
        
    Output : Plot

    '''
    
    fig = go.Figure(data=[go.Candlestick(x=data['datetime'],
                                     open=data['open'],
                                     high=data['high'],
                                     low=data['low'],
                                     close=data['close'])])
    
    fig.update_layout(title='Cotation of ' + str(symbol) + ' per ' + str(interval) ,
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)



    fig.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def visualize_with_indicator(data, symbol, interval, indicator):
    ''' 
    Aim : Visualize the candle of the market of the considered Cryptocurrencie
    on the considered interval
    
    Input : 
    - data :
        type: pandas DataFrame 
                
    - symbol :
        type: string
        for instance 'BTC/USD'
        
    - interval : 
        type: string
        for instance '1day' or '5min'
    - indicator :
        type: string
        RSI EMA ATR
        
    Output : Plot

    '''
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Candles', str(indicator)])
    candle = go.Candlestick(x=data['datetime'],
                                        open=data['open'],
                                        high=data['high'],
                                        low=data['low'],
                                        close=data['close'])

    fig.add_trace(candle, row=1, col =1)


    rsi_trace=go.Scatter(x=data['datetime'], y=data[str(indicator)], mode='lines', name=str(indicator), yaxis='y2')

    fig.add_trace(rsi_trace, row=2, col=1)

    
    fig.update_layout(title='Cotation of ' + str(symbol) + ' per ' + str(interval) ,
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)



    fig.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def add_indicators(data, period=14):

    ema = ta.trend.ema_indicator(close = data['close'], window = period).dropna()
    rsi = ta.momentum.rsi(close=data['close'], window=period).dropna()
    atr = ta.volatility.AverageTrueRange(close=data['close'],high=data['high'], low=data['low'], window=period).average_true_range()
    atr = atr[atr>0]

    data = pd.DataFrame(data.loc[period-1:])

    data['RSI'] = rsi
    data['EMA'] = ema
    data['ATR'] = atr


    return data.reset_index().drop('index', axis=1)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def scaling_data(data):

    m = data.drop('datetime', axis =1)
    scaler =MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(m), columns= data.columns[1:])
    scaled_data.insert(0, 'datetime', data['datetime'])
    
    return scaled_data


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def data_preprocess(scaled_data,regressor, prediction_time):
    '''
    Aim : Shift the data to do regression on time series 
    for example, if you want to predict the next 30 days by changing the data, 
    each line of price is associated with the value taken 30 days later.

    Input:
    - scaled_data   type : DataFrame
    - prediction_time   type : int

    Output:
    - price Type : Numpy Array 
    - target Type : Numpy Array 

    '''
    target = scaled_data['close'].shift(-prediction_time).dropna()
    target = np.array(target).reshape(-1, 1)

    price = np.array(scaled_data[regressor])[:-prediction_time]
    
    return price, target 

def apply_linear_regression(scaled_data, prediction_time, price, target, regressor):

    price_train, price_test, target_train, target_test = train_test_split(price, target, test_size = 0.7)
    lr = LinearRegression().fit(price_train, target_train)

    price_to_predict = price[-prediction_time:] 
    lr_prediction = lr.predict(price_to_predict)


    prediction_matrix = pd.DataFrame(scaled_data['close'].tail(prediction_time))
    prediction_matrix['prediction'] = lr_prediction

    price_to_future = np.array(scaled_data[regressor])[-prediction_time:].reshape(-1, 1)
    future = lr.predict(price_to_future)

    target_predict = lr.predict(price_test)
    r2 = r2_score(target_test, target_predict)

    return prediction_matrix, future, r2 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def visualize_linear_reg(prediction_matrix,scaled_data, zoom = None):

    plt.xlabel('Days')
    plt.ylabel('BTC/USD ($)(scaled data)')
    plt.plot(scaled_data['close'])
    plt.plot(prediction_matrix[['close', 'prediction']])
    plt.legend(['Real Price', 'Real, price', 'Prediction'])
    if zoom is not None : 
        plt.xlim(zoom[0], zoom[1])
    plt.title('Prediction of close price of BTC/USD for the Last Month by Linear Regression')
    plt.show


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def apply_linear_regression(scaled_data, prediction_time, price, target, regressor):

    price_train, price_test, target_train, target_test = train_test_split(price, target, test_size = 0.7)
    lr = LinearRegression().fit(price_train, target_train)

    price_to_predict = price[-prediction_time:] 
    lr_prediction = lr.predict(price_to_predict)


    prediction_matrix = pd.DataFrame(scaled_data['close'].tail(prediction_time))
    prediction_matrix['prediction'] = lr_prediction

    price_to_future = np.array(scaled_data[regressor])[-prediction_time:]
    future = lr.predict(price_to_future)

    target_predict = lr.predict(price_test)
    r2 = r2_score(target_test, target_predict)

    return prediction_matrix, future, r2 


#------------------------------------------------------------------------------------------------------------------------------

def visualize_future(scaled_data, future, zoom = None):

    plt.xlabel('Days')
    plt.ylabel('BTC/USD ($)(scaled data)')
    arr1 = np.array(scaled_data['close']).reshape(-1, 1)
    arr2 = np.array(future).reshape(-1, 1) 
    ct = np.concatenate((arr1, arr2))
    plt.axvline(x = arr1.shape[0], color = 'r', linestyle = '--', label = 'Prediction')
    plt.plot(ct)
    if zoom is not None : 
        plt.xlim(zoom[0], zoom[1])
    plt.title('Prediction of close price of BTC/USD')

    plt.show




