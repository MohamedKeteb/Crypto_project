import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import requests 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import ta


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
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Candles', 'RSI'])
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



