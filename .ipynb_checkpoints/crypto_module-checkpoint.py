import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import requests 
import plotly.graph_objects as go
from ta.momentum import *



def load_data(symbol, start_data, end_data, interval, api_url):
    
    data = requests.get(api_url).json() # API request to have the data in a json datatyped
    data = pd.DataFrame(data['values'])
    data.dropna()
    
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors = 'coerce')
    
    return data





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

def finance_visualize(data, symbol, interva):
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


