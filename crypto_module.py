import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import requests 
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error



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
def close_rolling_mean(d, values) :
    
    return(values['close'].rolling(window=d).mean())

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

    Output:w
    - price Type : Numpy Array 
    - target Type : Numpy Array 

    '''
    target = scaled_data['close'].shift(-prediction_time).dropna()
    target = np.array(target).reshape(-1, 1)

    price = np.array(scaled_data[regressor])[:-prediction_time]
    
    return price, target 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def visualize_model(prediction_matrix,scaled_data, zoom = None):

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

    price_train, price_test, target_train, target_test = train_test_split(price, target, test_size = 0.3)
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


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def apply_svr(scaled_data, prediction_time, price, target, regressor, best_C, best_gamma):

    price_train, price_test, target_train, target_test = train_test_split(price, target, test_size = 0.3)
    svr_rbf = SVR(kernel = 'rbf', C = best_C, gamma= best_gamma)
    svr_rbf.fit(price_train, np.ravel(target_train))

    price_to_predict = price[-prediction_time:] 
    svr_prediction = svr_rbf.predict(price_to_predict)


    prediction_matrix = pd.DataFrame(scaled_data['close'].tail(prediction_time))
    prediction_matrix['prediction'] = svr_prediction

    price_to_future = np.array(scaled_data[regressor])[-prediction_time:]
    future = svr_rbf.predict(price_to_future)

    svr_accuracy = svr_rbf.score(price_test, target_test)

    return prediction_matrix, future, svr_accuracy 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def cross_validation_parameter(param_grid, price_train, target_train):

    svr_rbf = SVR(kernel = 'rbf') # We use the Support Vector Regression
    search = GridSearchCV(svr_rbf, param_grid, cv=3, scoring = 'neg_mean_squared_error', n_jobs=-1)
    search.fit(price_train, np.ravel(target_train)) # we fit the cross validation on the data
    
    best_C = search.best_params_['C']
    best_gamma = search.best_params_['gamma']

    return best_C, best_gamma




#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




# Function to create sequences for training the model
def create_sequences(scaled_data, sequence_length):
    xs, ys = [], []
    # Extract the column of data we want to predict (The close price)
    data=scaled_data.iloc[:, 4]
    # Iterate through the data to create sequences
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)] # Input sequence
        y = data[i + sequence_length] # Target value
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def visualize_RNN_prediction(y_train, y_test,predicted_values):
    # Combining y_train and y_test
    full_y = np.concatenate([y_train, y_test])

    # Creating a time axis for the full dataset
    time_steps = np.arange(len(full_y))

    # Determine the starting point for y_test in the combined array
    test_start = len(y_train)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot y_train part
    plt.plot(time_steps[:test_start], full_y[:test_start], label='Real price', color='blue')

    # Plot y_test part
    plt.plot(time_steps[test_start:], full_y[test_start:], label='Real price', color='orange')

    # Plot predicted_values on top of y_test
    plt.plot(time_steps[test_start:], predicted_values, label='Predicted Values', color='green', linestyle='--')

    plt.title('Full Data with Real price and Predicted price')
    plt.xlabel('time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def lstm_model(X, y):
    # We split the data into training and testing sets
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)), # LSTM layer with 50 units and return sequences
    Dropout(0.2), # Dropout layer to prevent overfitting
    LSTM(50, return_sequences=False), 
    Dropout(0.2),
    Dense(25), 
    Dense(1) 
    ])

    model.compile(optimizer='adam', loss='mean_squared_error') # Use Adam optimizer and mean squared error loss to optimize the prediction
    model.fit(X_train, y_train, batch_size=351, epochs=100) # Train for 200 epochs (= How many times the entire dataset is used for training) with a batch size (=How many data samples are processed at a time during an epoch) of 351
    predicted_values= model.predict(X_test)

    return y_train, y_test, predicted_values

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def recursive_prediction(X,y, t):
        # We split the data into training and testing sets
    
    model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)), # LSTM layer with 50 units and return sequences
    Dropout(0.2), # Dropout layer to prevent overfitting
    LSTM(50, return_sequences=False), 
    Dropout(0.2),
    Dense(25), 
    Dense(1) 
    ])

    model.compile(optimizer='adam', loss='mean_squared_error') # Use Adam optimizer and mean squared error loss to optimize the prediction
    model.fit(X, y, batch_size=351, epochs=100) # Train for 200 epochs (= How many times the entire dataset is used for training) with a batch size (=How many data samples are processed at a time during an epoch) of 351
    
    prediction = y[-X.shape[1]:].tolist()

    while len(prediction) - X.shape[1] < t:
        l = np.array([prediction[-X.shape[1]:]])
        p = model.predict(l)
        prediction.append(p[0][0])
    return prediction




    
