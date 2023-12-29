#------------------------------------------------------------------------------------------------------------------
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
import statsmodels.api as sm
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels
import pmdarima as pm
import datetime

# This function loads Bitcoin data from an API.
def load_data(symbol, start_date, end_date, interval):
    '''
    Aim : Load the data from the API 

    Input :
    - symbol :
        type: string
        for instance 'BTC/USD'
    - start_date : 
        type : string 
        format : "%Y-%m-%d 00:00:00"
    - end_data :
        type : string
        format : "%Y-%m-%d 00:00:00"
    - interval : 
        trype : string 
        e.g '1day',  '1h'
    
    output : 
        data
        type : Pandas DataFrame
    

    '''


    api_key = 'de6ee984b4c24a0c9c7a43d7dca8b75e' # The Api key of Twelvedata.com (800 requests per day)
    order = 'asc'
    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&start_date={start_date}&end_date={end_date}&interval={interval}&order={order}&apikey={api_key}'
    
    data = requests.get(api_url).json() # API request to have the data in a json datatyped
    data = pd.DataFrame(data['values']) # request the key 'values'
    data.dropna() # drpo NaN values
    
    # change the type from object to numeric in order to plot the time series properly
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
    # Creating a candlestick chart using Plotly.
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
# This function adds technical indicators to the given bitcoin data. 
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
    # Create a copy of the data to avoid modifying the original DataFrame
    scaled_data = data.copy()

    # Find the global minimum and maximum values from the 'low' and 'high' columns, respectively
    global_min = data['low'].min()
    global_max = data['high'].max()

    # Define a function to scale each value
    def scale_value(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    # Scale the 'open' and 'close' prices with respect to the global min and max
    scaled_data['open'] = data['open'].apply(scale_value, args=(global_min, global_max))
    scaled_data['close'] = data['close'].apply(scale_value, args=(global_min, global_max))

    # we directly scale 'high' and 'low' with global min and max as well
    scaled_data['high'] = data['high'].apply(scale_value, args=(global_min, global_max))
    scaled_data['low'] = data['low'].apply(scale_value, args=(global_min, global_max))

    # Scale other columns individually 
    for column in data.columns:
        if column not in ['datetime', 'open', 'high', 'low', 'close']:
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Scale each column separately
            scaled_column = scaler.fit_transform(data[[column]])
            # Store the scaled values back in the DataFrame
            scaled_data[column] = scaled_column.flatten()

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




    
def inverse_scalling(x, data):
    
    y = pd.DataFrame(x)
    inv_scale  = lambda z : z * (data['high'].max() - data['low'].min()) + data['low'].min()
    x = np.array(y.apply(inv_scale))

    return x


#----------------------------- Functions for ARIMA model computations------------------------------------------------------------#

# 01- Add moving Average/Std to the table containing the cotation 

def add_arima_indicators(data, price, period, ln = False) : 


    """ Inputs : 

        price : str = the price we aim to predict (close, high...)

        period : the length of the rolling window used to compute rolling statistics from the price we defined

        ln : Boolean, to transform or not the value by applying log 
 
    """    
    
    if ln == True :

        for i in data.columns[1:] :

            data[str(i)] = np.log(data[str(i)])


    ma = data[str(price)].rolling(window=period).mean().dropna()
    mstd = data[str(price)].rolling(window=period).std().dropna()
    data = pd.DataFrame(data.loc[period-1:])

    data['ma'] = ma
    data['mstd'] = mstd
    
    return data.reset_index().drop('index', axis=1)


# 02- Visualize the cotations along with the moving average of our choice 

def arima_viz_with_indicator(data, symbol, interval) :

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['Candles', 'Moving Average', 'Moving Std'])
    candle = go.Candlestick(x=data['datetime'],
                                        open=data['open'],
                                        high=data['high'],
                                        low=data['low'],
                                        close=data['close'], 
                                        name = "Cotation")

    fig.add_trace(candle, row=1, col =1)


    ma_trace=go.Scatter(x=data['datetime'], y=data['ma'], mode='lines', name="Moving Average", yaxis='y2')

    fig.add_trace(ma_trace, row=2, col=1)

    mstd_trace=go.Scatter(x=data['datetime'], y=data['mstd'], mode='lines', name="Moving Std", yaxis='y3')

    fig.add_trace(mstd_trace, row=3, col=1)

    
    fig.update_layout(title='Cotation of ' + str(symbol) + ' per ' + str(interval) ,
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)



    fig.show()


# 03- To compute ADF and KPSS test for stationnarity

def adf_test(data, price, to_print = True) : 
    """
    Inputs : 
            to_print : Boolean allowing us wether to print the results of test for commentary purposes or just 
                       assigning them to variables for computationnal purposes 
    """
    result = adfuller(data[str(price)])
    adf_stat = result[0]
    p_value = result[1]

    if to_print == True : 

        print('ADF statistic : %f' % adf_stat)
        print('p-value : %f' % p_value)

    else :
        
        return adf_stat, p_value





def kpss_test(data, price, to_print = True) : 
    
    kpsstest = kpss(data[str(price)], regression="ct", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:2], index=["Test Statistic", "p-value"]
    )
    if to_print == True :
        
        print("Results of KPSS Test:")
        print(kpss_output)

    else :
        
        return kpss_output


# 04- To find the number of times we must differentiate the data to achieve stationnarity

def find_diff_order(data, price, to_print = True) :

    d = 0
    diff_data = data
    adf_statistic, p_value = adf_test(data, price, to_print= False)

    kpss_test(data, price, to_print= False)

    while p_value > 0.05 :

        # difference the time series
        diff_data[str(price)] = diff_data[str(price)].diff()
        # drop the null values
        diff_data.dropna(inplace = True)
            # add 1 to d for each iteration to represent 1 differencing
        d += 1
            # perform adf test again to asses p value and exit loop if stationary
        adf_statistic, p_value = adf_test(diff_data, price, to_print= False)
            # perform KPSS test
        

    if to_print == True :

        kpss_test(diff_data, price)
        print(f"Success... TS now stationary after {d} differencing")

    else :
        
        return d, diff_data

# 05- To plot simultaneaously the ACF and PACF to look for AR and MA parameters 

def acf_pacf(data, price):
 
    fig, ax = plt.subplots(1, figsize=(9,4), dpi=100)
    sm.graphics.tsa.plot_acf(data[str(price)], lags=20, ax = ax)
    plt.ylim([-0.05, 0.25])
    plt.yticks(np.arange(-0.20,1.2, 0.1))
    plt.show()
    
    fig, ax = plt.subplots(1, figsize=(9,4), dpi=100)
    sm.graphics.tsa.plot_pacf(data[str(price)], lags = 20, ax = ax)
    plt.ylim([-0.05, 0.25])
    plt.yticks(np.arange(-0.20,1.2, 0.1))
    plt.show()

# 06- To compare the distribution in the two half of our differenced data

def fast_dist_check(data, price):
    n = int(len(data[str(price)])/2)
    print(data[str(price)][:n].describe())
    print(data[str(price)][n:].describe())

    fig, ax = plt.subplots(1, figsize=(8,6), dpi=100)
    data[str(price)][:n].hist()
    data[str(price)][n:].hist()

# 07- To compute Ljung-box test on our differenced data

def LB_test(data, price) : 
    return sm.stats.acorr_ljungbox(data[str(price)], lags= [20], return_df= True)

# 08- Performing an auto-arima to the best values for parameters based on AIC criterion

def auto_arima(data, price):

    model = pm.auto_arima(data[str(price)],
                          start_p=3,
                          start_q=3,
                          test='adf',
                          max_p=15, 
                          max_q=15,
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True,
                          stepwise = True)
    # difference df by d found by auto arima
    differenced_by_auto_arima = data[str(price)].diff(model.order[1])
    return model.order, differenced_by_auto_arima, model.resid()

# 09- Forecasting futures values with ARIMA Model and plotting them alongside the real observed values 



def model(data, price,n, start_from, symbol, interval, p,d,q):
    """
    Inputs :
        start_from : The moment from which we should start to predict

        n : The number of future values we want to predict 
        
        p_d_q : Orders of the ARIMA model

    """
    # create date axis for predictions

    future =  [str(datetime.datetime.strptime(str(start_from), "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=x)) for x in range(n)]

    f = [str(datetime.datetime.strptime(str(start_from), "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=x)) for x in range(-15,n)]
    
    #Loading the real future values

    real = load_data(str(symbol),str(start_from), future[len(future)-1], str(interval))
    real["close"] = np.log(real["close"])

    time_series = np.log(data[str(price)])

    # fit

    model = statsmodels.tsa.arima.model.ARIMA(time_series, order = (p,d,q), 
                                                enforce_invertibility= False,
                                                enforce_stationarity= False)
    fitted = model.fit()
    fc = fitted.get_forecast(n) 

    #Set confidence to 95% 

    fc = (fc.summary_frame(alpha=0.05))

    #Get average ARIMA forecast

    fc_mean = fc['mean']

    #Get the extremity of confidence forecast interval

    fc_lower = fc['mean_ci_lower']
    fc_upper = fc['mean_ci_upper'] 

    #Plot last 15 price movements

    plt.figure(figsize=(12,8), dpi=100)
    plt.plot(data['datetime'][-15:],data[str(price)][-15:], label='BTC Price')

    #Plot mean forecast

    plt.plot(future, np.exp(fc_mean), label='Average ARIMA value', linewidth = 1.5, linestyle = 'dashdot') 

    #Create confidence interval

    plt.fill_between(future, np.exp(fc_lower),np.exp(fc_upper), color='b', alpha=.15, label = '95% Confidence')

    # Plotting the real future values

    plt.plot(future, real["close"], label = "Real values")
  

    plt.title(f"Bitcoin {n} Day Forecast")
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels(f)
    plt.xticks(rotation = 90)
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


# 10- Plot comparing time series and its moving statistics#

def stationnarity_fast_check(data, price, period) : 


    rolling_mean = data[str(price)].rolling(window=period).mean()
    rolling_std = data[str(price)].rolling(window=period).std()
    
    original = plt.plot(data[str(price)], color='blue', label= str(price) + 'cotation')
    mean = plt.plot(rolling_mean, color='red', label='Rolling mean')
    std = plt.plot(rolling_std, color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling mean/standard deviation')
    plt.show(block=False)


