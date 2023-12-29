# 00_Useful modules importation #


from statsmodels.graphics.tsaplots import pacf, acf
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.stattools import kpss 
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import pmdarima as pm
import json
import matplotlib.pyplot as plt 


# 01_This function allows to fast-check the stationnarity of the series by plotting the price we are interested in and its rolling mean/standard deviation on the same plot and then computing the ADF test.#



# 02- This function add the rolling mean/standard deviation to the table to check stationnarity later #

def add_arima_indicators(data,price, period) :

    ma = data[str(price)].rolling(window=period).mean().dropna()
    mstd = data[str(price)].rolling(window=period).std().dropna()
    data = pd.DataFrame(data.loc[period-1:])

    data['ma'] = ma
    data['mstd'] = mstd
    
    return data.reset_index().drop('index', axis=1)

# 03- To visualize the time series along with its moving statistics #

def arima_viz_with_indicator(data, symbol, interval) :

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['Candles', 'Moving Average/Std'])
    candle = go.Candlestick(x=data['datetime'],
                                        open=data['open'],
                                        high=data['high'],
                                        low=data['low'],
                                        close=data['close'])

    fig.add_trace(candle, row=1, col =1)


    ma_trace=go.Scatter(x=data['datetime'], y=data['ma'], mode='lines', name="Moving Average", yaxis='y2')

    fig.add_trace(rsi_trace, row=2, col=1)

    mstd_trace=go.Scatter(x=data['datetime'], y=data['mstd'], mode='lines', name="Moving Std", yaxis='y3')

    fig.add_trace(rsi_trace, row=3, col=1)

    
    fig.update_layout(title='Cotation of ' + str(symbol) + ' per ' + str(interval) ,
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)



    fig.show()


# 04- To run stationnarity test on differencing series 

def find_diff_order_2(data, price) :

    adf_statistic, adf_p_value = adf_test(data, price)

    kpss_p_value = kpss_test(data, price, to_print= False).loc["p-value"]

    if adf_p_value > 0.05 or kpss_p_value < 0.05 :
 
        d = 0

        while adf_p_value > 0.05 or kpss_p_value < 0.05 :
            print("")
            print("")
            print("P-values not good, trying differencing")
            # difference the time series
            data[str(price)] = data[str(price)].diff()
            # drop the null values
            data.dropna(inplace = True)
            # add 1 to d for each iteration to represent 1 differencing
            d += 1
            # perform adf test again to asses p value and exit loop if stationary
            adf_statistic, p_value = adf_test(data, price)
            # perform KPSS test
            kpss_p_value = kpss_test(data, price, to_print= False).loc["p-value"]
        print(f"Success... TS now stationary after {d} differencing")
  
        return d
