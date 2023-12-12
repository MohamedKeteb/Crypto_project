# 01_This function allows to fast-check the stationnarity of the series by plotting the price we are interested in and its rolling mean/standard deviation on the same plot and then computing the ADF test.

def stationnarity_fast_check(data,price, period) : 
    rolling_mean = data['price'].rolling(window=period).mean()
    rolling_std = data['price'].rolling(window=period).std()
    
# Plot comparing time series and its moving statistics
    original = plt.plot(data['price'], color='blue', label='Close cotation')
    mean = plt.plot(rolling_mean, color='red', label='Rolling mean')
    std = plt.plot(rolling_std, color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling mean/standard deviation')
    plt.show(block=False)
        
# Dickey–Fuller's Test :
    result = adfuller(data['price'])
    print('ADF statistics : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Critical values :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))