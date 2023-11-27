import streamlit as st 
import time
from PIL import Image
import crypto_module
import datetime

time_now = datetime.datetime.now()
format_date = "%Y-%m-%d %H:00:00"
date = time_now.strftime(format_date)
date_  = str(int(date[:4])-1) + date[4:]

start_date = date_
end_date = date



image = Image.open('app_image.jpeg')
st.image(image, width = 700)
st.info('Use of Long short-terme memory deep learning model to predict the close price of a cryptocurrency', icon="ℹ️")
st.warning('Warning : the given prediction is by no means a reliable investment !')

crypto_selectbox = st.selectbox('Choose your Cryptocurrency', 
                             ('BTC/USD', 'ETH/USD', 'BNB/USD'))


t = None
select_time = st.radio('Choose the Data interval', ['Hourly', 'Daily'])
if select_time == 'Hourly':
    slider_hour = st.slider('Choose the time in hour to predict', 1, 10, 1)
    t = slider_hour

else:
    select_days = st.slider('Choose the time in days to predict', 1, 10, 1)
    t = select_days

button_prediction = st.button('Submit data')

interval = None
data = None
scaled_data = None
if button_prediction:
    with st.spinner('Prediction in progress ...'):
        if crypto_selectbox == 'Hourly':
            interval = '1h'
        else:
            interval = '1day'
        data = crypto_module.load_data(crypto_selectbox, start_date, end_date, interval)
        scaled_data = crypto_module.scaling_data(data)
        X, y = crypto_module.create_sequences(scaled_data, sequence_length = 10) # Prediction based on 10 periods (10 days or 10 hours) 
        prediction = crypto_module.recursive_prediction(X, y, t)

    st.success('Done', icon="✅")

    st.line_chart(prediction[-t:])









