import streamlit as st 
import time
from PIL import Image

image = Image.open('app_image.jpeg')
st.image(image, width = 700)
st.text_area('','Use of Long short-terme memory deep learning model to predict the close price of a cryptocurrency')
st.warning('Warning : the given prediction is by no means a reliable investment !')

crypto_selectbox = st.selectbox('Choose your Cryptocurrency', 
                             ('BTC', 'ETH', 'BNB'))


select_time = st.radio('Choose the Data interval', ['Hourly', 'Daily'])
if select_time == 'Hourly':
    slider_hour = st.slider('Choose the time in hour to predict', 1, 10, 1)

else:
    select_days = st.slider('Choose the time in days to predict', 1, 10, 1)

button_prediction = st.button('Submit freatures')

if button_prediction:
    with st.spinner('Prediction in progress ....'):
        time.sleep(5)
    st.success('Done!')

    


