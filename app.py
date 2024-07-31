import pandas as pd
import numpy as np
import pandas_datareader as pdr
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import load_model

from alpha_vantage.timeseries import TimeSeries

api_key = ' IJBWZR4DG0810QQR'  

symbol = 'TSLA'  # Updated to 'TSLA' for testing

# Initialize the Alpha Vantage TimeSeries object
ts = TimeSeries(key=api_key, output_format='pandas')

def get_stock_info(symbol):
    try:
        # Get stock overview data
        data, meta_data = ts.get_quote_endpoint(symbol=symbol)

        # Check if data and meta_data are not None
        if data is not None and meta_data is not None:
            # Display relevant information
            st.subheader(f"Stock Information for Ticker: {symbol}")
            st.write(f"Company Name: {meta_data.get('2. Name', 'Not Available')}")
            st.write(f"Last Refreshed: {meta_data.get('3. Last Refreshed', 'Not Available')}")
            st.write(f"Price: {data.get('05. price', 'Not Available')}")
            st.write(f"Open: {data.get('02. open', 'Not Available')}")
            st.write(f"High: {data.get('03. high', 'Not Available')}")
            st.write(f"Low: {data.get('04. low', 'Not Available')}")
            st.write(f"Volume: {data.get('06. volume', 'Not Available')}")

            if 'Note' in meta_data:
                st.warning(f"API Note: {meta_data['Note']}")

        else:
            st.error(f"No data retrieved for the specified symbol: {symbol}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

st.title("Stock Trend Prediction")

# User entered information 
ticker = st.text_input("Enter Stock Ticker:", 'TSLA')  # Updated to 'TSLA' for testing
if st.button("Get Stock Information"):
    get_stock_info(ticker)


st.subheader("Enter start information: ")
sty=st.number_input("Enter start year: ",2000)
stm=st.number_input("Enter start month: ",1)
stz=st.number_input("Enter start date: :",1)


st.subheader("Enter End information: ")
eny=st.number_input("Enter end year: ",2020)
enm=st.number_input("Enter end month: ",12)
end=st.number_input("Enter end date: :",31)

start=datetime(sty,stm,stz)
endd=datetime(eny,enm,end)

#Stock data from the above information
st.subheader('Data from {} to {}'.format(sty,eny))
df= yf.download(ticker, start , endd)

st.write(df.describe())

#Closing Price Graph
st.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#moving average 100 graph
st.subheader("Closing Price vs Time Chart with Moving Average 100")
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)

#Moving average 200 graph
st.subheader("Closing Price vs Time Chart with Moving Average 200")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig)


#Training data
datatrain=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
datatesting=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#Scaling the data between 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

#fit transforming the above scaled data using scaler fit transform
newdatatrain=scaler.fit_transform(datatrain)
x_train=[]
y_train=[]


#appending into x_test and y_test
for i in range(100,datatrain.shape[0]):
    x_train.append(newdatatrain[i-100: i])
    y_train.append(newdatatrain[i,0])
x_train , y_train = np.array(x_train),np.array(y_train)


#Loading LSTM model
model=load_model('datamod.h5')

#getting the previous 100 days of training data over testing
p100d=datatrain.tail(100)


finaldata = pd.concat([p100d, datatesting], ignore_index=True)

#Performing fit transform over the appended data
inpd=scaler.fit_transform(finaldata)
x_test=[]
y_test=[]

for i in range(100,inpd.shape[0]):
    x_test.append(inpd[i-100: i])
    y_test.append(inpd[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
y_pred=model.predict(x_test)


#Muliplying with the scale factor to generate the correct values
scalefactor=1/0.02099517
y_pred=y_pred*scalefactor
y_test=y_test*scalefactor

#Final Predicted graph
st.subheader("Original Price Versus Predicted Price")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_pred,'r',label='predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig2)