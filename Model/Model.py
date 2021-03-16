import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from DataFit import fitdata

#for deep learning model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

class Model():
    sc = []
    x_train = []
    y_train = []
    x_test = []
    open_price = []
    train_set = []
    test_set = []
    rain_set_scaled = []
    reg = []
    pred = []
    ticker = []
    
    def __init__(self):
        self.sc = MinMaxScaler()
        self.reg = Sequential()
        
    def CollectData(self,ticker,sentiment=None):
        self.ticker = ticker
        df = pd.read_csv(ticker+'.csv')
        df.shape
        df = df[::-1]
        df = df.reset_index(drop=True)
        df.head()
        df = df.sort_values(by = 'Date')
        df['Open'] = df['Open'].str.replace(',', '').str.replace('$', '').astype(float)
        df['Close'] = df['Close'].str.replace(',', '').str.replace('$', '').astype(float)
        if(sentiment !=None):
            df.loc[len(df)] = [float("nan"),float("nan"),sentiment,float("nan"),float("nan"),float("nan")]
            
        self.open_price = df.iloc[:,1:3]
        self.train_set = self.open_price[:223].values
        self.test_set = self.open_price[223:].values
        dates = pd.to_datetime(df['Date'])
        
        return self
        
    def SetTRaining(self):
        self.train_set_scaled = self.sc.fit_transform(self.train_set)
        for i in range(1,10):
            self.x_train.append(self.train_set_scaled[i-1:i,0])
            self.y_train.append(self.train_set_scaled[i,0])

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_train = np.reshape(self.x_train,(self.x_train.shape[0],self.x_train.shape[1],1))
        self.x_train.shape
        
        return self
        
    def InitialiseRNN(self):
        self.reg.add(LSTM(units = 100,return_sequences=True,input_shape=(self.x_train.shape[1],1)))
        self.reg.add(Dropout(0.2))
        self.reg.add(LSTM(units = 50,return_sequences=True))
        self.reg.add(Dropout(0.2))
        self.reg.add(LSTM(units = 50,return_sequences=True))
        self.reg.add(Dropout(0.2))
        self.reg.add(LSTM(units=50))
        self.reg.add(Dropout(0.2))
        self.reg.add(Dense(units=2))
        self.reg.compile(optimizer = 'adam',loss='mean_squared_error')
        self.reg.fit(self.x_train,self.y_train, epochs=50, batch_size =1,verbose=0)
        
        return self
        
    def StartTraining(self):
        input = self.open_price[len(self.open_price)-len(self.test_set)-60:].values
        input.shape
        input = self.sc.transform(input)
        
        x_test = []
        for i in range(1,31):
            x_test.append(input[i-1:i,0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        x_test.shape
            
        self.pred = self.reg.predict(x_test)
        self.pred = self.sc.inverse_transform(self.pred)
        
        return self
        
    def PlotGraph(self):
        print(fitdata(self.test_set))
        print(fitdata(self.pred))
        plt.plot(fitdata(self.test_set),color='green')
        plt.plot(fitdata(self.pred),color='red')
        plt.title(self.ticker)                                                                                                                                                                                      
        plt.show()
        
        return self
        
    def PredictTomorrow(self):
        #print("$ "+str(np.float32(fitdata(self.pred)[-1]).item()))
        return np.float32(fitdata(self.pred)[-1]).item()
    
