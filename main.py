import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from keras.models import load_model

filename="EURUSD.xlsx"

def RNNmodel1(filename):
    xl = pd.ExcelFile(filename)
    res = len(xl.sheet_names)
    start=2000
    dataset = pd.read_excel(filename,str(start))
    for i in range(1,res):
        year=i+start
        tempdataset = pd.read_excel(filename,str(year))
        dataset= dataset.append(tempdataset, ignore_index=True)
    #training for T+5
    X_train= dataset.iloc[0:-5].values
    Y_train= dataset.iloc[5:].values
    X_train= X_train[:]
    Y_train= Y_train[:,-1]
    xlen=len(X_train)
    X_train= np.reshape(X_train, (xlen,8,1))
    #--------RNN MODEL---------#
    def baseline_model():
        regressor = Sequential()
        regressor.add(LSTM(units=64,return_sequences=True, activation='relu',input_shape=(None,1)))
        regressor.add(LSTM(units=64,return_sequences=True, activation='relu',input_shape=(None,1)))
        regressor.add(LSTM(units=64,return_sequences=True, activation='relu',input_shape=(None,1)))
        regressor.add(LSTM(units=8, activation='relu',input_shape=(None,1)))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mse')
        return regressor
    estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=32, verbose=1)
    estimator.fit(X_train, Y_train)
    estimator.model.save('RNNmodel1_EURUSD.h5')
    return estimator
    
    

if __name__=="__main__":
    estimator=RNNmodel1(filename)

    #-------------VALIDATING MODEL--------------#
    TESTdataset= pd.read_excel('EURUSD_2021JUN.xlsx','2021')
    X_test= TESTdataset.iloc[0:-5].values
    Y_test= TESTdataset.iloc[5:].values
    X_test= X_test[:]
    Y_test= Y_test[:,-1]
    tlen=len(X_test)
    X_test= np.reshape(X_test, (tlen,8,1))
    #---------for prediction of data using saved model------#
    def baseline_model():
        pass
    estimator = KerasRegressor(build_fn=baseline_model, epochs=None, batch_size=None, verbose=None)
    estimator.model = load_model('RNNmodel1_EURUSD.h5')
    y_pred= estimator.predict(X_test)
    currentPrice= TESTdataset.iloc[:].values
    currentPrice=currentPrice[:,-1]
    # #------TO MUCH DATA FOR MATPLOT LIB---#
    
    plt.figure(figsize=(150,150))
    plt.plot(currentPrice,color='green', label='REAL PRICE current time T')
    plt.plot(Y_test,color='red', label='REAL PRICE  for T+5')
    plt.plot(y_pred,color='blue', label='PRED PRICE for T+5 min')
    plt.title('YEAR 2021 JULY APPLE')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()