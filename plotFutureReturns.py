import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

from dataReader import get_data

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

company_list, company_names = get_data()

for i, company in enumerate(company_list):
    df = company[['Adj Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(np.array(df).reshape(-1, 1))

    # splitting dataset into train and test split
    training_size = int(len(df) * 0.65)
    test_size = len(df) - training_size
    train_data, test_data = df[0:training_size, :], df[training_size:len(df), :1]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 50
    if len(train_data) > time_step and len(test_data) > time_step:
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
    else:
        print(f"Time step is larger than the size of the data for {company_names[i]}. Skipping this company.")
        continue

    # reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=0)

    # Let's Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Plotting
    # shift train predictions for plotting
    look_back = time_step
    trainPredictPlot = np.empty_like(df)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df)
    testPredictPlot[:, :] = np.nan
    start_idx = len(train_predict) + (look_back * 2)
    testPredictPlot[start_idx:start_idx + len(test_predict), :] = test_predict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.title(f"{company_names[i]} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend(["Actual", "Train Prediction", "Test Prediction"])
    plt.show()