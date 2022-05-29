import numpy as np
import pandas_datareader as web
import datetime as dt
import keras_tuner as kt
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

"""Important Global variables
string ticker: ticker symbol used to get historical data of that company.
int featureNo: default to 4 as the amount of features used are 'High', 'Low', 'Open', 'Adj Close'. 
int predictionDays: amount of previous day data used to make the next day prediction. Value is a hyperParamer. 
"""
ticker = ""
featureNo = 4 #Default value
predictionDays = 1 #Default value

"""Hyper Params to be tweaked for testing (Read Me... for more info)"""
hpValues = {
    #Max number of layers that can be used      0-5(aprx recomended values for default data set)
    "maxLayers": 3,
    #Max number of units used in network        2-2580(aprx recomended values for default data set)
    "maxUnits": 512,
    #Amount the units decriment each step       (No recomended amount)
    "unitDecriment": 32,
    #Max dropout used for dropout layers       0.2-0.8(aprx recomended values for default data set)
    "maxDropout": 0.5,
    #Amount of dropout decresed with each step       (No recomended amount))
    "dropoutDecriment": 0.1}

"""Train Data Dates(yyyy, mm, dd)""" #Default values
trainStart = dt.datetime(2012, 1, 1)
trainEnd = dt.datetime(2022, 4, 1)
"""Test Data Dates(yyyy, mm, dd)""" #Default value
TestStart = dt.datetime(2021, 4, 2)
TestEnd = dt.datetime.now()

def getTrainData():
    """Uses pandas_datareader to read historical data from yahoo finance to use as the train data. 
    Also, creates a standard scaler using sklearn to make it easier for the model.
    return: Matrix of scaled down data, the scaler used to scale the data.
    """
    data = web.DataReader(ticker, "yahoo", trainStart, trainEnd)
    # The scaler is fit using the value of the features listed and used to scale the data.
    scaler = StandardScaler().fit(data[["High", "Low", "Open", "Adj Close"]].values)
    scaledData = scaler.transform(data[["High", "Low", "Open", "Adj Close"]].values)
    return scaledData, scaler

def getTestData(scaler):
    """Uses pandas_datareader to read historical data from yahoo finance to use as the test data.
    scaler is input to scale the test data as done to the train data.
    return: Scaled down matrix of historical data, matrix of actual historical data.
    """
    data = web.DataReader(ticker, "yahoo", TestStart, TestEnd)
    actualPrices = data[["High", "Low", "Open", "Adj Close"]]
    actualPrices = np.array(actualPrices)

    data = scaler.transform(actualPrices)
    #Deletes first few days of actual prices that was used to make the first prediction.
    for i in range(0, predictionDays):
        actualPrices = np.delete(actualPrices, 0, 0)

    return data, actualPrices

def formatTrainData(importedTrainData):
    """Pre-processing of train data to format it correctly for machine learning.
    Matrix/List importedTrainData train data that will be used as machine learning.
    return: Matrix/List trainX the correctly formated inputs that will be used for the neural network,
    Matrix/List trainY the correctly formated data input that will be used when error checking within machine learning.
    """
    trainX = []
    trainY = []
    # loop starts from predictionDays to account for negative values when trying to retrive train data.
    for i in range(predictionDays, len(importedTrainData)):
        #list of values appened to trainX ranging from start of prediction days to end of prediction days.
        trainX.append(importedTrainData[i - predictionDays: i])
        #value of the day, after prediction days, appened to trainY.
        trainY.append(importedTrainData[i, 0: featureNo])

    trainX, trainY = np.array(trainX), np.array(trainY)
    return trainX, trainY

def formatTestData(importedTestData):
    """Pre-processing of test data to format it correctly for machine learning.
    Matrix/List importedTestData test data that will be used as the input when predicting, post learning.
    return: Matrix/List testX the correctly formated data that'll be used to make predictions.
    """
    testData = importedTestData

    testX = []
    #loop starts from predictionDays to account for negative values when trying to retrive test data.
    for i in range (predictionDays, len(testData)):
        #list of values appened to testX ranging from start of prediction days to end of prediction days.
        testX.append(testData[i - predictionDays: i])

    testX = np.array(testX)

    return testX

def createModel(hp):
    """Builds and trains a neural network model using the imported hyperparameters and train data.
    Object hp is an object that we pass to the model-building function, that allows us to define the space search of the hyperparameters
    return: Seqential model the neural network model that was trained using the x and y inputs with the set 
    hyperparameters.
    """
    model = Sequential()
    #Global variable hpValues used in model building to specify the min and max values as well as the steps to take.
    #First LSTM layer specifies the input shape. The last LSTM layer is to set return sequences to false so the model knows it will be the last LSTM layer.
    #Model can be changed depending on the data input however testing will need to be done to assure overfirring is not a problem.
    model.add(LSTM(hp.Int("inputUnit",min_value=hpValues["unitDecriment"],max_value=hpValues["maxUnits"],step=hpValues["unitDecriment"]),return_sequences=True,input_shape=(trainX.shape[1],trainX.shape[2])))
    model.add(Dropout(hp.Float('input_Dropout_rate',min_value=hpValues["dropoutDecriment"],max_value=hpValues["maxDropout"],step=hpValues["dropoutDecriment"])))
    for i in range(hp.Int("nLayers", 0, hpValues["maxLayers"])):
        model.add(LSTM(hp.Int(f'{i}lstmLayer',0,max_value=hpValues["maxUnits"],step=hpValues["unitDecriment"]),return_sequences=True))
        model.add(Dropout(hp.Float(f'{i}dropoutLayer',min_value=hpValues["dropoutDecriment"],max_value=hpValues["maxDropout"],step=hpValues["dropoutDecriment"])))
    model.add(LSTM(hp.Int('layer_2_neurons',min_value=0,max_value=hpValues["maxUnits"],step=hpValues["unitDecriment"])))
    model.add(Dropout(hp.Float('end_Dropout_rate',min_value=hpValues["dropoutDecriment"],max_value=hpValues["maxDropout"],step=hpValues["dropoutDecriment"])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ["mse"])

    return model

def hyperPerameterTweaking(trainX, trainY, testX, actualPrices):
    """Uses hyperband tuner to build model and test depending on the model that manages to get the mse the lowest. The the best models are accuracy checked
    using test data to ensure the model picked hasnt overfitted which leads to a low mse without being accurate.
    Matrix/List trainX the inputs that are used to train the neural network,
    Matrix/List trainY the inputs that will be used to error check the neural network whilst training,
    Matrix/List testX the inputs that are used to predict values once the network has been trained.
    Return: List (tuner.get_best_hyperparameters(num_trials=10)[modelNo]) this is the best preforming model in both mse and percentage error.
    Return Object tuner the hyperband tuner that is used to go through different hyperpareter values and saves the trials to the dictionary specified.
    """
    predictionAccuracyScores = {}
    #Tuner instantiated with the objective of minimal mse (max_epochs, factor are both set to default values).
    tuner = kt.Hyperband(
        createModel, 
        objective = "mse",
        max_epochs = 32,
        directory = f"Neural network\SelfMadeNNs\RNNs\hpTuner\{ticker}",
        project_name = f"{ticker}")
    #Preforms the search for the best hyperparameters using the train data(epochs and batch size used holds default values).
    tuner.search(trainX, trainY, epochs = 20, batch_size = 256)
    #Loops through the top loopAmount of best preforming models
    loopsAmount = 10
    for i in range (0, loopsAmount):
        bestHyperparams = tuner.get_best_hyperparameters(num_trials=loopsAmount)[i]
        model = tuner.hypermodel.build(bestHyperparams)
        #Trains the model and predicts using the test data
        model.fit(trainX, trainY, epochs = 32)
        predictedPrices = model.predict(testX)
        #Calculates the percentage error and stores the results in a dictionary
        percentageError = testAccuracy(predictedPrices, actualPrices)
        predictionAccuracyScores.update({i: percentageError})
        sortedAccScores = sortDict(predictionAccuracyScores)
        modelNo, errorAmount = list(sortedAccScores.items())[0]
        
    return tuner.get_best_hyperparameters(num_trials=loopsAmount)[modelNo], tuner


def testAccuracy(predictedPrices, actualPrices):
    """Calculates the percentage error using the the predicted prices and actual prices of the stock.
    Matrix/List predictedPrices prices that the model predicted,
    Matrix/List actualPrices actual prices of the test data.
    Return: float percentageError absolute value of the percentage error calculated
    """
    #Sums up all the values in the Matrix/Lists
    predSum = np.sum(predictedPrices)
    actSum = np.sum(actualPrices)
    #Uses percentage error formula.
    percentageError = np.subtract(predSum, actSum)
    percentageError = np.divide(percentageError, actSum)
    percentageError = np.multiply(percentageError, 100)
    return abs(percentageError)

def sortDict(dict):
    """Sorts dictionaries by acending order of their values.
    Dictionary dict the dictionary that will be sorted.
    Return: Dictionary sortedDict a dictionary sorted by acending values"""
    sortedDict = {k: v for k, v in sorted(dict.items(), key = lambda item: item[1])}
    return sortedDict

def modelAccuracyMapping(actualPrices, predictedPrices):
    """Models one feature of the actual stock prices on the predicted stock prices onto a graph.
    Matrix/List actualPrices actual historical data of the test data,
    Matrix/List predictedPrices predictions made using the test data.
    Return: Graph showing a plotted graph of the values specified."""
    #Takes all the values in the first column of each row of the matrixs'.
    predictedPrices = predictedPrices[:, 0]
    actualPrices = actualPrices[:, 0]
    #Plots a basic graph showing the predicted prices against the actual prices.
    plt.plot(actualPrices, color = "black", label = f"Actual {ticker} price")
    plt.plot(predictedPrices, color = "green", label = f"Predicted price")
    plt.title(f"{ticker} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{ticker} Share price")
    plt.legend()
    plt.show()

#Fetches and formats train data
trainData, scaler = getTrainData()
trainX, trainY = formatTrainData(trainData)
#Fetches and formats test data
testX, actualPrices = getTestData(scaler)
testX = formatTestData(testX)

bestHyperparams, tuner = hyperPerameterTweaking(trainX, trainY, testX, actualPrices)
bestModel = tuner.hypermodel.build(bestHyperparams)
#Fits the model and stores mse each time to then check the epoch that has the lowest mse.
bestEpochTest = bestModel.fit(trainX, trainY, epochs = 32)
accuracy_perEpoch = bestEpochTest.history["mse"]
bestEpoch = accuracy_perEpoch.index(min(accuracy_perEpoch)) + 1
#refits the model using the best epoch loop found previously and displays the predicted prices on the actual prices.
bestModel = tuner.hypermodel.build(bestHyperparams)
bestModel.fit(trainX, trainY, epochs = bestEpoch)
predictedPrices = bestModel.predict(testX)
predictedPrices = scaler.inverse_transform(predictedPrices)
modelAccuracyMapping(actualPrices, predictedPrices)