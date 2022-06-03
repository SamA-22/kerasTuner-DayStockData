import numpy as np
import pandas_datareader as web
import datetime as dt
import keras_tuner as kt
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

class stockHpTuner:
    """
    A class used to create a neural network with the best hyperparameters for day stock data.

    ...

    Attributes
    ----------
    ticker: string
        ticker of the stock that will be used
    features: list
        features that will be used to train the model
    predictionDays: int
        Amount of previouse days the model will use to make the next day prediction
    maxLayers: int
        Max number of layers that can be used in the model
    maxUnits: int
        Max number of units that can be used in the model
    unitDecriment: int
        Ammount of units decrimented each step when hyperparameter searching
    maxDropout: float
        Max dropout used fo rthe dropout layers within the model
    dropoutDecriment: float
        Ammount of dropout decreased each step when hyperparameter searching
    trainStart: datetime
        Start date of when train data should be retrived
    trainEnd: datetime
        End date of when train data should be retrived
    tesStart: datetime
        Start date of when test data should be retrived
    testEnd: datetime
        End date of when test data should be retrived
    batchSize: int
        Size of the input data that will be used in each batch when training the model
    searchEpochs: int
        maximum number of epochs to train one model when hyperparameter searching
    errorCheckLoopAmount: int
        Amount of models to be looped over and accuracy checked to ensure the best model is achieved
    predictEpochs: int
        Amount of epochs that will be used to train the model before making predicions
    _trainX: None
        Parameter that will store the data that will be used to train the model
    _trainY: None
        Parameter that will store the data that will be used to error check the model when trained with the trainX data
    
    Methods
    -------
    trainX(self)
        getter and setter used to update the _trainX parameter
    trainY(self)
        getter and setter used to update the _trainY parameter
    getTrainData(self)
        Pre-processes train data to acquire format and scale the data used for machine learning
    getTestData(self, scaler)
        Pre-processing test data to acquire format and scale the data to test the train neural network
    createModel(self, hp)
        Builds and trains a neural network model using the imported hyperparameters and train data
    hyperPerameterTweaking(self, trainX, trainY, testX, actualPrices)
        Uses hyperband tuner to build and train models. It searches for the models that achieve lowest mse. The the best models are accuracy checked using test data to ensure the model picked hasnt been overfitted which leads to a low mse without being accurate
    testAccuracy(predictedPrices, actualPrices)
        Calculates the percentage error using the the predicted prices and actual prices of the stock
    sortDict(dict)
        Sorts dictionaries by acending order of their values
    modelAccuracyMapping(self, actualPrices, predictedPrices)
        Models one feature of the actual stock prices on the predicted stock prices onto a graph
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        ticker: string
            ticker of the stock that will be used
        features: list, optional
            features that will be used to train the model
        predictionDays: int, optional
            Amount of previouse days the model will use to make the next day prediction
        maxLayers: int, optional
            Max number of layers that can be used in the model
        maxUnits: int, optional
            Max number of units that can be used in the model
        unitDecriment: int, optional
            Ammount of units decrimented each step when hyperparameter searching
        maxDropout: float, optional
            Max dropout used fo rthe dropout layers within the model
        dropoutDecriment: float, optional
            Ammount of dropout decreased each step when hyperparameter searching
        trainStart: datetime, optional
            Start date of when train data should be retrived
        trainEnd: datetime, optional
            End date of when train data should be retrived
        tesStart: datetime, optional
            Start date of when test data should be retrived
        testEnd: datetime, optional
            End date of when test data should be retrived
        batchSize: int, optional
            Size of the input data that will be used in each batch when training the model
        searchMaxEpochs: int, optional
            maximum number of epochs to train one model when hyperparameter searching
        errorCheckLoopAmount: int, optional
            Amount of models to be looped over and accuracy checked to ensure the best model is achieved
        predictEpochs: int, optional
            Amount of epochs that will be used to train the model before making predicions
        _trainX: None
            Parameter that will store the data that will be used to train the model
        _trainY: None
            Parameter that will store the data that will be used to error check the model when trained with the trainX data
        """
        varsDefaults = {
            "ticker": "",
            "features": ["High", "Low", "Open", "Adj Close"],
            "predictionDays": 1,
            "maxLayers": 3,
            "maxUnits": 512,
            "unitDecriment": 32,
            "maxDropout": 0.5,
            "dropoutDecriment": 0.1,
            "trainStart": dt.datetime(2012, 1, 1),
            "trainEnd": dt.datetime(2022, 4, 1),
            "testStart": dt.datetime(2022, 4, 2),
            "testEnd": dt.datetime.now(),
            "batchSize": 128,
            "searchMaxEpochs": 32,
            "errorCheckLoopAmount": 5,
            "predictEpochs": 32,
            "_trainX": None,
            "_trainY": None
        }

        for(var, default) in varsDefaults.items():
            setattr(self, var, kwargs.get(var, default))

    @property
    def trainX(self):
        """trainX getter.

        Returns 
        -------
        self._trainX : Matrix/List
            Parameter that will store the data that will be used to train the model"""
        return self._trainX
    @trainX.setter
    def trainX(self, value):
        """trainX setter, sets the attribute _trainX to the parameter value.

        Parameters
        ----------
        value : Matrix/List
            trainX data that has been gotten using getTrainData() method"""
        self._trainX = value

    @property
    def trainY(self):
        """trainY getter.

        Returns 
        -------
        self._trainY : Matrix/List
            Parameter that will store the data that will be used to train the model"""
        return self._trainY
    @trainY.setter
    def trainY(self, value):
        """trainY setter, sets the attribute _trainY to the parameter value.

        Parameters
        ----------
        value : Matrix/List
            trainY data that has been gotten using getTrainData() method"""
        self._trainY = value

    def getTrainData(self):
        """Pre-processes train data to acquire format and scale the data used for machine learning.

        Returns
        -------
        trainX : Marix/List
            the correctly formated inputs that will be used for the neural network
        trainY : Matrix/List
            the correctly formated data input that will be used when error checking within machine learning
        Scaler : object
            object used to scale and unscale data"""
        data = web.DataReader(self.ticker, "yahoo", self.trainStart, self.trainEnd)
        # The scaler is fit using the value of the features listed and used to scale the data.
        scaler = StandardScaler().fit(data[self.features].values)
        scaledData = scaler.transform(data[self.features].values)

        trainX = []
        trainY = []
        # loop starts from predictionDays to account for negative values when trying to retrive train data.
        for i in range(self.predictionDays, len(scaledData)):
            #list of values appened to trainX ranging from start of prediction days to end of prediction days.
            trainX.append(scaledData[i - self.predictionDays: i])
            #value of the day, after prediction days, appened to trainY.
            trainY.append(scaledData[i, 0: len(self.features)])

        trainX, trainY = np.array(trainX), np.array(trainY)

        return trainX, trainY, scaler

    def getTestData(self, scaler):
        """Pre-processing test data to acquire format and scale the data to test the train neural network.

        Parameters
        ----------
        scaler : object
            used to scale the test data as done to the train data

        Returns
        -------
        testX : Matrix/List
            the correctly formated data that'll be used to make predictions
        actrualPrices : Matrix/List
            actual historical data"""
        data = web.DataReader(self.ticker, "yahoo", self.testStart, self.testEnd)
        actualPrices = data[self.features]
        actualPrices = np.array(actualPrices)

        data = scaler.transform(actualPrices)
        #Deletes first few days of actual prices that was used to make the first prediction.
        for i in range(0, self.predictionDays):
            actualPrices = np.delete(actualPrices, 0, 0)

        testData = data

        testX = []
        #loop starts from predictionDays to account for negative values when trying to retrive test data.
        for i in range (self.predictionDays, len(testData)):
            #list of values appened to testX ranging from start of prediction days to end of prediction days.
            testX.append(testData[i - self.predictionDays: i])

        testX = np.array(testX)

        return testX, actualPrices

    def createModel(self, hp):
        """Builds and trains a neural network model using the imported hyperparameters and train data.

        Parameters
        ----------
        hp : object
            object that we pass to the model-building function, that allows us to define the space search of the hyperparameters
        
        Returns
        -------
        model : object
            Seqential model the neural network model that was trained using the x and y inputs with the set hyperparameters
        Object hp is an object that we pass to the model-building function, that allows us to define the space search of the hyperparameters
        return: Seqential model the neural network model that was trained using the x and y inputs with the set 
        hyperparameters.
        """
        model = Sequential()
        #Global variable hpValues used in model building to specify the min and max values as well as the steps to take.
        #First LSTM layer specifies the input shape. The last LSTM layer is to set return sequences to false so the model knows it will be the last LSTM layer.
        #Model can be changed depending on the data input however testing will need to be done to assure overfirring is not a problem.
        model.add(LSTM(hp.Int("inputUnit", min_value = self.unitDecriment, max_value = self.maxUnits, step = self.unitDecriment), return_sequences = True, input_shape = (self.trainX.shape[1], self.trainX.shape[2])))
        model.add(Dropout(hp.Float('input_Dropout_rate', min_value = self.dropoutDecriment, max_value = self.maxDropout, step = self.dropoutDecriment)))
        for i in range(hp.Int("nLayers", 0, self.maxLayers)):
            model.add(LSTM(hp.Int(f'{i}lstmLayer', 0, max_value = self.maxUnits, step = self.unitDecriment), return_sequences = True))
            model.add(Dropout(hp.Float(f'{i}dropoutLayer', min_value = self.dropoutDecriment, max_value = self.maxDropout, step = self.dropoutDecriment)))
        model.add(LSTM(hp.Int('layer_2_neurons', min_value = 0, max_value = self.maxUnits, step = self.unitDecriment)))
        model.add(Dropout(hp.Float('end_Dropout_rate', min_value = self.dropoutDecriment, max_value = self.maxDropout, step = self.dropoutDecriment)))
        model.add(Dense(self.trainY.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ["mse"])

        return model

    def hyperPerameterTweaking(self, trainX, trainY, testX, actualPrices):
        """Uses hyperband tuner to build model and test depending on the model that manages to get the mse the lowest. 
        The the best models are accuracy checkedusing test data to ensure the model picked hasnt overfitted which leads to a low mse without being accurate.

        Parameters
        ----------
        trainX : Matrix/List
            inputs that are used to train the neural network
        trainY : Matrix/List
            inputs that will be used to error check the neural network whilst training
        testX : Matrix/List
            inputs that are used to predict values once the network has been trained

        Returns
        -------
        (tuner.get_best_hyperparameters(num_trials = self.errorCheckLoopAmount)[modelNo]) : List
            this is the best preforming model in both mse and percentage error
        tuner : object
            the hyperband tuner that is used to go through different hyperpareter values and saves the trials to the dictionary specified"""
        predictionAccuracyScores = {}
        #Tuner instantiated with the objective of minimal mse (max_epochs, factor are both set to default values).
        tuner = kt.Hyperband(
            self.createModel, 
            objective = "mse",
            max_epochs = self.searchMaxEpochs,
            directory = f"Neural network\SelfMadeNNs\RNNs\hpTuner\hptuner-usingAClass\hpTunerMSE-day",
            project_name = f"{self.ticker}")
        #Preforms the search for the best hyperparameters using the train data(epochs and batch size used holds default values).
        tuner.search(trainX, trainY, epochs = self.searchMaxEpochs, batch_size = self.batchSize)
        #Loops through the top loopAmount of best preforming models
        for i in range (0, self.errorCheckLoopAmount):
            bestHyperparams = tuner.get_best_hyperparameters(num_trials = self.errorCheckLoopAmount)[i]
            model = tuner.hypermodel.build(bestHyperparams)
            #Trains the model and predicts using the test data
            model.fit(trainX, trainY, epochs = self.predictEpochs)
            predictedPrices = model.predict(testX)
            #Calculates the percentage error and stores the results in a dictionary
            percentageError = self.testAccuracy(predictedPrices, actualPrices)
            predictionAccuracyScores.update({i: percentageError})
            sortedAccScores = self.sortDict(predictionAccuracyScores)
            modelNo, errorAmount = list(sortedAccScores.items())[0]
            
        return tuner.get_best_hyperparameters(num_trials = self.errorCheckLoopAmount)[modelNo], tuner

    def testAccuracy(self, predictedPrices, actualPrices):
        """Calculates the percentage error using the the predicted prices and actual prices of the stock.

        Parameters
        ----------
        predictedPrices : Matrix/List
            prices that the model predicted
        actualPrices : Matrix/List
            actual prices of the test data
        
        Returns
        -------
        percentageError : Float
            absolute value of the percentage error calculated"""
        #Sums up all the values in the Matrix/Lists
        predSum = np.sum(predictedPrices)
        actSum = np.sum(actualPrices)
        #Uses percentage error formula.
        percentageError = np.subtract(predSum, actSum)
        percentageError = np.divide(percentageError, actSum)
        percentageError = np.multiply(percentageError, 100)
        return abs(percentageError)

    def sortDict(self, dict):
        """Sorts dictionaries by acending order of their values.

        Parameters
        ----------
        dict : dictionary
            a dictionary that will be sorted
        
        Returns
        -------
        sortedDict : dictionary
            a dictionary, sorted by acending values"""
        sortedDict = {k: v for k, v in sorted(dict.items(), key = lambda item: item[1])}
        return sortedDict

    def modelAccuracyMapping(self, actualPrices, predictedPrices):
        """Models one feature of the actual stock prices on the predicted stock prices onto a graph.

        Parameters
        ----------
        actualPrices : Matrix/List
            actual historical data of the test data
        predictedPrices : Matrix/List
            predictions made using the test data

        Retruns
        -------
        a graph : plt object
            Graph showing a plotted graph of the values specified"""
        #Takes all the values in the second column of each row of the matrixs'.
        predictedPrices = predictedPrices[:, 0]
        actualPrices = actualPrices[:, 0]
        #Plots a basic graph showing the predicted prices against the actual prices.
        plt.plot(actualPrices, color = "black", label = f"Actual {self.ticker} price")
        plt.plot(predictedPrices, color = "green", label = f"Predicted price")
        plt.title(f"{self.ticker} Share Price")
        plt.xlabel("Time")
        plt.ylabel(f"{self.ticker} Share price")
        plt.legend()
        plt.show()
