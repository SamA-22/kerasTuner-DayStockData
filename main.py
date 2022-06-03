from hpTunerMSE_Day import stockHpTuner
import datetime as dt

"""Main script to show how to utilise the hpTunerMSE_Day script"""
# A stockHpTuner object is created and parameters are listed with wanted values.
hpTuner = stockHpTuner(
    # (Mandatory parameter) ticker needs to be specified to determine the stock data that will be tuned.
    ticker = "AMD",
    # (optional) features that can be used ["High", "Low", "Open", "Close", "Adj Close", "Volume"] volume is not recommended. Default value removes volume and close.
    features = ["High", "Low", "Open", "Adj Close"],
    # (optional) shorter amount of prediction days used will allow higher accuracy when predicting close future days. Defualt value is 1
    predictionDays = 1,
    # (optional but change recomended depending on input data) warning increasing amount will increase time taken to run. 
    # Recomended values 0 - 5. Default value given is 3.
    maxLayers = 3,
    # (optional but change recomended depending on input data) warning increasing amount will increase time taken to run.
    # Recomended values (length of the output values) - (length of the input values). Default value given is 512.
    maxUnits = 512,
    # (optional but change recomended depending on input data) warning decreasing amount will increase time taken to run.
    # Recomended values depends on the maxUnits used. Default value given is 32.
    unitDecriment = 32,
    # (optional but change recomended depending on input data) warning increasing amount will increase time taken to run.
    # Recomended values 0.1 - 0.8. Default value given is 0.5.
    maxDropout = 0.5,
    # (optional but change recomended depending on input data) warning decreasing amount will increase time taken to run.
    # Recomended values depends on the maxDropout used. Default value given is 0.1.
    dropoutDecriment = 0.1,
    # (optional but change recomended. read the README file for more info) warning increasing how far back to take data will increase time taken to run.
    # Recomended values depends on the ticker used. Default value dt.datetime(2012, 1, 1).
    trainStart = dt.datetime(2018, 1, 1),
    # (optional but change recomended. read the README file for more info)
    # Recomended values depends on how much test data will be used. Default value dt.datetime(2022, 4, 1).
    trainEnd = dt.datetime(2022, 4, 1),
    # (optional but change recomended. read the README file for more info) warning Using a large amount of test data used limits train data.
    # No recomended values. Default value dt.datetime(2022, 4, 2).
    testStart = dt.datetime(2022, 4, 2),
    # (optional but change recomended. read the README file for more info)
    # Recomended value dt.datetime.now. Default value dt.datetime(dt.datetime.now).
    testEnd = dt.datetime.now(),
    # (optional) Recomended values 1 - (length of the input values). Default value 128.
    batchSize = 128,
    # (optional) warning increasing amount will increase time taken to run. Default value 32.
    searchMaxEpochs = 32,
    # (optional) warning increasing amount will increase time taken to run. Default value 5.
    errorCheckLoopAmount = 5,
    # (optional) warning increasing amount will increase time taken to run. Default value 32.
    predictEpochs = 32)

#Fetches and formats train data then sets objects parameters to that data.
trainX, trainY, scaler = hpTuner.getTrainData()
hpTuner.trainX = trainX
hpTuner.trainY = trainY
#Fetches and formats test data as well as the actual prices.
testX, actualPrices = hpTuner.getTestData(scaler)
#Tuning occures to obtain best hyperparameters using the given data given.
bestHp, tuner = hpTuner.hyperPerameterTweaking(trainX, trainY, testX, actualPrices)
bestModel = tuner.hypermodel.build(bestHp)
#Fits the a model and stores MSE history to then check the epoch that has the lowest MSE.
bestEpochTest = bestModel.fit(trainX, trainY, epochs = hpTuner.searchMaxEpochs)
history = bestEpochTest.history["mse"]
bestEpoch = history.index(min(history)) + 1
#Refits the model using the best epoch loop found previously and uses this model and test data to make predictions.
bestModel.fit(trainX, trainY, epochs = bestEpoch)
predictedPrices = bestModel.predict(testX)
predictedPrices = scaler.inverse_transform(predictedPrices)
#Uses the predicted and actual data to map the feature in the first index as a visual representation of acuracy.
hpTuner.modelAccuracyMapping(actualPrices, predictedPrices)
