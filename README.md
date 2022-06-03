# hpTuner-usingKerasTuner

Machine learning code in relation to stocks. Tunes hyperparameters and outputs optimum model using daily historical data

## Latest version/ Added features

- Version 2.0.0
  - Moved all code into a class cleaning the code
  - addition of docs to help others understand the code much simpler
  
## Key Information

### Libraries

- tensorflow-[keras](https://keras.io/) used for machine learning
- [keras-tuner](https://keras.io/keras_tuner/) used for finding best hyperparameters
- [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/#) used to read from Yahoo Fnance
- [matplotlib](https://matplotlib.org/) used to graph data
- [scikit-learn](https://scikit-learn.org/stable/) used to scale data

### Historical Data

- Historical data is retrived from [Yahoo Finance](https://uk.finance.yahoo.com/)

### Key Global Variables

Description of the parameters needed to be inputed are present in both main code and the backend code.

### Whilst running

- Code may take awhile to run depending on the max hyperprameters that are given. Changing the verbose in the createModel function (line 121) to 1-2 will output a graphic that will show if the code is still running.

## Potential Updates

- [x] Re-model the structure of the script to utilise classes.
- [] Delete irelivant trials and keep best tried to conserve storage space.
- [] When perfect hyperparams are found train the model till weights that predict most accurately are saved.
- [] List holding multiple tickers that will run one after the next.
- [] Try catch to hyperparameterTweaking() to ensure train data retrieved and set before training model. 
