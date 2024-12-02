import numpy as np
import pandas as pd
import yfinance as yf
import math

def get_stock(start, end, stock, restricted_days):
    '''
    Return the data as Numpy arrays and incrementing indexes for each value for the time series.

    Parameters:
    -- start: start date "Year-Month-Day"
    -- end: end date "Year-Month-Day"
    -- stock: The label of the stock to download from yahoo stocks
    -- restriced_days: int, the number of last days to be removed from the dataframe 

    Returns:
    -- df: 1d Numpy array containing all the closing prices from the start to end data except for the last 100 days
    -- restricted_df: Contains the last <restriced_days> days of the data
    -- index: incrementing indexes for the closing prices
    '''
    
    yfd = yf.download(stock, start=start, end=end)
    df = pd.DataFrame({'Close': yfd['Close']})
    df = df.dropna().reset_index(drop=True)

    restricted_df = np.array(df[len(df) - restricted_days:]) # last 100 days prices
    df = np.array(df[:len(df) - restricted_days]) # remove the restricted days from the main df
    index = np.linspace(1, df.shape[0], df.shape[0])

    return df, restricted_df, index 

def bin_data(df, bin_size):
    '''
    Returns the data split into bins of size <bin_size>. The bins are used to predict the day on index <bin_size> + 1.
    
    Parameters:
    -- df: dataset containing the closing prices for the stock
    -- bin_size: size of bins for the data

    Returns:
    -- bins: Numpy array size (len(df) - bin_size, bin_size), bins created based on a sliding window incrementing by 1
    '''
    
    bins = np.array([df[i:i + bin_size].T.reshape(-1) for i in range(0, len(df) - bin_size - 1, 1)])
    return bins 

def split_data(bins, df, index, bin_size):
    # Generally it is good practice to train data on 80 % of the data and test on 20%
    train_size = math.floor(len(bins)* 0.8) # 80% of the data
    test_size = len(bins) - train_size # 20% of the data
    assert train_size + test_size == len(bins)
    
    # split into X and y
    X_train = np.array(bins[:train_size])
    y_train = np.array(df[bin_size:bin_size + train_size]).T.reshape(-1)
    train_index = index[:train_size]
    
    X_test = np.array(bins[train_size:])
    y_test = np.array(df[bin_size + train_size: bin_size + train_size + test_size]).T.reshape(-1) # the value after the bin_size
    test_index = index[train_size + 1: train_size + 1 + test_size]

    return X_train, y_train, X_test, y_test, train_index, test_index, train_size, test_size
    

def test_stock(stock, restrict=100, bin_size=365):

    # Get the stock data
    df, restricted_df, index = get_stock(stock[1], stock[2], stock[0], restrict)

    # Create bins
    bins = bin_data(df, bin_size)

    # Split the data
    X_train, y_train, X_test, y_test, train_index, test_index, train_size, test_size = split_data(bins, df, index)

    # Run the linear model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = np.floor(model.predict(X_test))
    y_test = np.floor(y_test)

    ''' Auto-regressive '''
    running_bin = bins[-1].copy()
    next_predictions = []
    for i in range(restrict): # predict the next 100 days
        prediction = model.predict([running_bin])
    
        # feed the prediction back into the input
        running_bin[:-1] = running_bin[1:]
        running_bin[-1] = prediction[0]
    
        # store predictions
        next_predictions.append(prediction[0])
    
    # print(next_predictions[-10:])
    days = np.arange(1, len(next_predictions) + 1)
    
    
    plt.figure()
    plt.title(f"{stock[0]} stock")
    plt.plot(days, next_predictions, label = "Auto-Regressive")
    plt.plot(days, restricted_df, label = f"Actual")
    plt.xlabel(f"Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()