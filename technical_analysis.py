def moving_average(equity, window=30):
    return equity.data['Close'].rolling(window=window).mean()