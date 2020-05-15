from market import EquityData
from models.lstm import split, split_multivariate, show_plot
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import plot_train_history
from technical_analysis import moving_average


import pandas as pd
from ta.utils import dropna
from ta.volatility import BollingerBands


tf.random.set_seed(42)

BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCHS = 10
CLASSIFICATION = True

step = 1
history_size = 21
target_distance = 2

features_considered = ['Close', 'Volume', 'MA_short', 'MA_long', 'Wilders_EMA', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_bbhi', 'bb_bbli']
# features_considered = ['Close', 'MA_short', 'MA_long', 'Wilders_EMA']
# Initialize Bollinger Bands Indicator
e = EquityData('data/SPY.csv', 'SPY')
indicator_bb = BollingerBands(close=e.close(), n=20, ndev=2)
e.data['MA_short'] = moving_average(e, window=history_size)
e.data['MA_long'] = moving_average(e, window=5)
e.data['Wilders_EMA'] = e.close().ewm(alpha=1/history_size, adjust=False).mean()

# Add Bollinger Bands features
e.data['bb_bbm'] = indicator_bb.bollinger_mavg()
e.data['bb_bbh'] = indicator_bb.bollinger_hband()
e.data['bb_bbl'] = indicator_bb.bollinger_lband()
# Add Bollinger Band high indicator
e.data['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
# Add Bollinger Band low indicator
e.data['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
e.data = e.data[21:]

EVALUATION_INTERVAL = int(e.data.shape[0]/BATCH_SIZE) * 2
features = e.data[features_considered]
assert(list(features)[0] == 'Close')
features.index = e.date()


# features.plot(subplots=True)
# plt.show()

dataset = features.values
x_train_single, y_train_single, x_val_single, y_val_single = split_multivariate(dataset, history_size, target_distance,
                                                                                step, single_step=True, classification=CLASSIFICATION)
                                                                                
print('Single window of past history : {}'.format(x_train_single[0].shape))

train_data_single = tf.data.Dataset.from_tensor_slices(
    (x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices(
    (x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


single_step_model = tf.keras.models.load_model('checkpoints/multivariate_single_model')

# for x, y in val_data_single.take(3):
#   y_pred = single_step_model.predict(x)[0]
#   print(f"prediction: {y_pred}")
#   if CLASSIFICATION:
#     if y_pred >= 0.5:
#       y_pred = 1
#     else:
#       y_pred = 0
#   plot = show_plot([x[0][:, 0].numpy(), y[0].numpy(),
#                     y_pred], target_distance,
#                    'Single Step Prediction')
#   plot.show()



if CLASSIFICATION:
    avgs = []
    for x, y in val_data_single.take(int(x_val_single.shape[0] / BATCH_SIZE)):
        r = single_step_model.evaluate(x, y)
        avgs.append(r[1])
    
    print(single_step_model.metrics_names)
    print(sum(avgs)/len(avgs))