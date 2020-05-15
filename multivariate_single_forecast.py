from market import EquityData
from models.lstm import split, split_multivariate, show_plot
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import plot_train_history
from technical_analysis import moving_average


import pandas as pd
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta import add_all_ta_features
import pyfinancialdata



tf.random.set_seed(42)

BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCHS = 10
CLASSIFICATION = True

step = 1
history_size = 180
target_distance = 6

features_considered = ['Close', 'Volume', 'MA_short', 'MA_long', 'Wilders_EMA', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_bbhi', 'bb_bbli']
features_considered = ['Close', 'Volume', 'Wilders_EMA', 'bb_bbm', 'bb_bbh', 'bb_bbl',
                       'bb_bbhi', 'bb_bbli', 'trend_adx', 'momentum_stoch_signal', 
                       'trend_trix', 'trend_ichimoku_a', 'momentum_kama', 'momentum_tsi', 
                       'volatility_kcli']
features_considered = ['close', 'MA_long', 'MA_short']
# features_considered = ['Close', 'MA_short', 'MA_long', 'Wilders_EMA']
# Initialize Bollinger Bands Indicator
e = EquityData('data/SPY.csv', 'SPY')
data = pyfinancialdata.get_multi_year(provider='histdata', instrument='SPXUSD', years=[2016, 2017, 2018], time_group='10min')
e.data = data
indicator_bb = BollingerBands(close=e.data['close'], n=20, ndev=2)
# e.data = add_all_ta_features(e.data, open="open", high="high", low="low", close="close")

e.data['MA_long'] = e.data['close'].rolling(window=52).mean()
e.data['MA_short'] = e.data['close'].rolling(window=7).mean()
# e.data['MA_short'] = moving_average(e, window=5)
# e.data['Wilders_EMA'] = e.close().ewm(alpha=1/history_size, adjust=False).mean()

# # Add Bollinger Bands features
# e.data['bb_bbm'] = indicator_bb.bollinger_mavg()
# e.data['bb_bbh'] = indicator_bb.bollinger_hband()
# e.data['bb_bbl'] = indicator_bb.bollinger_lband()
# # Add Bollinger Band high indicator
# e.data['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
# # Add Bollinger Band low indicator
# e.data['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
# e.data = e.data[21:]
print(list(e.data))
# import pdb
# pdb.set_trace()

EVALUATION_INTERVAL = int(e.data.shape[0]/BATCH_SIZE) * 1
features = e.data[features_considered]
features.index = e.data.index
# assert(list(features)[0] == 'Close')
features = features.dropna()
features = features[26:]


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

single_step_model = tf.keras.models.Sequential()

single_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True,
                                           input_shape=x_train_single.shape[-2:]))
# single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))

single_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
if CLASSIFICATION:
  single_step_model.add(tf.keras.layers.Dense(16, activation='relu'))
  single_step_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
else:
  single_step_model.add(tf.keras.layers.Dense(1))

# single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
single_step_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

val_callback = tf.keras.callbacks.ModelCheckpoint(
    'checkpoints/multivariate_single_model', monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch'
)

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50, callbacks=[val_callback])


plot_train_history(single_step_history,
                   'Single Step Training and validation loss')


for x, y in val_data_single.take(10):
  y_pred = single_step_model.predict(x)[0]
  print(f"prediction: {y_pred}")
  if CLASSIFICATION:
    if y_pred >= 0.5:
      y_pred = 1
    else:
      y_pred = 0
  plot = show_plot([x[0][:, 0].numpy(), y[0].numpy(),
                    y_pred], target_distance,
                   'Single Step Prediction')
  plot.show()
