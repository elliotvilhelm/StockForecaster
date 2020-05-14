from market import EquityData
from models.lstm import split, split_multivariate, show_plot, create_time_steps
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import plot_train_history, multi_step_plot


BATCH_SIZE = 256
BUFFER_SIZE = 10000

EVALUATION_INTERVAL = 200
EPOCHS = 10

step = 1
history_size = 20
target_distance = 7

features_considered = ['Close', 'Volume']


e = EquityData('data/SPY.csv', 'SPY')
features = e.data[features_considered]
features.index = e.date()


features.plot(subplots=True)
plt.show()

dataset = features.values
x_train_multi, y_train_multi, x_val_multi, y_val_multi = split_multivariate(dataset, history_size, target_distance, step, single_step=False)

print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]), step)

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(target_distance))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')


multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0], step)