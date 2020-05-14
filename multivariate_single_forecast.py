from market import EquityData
from models.lstm import split, split_multivariate, show_plot
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import plot_train_history


tf.random.set_seed(42)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

EVALUATION_INTERVAL = 200
EPOCHS = 10

features_considered = ['Close', 'Volume']


e = EquityData('data/SPY.csv', 'SPY')
features = e.data[features_considered]
features.index = e.date()

step = 1
history_size = 20
target_distance = 7

features.plot(subplots=True)
plt.show()

dataset = features.values
x_train_single, y_train_single, x_val_single, y_val_single = split_multivariate(dataset, history_size, target_distance, step, single_step=True)
print('Single window of past history : {}'.format(x_train_single[0].shape))

train_data_single = tf.data.Dataset.from_tensor_slices(
    (x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices(
    (x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)


plot_train_history(single_step_history,
                   'Single Step Training and validation loss')


for x, y in val_data_single.take(3):
  plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                    single_step_model.predict(x)[0]], 12,
                   'Single Step Prediction')
  plot.show()