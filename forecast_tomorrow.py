from market import EquityData
from models.lstm import split_multivariate
from utils import create_time_steps
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


features_considered = ['Close', 'Volume']

e = EquityData('data/SPY.csv', 'SPY')
features = e.data[features_considered]
features.index = e.date()

step = 1
history_size = 90
target_distance = 7

dataset = features.values
train_split = int(len(dataset) * 0.7)

data_mean = dataset[:train_split].mean(axis=0)
data_std = dataset[:train_split].std(axis=0)
dataset = (dataset-data_mean)/data_std
data = dataset[-history_size:]
data = np.reshape(data, (1, data.shape[0], data.shape[1]))

model = tf.keras.models.load_model('saved_models/multivariate_multi_model')
y = model.predict(data)


plt.figure(figsize=(12, 6))
num_in = create_time_steps(data.shape[1])
num_out = target_distance

data = np.reshape(data, (data.shape[1], data.shape[2]))


print(data[:,1].shape)
print(num_in)
plt.plot(num_in, np.array(data[:, 1]), label='History')
plt.plot(np.arange(num_out)/step, np.array(y[0]), 'ro',
            label='Predicted Future')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

print(y)