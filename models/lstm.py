import numpy as np
from market import EquityData
import tensorflow as tf
import matplotlib.pyplot as plt


tf.random.set_seed(42)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False, classification=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      if classification:
        if target[i] <= target[i+target_size]: # went up
          labels.append(1)
        else:
          labels.append(0)

      else:
        labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


def create_time_steps(length):
  return list(range(-length, 0))


def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.grid(True)
  plt.xlabel('Time-Step')
  plt.show()
  return plt


def split(equity_data):
  history_size = 20
  target_distance = 1
  train_split = int(len(equity_data.data) * 0.7)

  uni_data = equity_data.close()
  uni_data.index = equity_data.date()
  uni_data = uni_data.values
  uni_train_mean = uni_data[:train_split].mean()
  uni_train_std = uni_data[:train_split].std()
  uni_data = (uni_data-uni_train_mean)/uni_train_std



  x_train_uni, y_train_uni = univariate_data(uni_data, 0, train_split, history_size, target_distance)
  x_val_uni, y_val_uni = univariate_data(uni_data, train_split, None, history_size, target_distance)

  return x_train_uni, y_train_uni, x_val_uni, y_val_uni


def split_multivariate(dataset, history_size, target_distance, step, single_step=False, classification=False):
  train_split = int(len(dataset) * 0.7)

  data_mean = dataset[:train_split].mean(axis=0)
  data_std = dataset[:train_split].std(axis=0)
  dataset = (dataset-data_mean)/data_std


  x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                   train_split, history_size,
                                                   target_distance, step,
                                                   single_step=single_step, classification=classification)
  x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                                train_split, None, history_size,
                                                target_distance, step,
                                                single_step=single_step, classification=classification)
  
  return x_train_single, y_train_single, x_val_single, y_val_single


def baseline(history):
  return np.mean(history)

