# LIBRERIAS

import jupyter, scipy, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import scipy.signal  
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import backend as K
#from tensorflow.keras import datasets, layers, models


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datetime import datetime

import IPython
import IPython.display
import time

import csv


#Cargamos los datos
df = pd.DataFrame(pd.read_csv("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Datos_teleconexion_vs\\df_tf_nodate.csv"))

df2 = df[['AO','NAO','SCAND']]

# Separamos en train, test y validación
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]


# CLASE PARA MANEJAR VENTANAS TEMPORALES

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  

##############################################################################################################

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

###############################################################################################################

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

###############################################################################################################

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

################################################################################################################

def plot(self, model=None, plot_col='AO', max_subplots=1):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [d]')

WindowGenerator.plot = plot

# ROOT MEAN SQUARED ERROR

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# MODELOS

def compile_and_fit(model, window, patience=2, epochs = 500, es = 1):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                    patience=patience,
                                                    mode='min')
  model.compile(loss=rmse,
                optimizer=tf.optimizers.SGD(clipnorm=1.0),
                metrics=[tf.metrics.RootMeanSquaredError()])
  
  if es == 1:
    history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  else:
    history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val)
  return history

###############  CREACION DE VENTANA MOVIL  #############################

input_width = 30
OUT_STEPS = 10
    
multi_window = WindowGenerator(input_width=input_width,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=['AO'])

multi_window
#multi_window.plot()

train_mae_AUX = []
val_mae_AUX = []
test_mae_AUX = []

linear_box = []
dense_box = []
conv_box = []
lstm_box = []
rnn_box = []
rnnc_box = []

n_iter = 10
for i in range(n_iter):
    
    multi_train_performance = {}
    multi_val_performance = {}
    multi_performance = {}

    # LINEAL

    multi_linear_model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history_Linear = compile_and_fit(multi_linear_model, multi_window)
    multi_train_performance['Linear'] = history_Linear.history['root_mean_squared_error'][-1]

    IPython.display.clear_output()
    multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
    multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
    #multi_window.plot(multi_linear_model)


    # DENSO

    multi_dense_model = tf.keras.Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(32,activation='relu'),
        keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history_Dense = compile_and_fit(multi_dense_model, multi_window)
    multi_train_performance['Dense'] = history_Dense.history['root_mean_squared_error'][-1]

    IPython.display.clear_output()
    multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
    multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
    #multi_window.plot(multi_dense_model)


    # CONVOLUCIONAL (MUCHO OVERFITTING) 

    CONV_WIDTH = 3
    multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(128, activation='relu', kernel_size=(CONV_WIDTH)), 
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history_Conv = compile_and_fit(multi_conv_model, multi_window,epochs=500)
    multi_train_performance['Conv'] = history_Conv.history['root_mean_squared_error'][-1]

    IPython.display.clear_output()

    multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
    multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
    #multi_window.plot(multi_conv_model)


    # LSTM

    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(128, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history_LSTM = compile_and_fit(multi_lstm_model, multi_window,epochs=500)
    multi_train_performance['LSTM'] = history_LSTM.history['root_mean_squared_error'][-1]

    IPython.display.clear_output()

    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    #multi_window.plot(multi_lstm_model)


    # RNN SIMPLE

    modelo_RNN = keras.models.Sequential([
        #tf.keras.layers.Input(shape=(num_features,)),
        #tf.keras.layers.SimpleRNN(128,return_sequences=True,activation='relu'),
        tf.keras.layers.SimpleRNN(128,activation='relu'),
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                              kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ])

    history_RNN = compile_and_fit(modelo_RNN, multi_window,epochs=500)
    multi_train_performance['RNN'] = history_RNN.history['root_mean_squared_error'][-1]

    multi_val_performance['RNN'] = modelo_RNN.evaluate(multi_window.val)
    multi_performance['RNN'] = modelo_RNN.evaluate(multi_window.test, verbose=0)
    #multi_window.plot(modelo_RNN)


    # RNN COMPLEJA

    modelo_RNNC = keras.models.Sequential([
        #tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.SimpleRNN(128,return_sequences=True,activation='relu'),
        tf.keras.layers.SimpleRNN(64,return_sequences=True,activation='relu'),
        tf.keras.layers.SimpleRNN(32,activation='relu'),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ])

    history_RNNC = compile_and_fit(modelo_RNNC, multi_window)
    multi_train_performance['RNN_C'] = history_RNNC.history['root_mean_squared_error'][-1]

    multi_val_performance['RNN_C'] = modelo_RNNC.evaluate(multi_window.val)
    multi_performance['RNN_C'] = modelo_RNNC.evaluate(multi_window.test, verbose=0)
    #multi_window.plot(modelo_RNNC)

    train_mae = list(multi_train_performance.values())
    val_mae = [v[1] for v in multi_val_performance.values()]
    test_mae = [v[1] for v in multi_performance.values()]
    
    if train_mae_AUX:
        train_mae_AUX = [x + y for x, y in zip(train_mae_AUX, train_mae)]
        val_mae_AUX = [x + y for x, y in zip(val_mae_AUX, val_mae)]
        test_mae_AUX = [x + y for x, y in zip(test_mae_AUX, test_mae)]
    else:
        train_mae_AUX = train_mae
        val_mae_AUX = val_mae
        test_mae_AUX = test_mae
    
    linear_box.append(test_mae[0])
    dense_box.append(test_mae[1])
    conv_box.append(test_mae[2])
    lstm_box.append(test_mae[3])
    rnn_box.append(test_mae[4])
    rnnc_box.append(test_mae[5])


train_mae_DEF = [x / n_iter for x in train_mae_AUX]
val_mae_DEF = [x / n_iter for x in val_mae_AUX]
test_mae_DEF = [x / n_iter for x in test_mae_AUX]

############################  BOXPLOT  ################################################

boxplot_data = [linear_box,dense_box,conv_box,lstm_box,rnn_box,rnnc_box]
fig, ax = plt.subplots()
ax.boxplot(boxplot_data)
nombres = ['Linear', 'Dense', 'Conv', 'LSTM', 'RNN', 'RNN_C']
ax.set_xticklabels(nombres,fontsize=13, rotation=45)
ax.set_ylabel('RMSE')
plt.show()


########################  DIAGRAMA DE BARRAS  #########################################

x = np.arange(len(multi_performance)) 
width = 0.15
plt.figure()
plt.title(f'Average Performance Time Window {input_width} - {OUT_STEPS}')
plt.bar(x - 0.17, train_mae_DEF, width, label='train')
plt.bar(x , val_mae_DEF, width, label='Validation')
plt.bar(x + 0.17, test_mae_DEF, width, label='Test')
plt.axhline(y=test_mae_DEF[0], color='b', linestyle='--')
plt.xticks(ticks=x, labels=multi_performance.keys(), fontsize = 14,
           rotation=45)
plt.ylabel(f'RMSE (average over all times and outputs)')
_ = plt.legend()
plt.show()

nombre_archivo = f'resultados_VM_{input_width} - {OUT_STEPS}_128_neurons_RMSE.csv'

with open(nombre_archivo, 'w', newline='') as archivo_csv:
    writer = csv.writer(archivo_csv)
    writer.writerow(['linear_box', 'dense_box', 'conv_box', 'lstm_box','rnn_box','rnnc_box','train_mae_DEF','val_mae_DEF','test_mae_DEF'])  # Escribir encabezados
    writer.writerows(zip(linear_box,dense_box,conv_box,lstm_box,rnn_box,rnnc_box,train_mae_DEF,val_mae_DEF,test_mae_DEF))  # Escribir filas de datos



