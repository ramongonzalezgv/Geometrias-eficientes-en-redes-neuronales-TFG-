# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np
import random

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse
import tensorflow as tf
from tensorflow import keras
from keras import backend as K


#################################  DATOS  ##################################################################

df = pd.DataFrame(pd.read_csv("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Datos_teleconexion_vs\\df_tf_nodate.csv"))

# Separamos en train, test y validación
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train = df[0:int(n*0.7)]
val= df[int(n*0.7):int(n*0.9)]
test= df[int(n*0.9):]


################################  MODELOS  ##############################################3

X_train = np.array(train.drop(['AO'],axis='columns')); Y_train = np.array(train['AO'])
X_valid = np.array(val.drop(['AO'],axis='columns')); Y_valid = np.array(val['AO'])
X_test = np.array(test.drop(['AO'],axis='columns')); Y_test = np.array(test['AO'])

performance_test = {}
performance_val = {}
performance_train = {}

# HIPERPARAMETROS

metric_name = 'mean_absolute_error' 
loss=tf.losses.MeanAbsoluteError()
learning_rate = 0.01
opti = tf.optimizers.SGD(learning_rate=learning_rate)
metrics=[tf.metrics.MeanAbsoluteError()]
epocas = 100

# ROOT MEAN SQUARED ERROR

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# MODELO LINEAL

Lineal_Model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(16,))
])

Lineal_Model.compile(loss=rmse,
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.metrics.RootMeanSquaredError()])

Lineal_Model_history = Lineal_Model.fit(X_train,Y_train,epochs=100,verbose=1,
                      validation_data=(X_valid,Y_valid))


performance_train['Lineal'] = Lineal_Model_history.history['root_mean_squared_error'][-1]
performance_val['Lineal'] = Lineal_Model_history.history['val_root_mean_squared_error'][-1]

performance_test['Lineal'] = Lineal_Model.evaluate(X_test,Y_test, batch_size=100)

# MODELO SIGMOIDE DENSO

ANN_Model_sigmoid = keras.Sequential([
    keras.layers.InputLayer(input_shape=(16,)),# input layer (1)
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(60,activation='sigmoid'),
    keras.layers.Dense(30,activation='sigmoid'),
    keras.layers.Dense(30,activation='sigmoid'),
    keras.layers.Dense(1) # output layer (3)
])

ANN_Model_sigmoid.compile(loss=rmse,
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.metrics.RootMeanSquaredError()])

ANN_Model_Sigmoid_history = ANN_Model_sigmoid.fit(X_train,Y_train,epochs=100,verbose=1,
                      validation_data=(X_valid,Y_valid))

sigmoid_loss_values = ANN_Model_Sigmoid_history.history['loss']
sigmoid_epoch_values = range(1, len(sigmoid_loss_values) + 1)


performance_train['sigmoid deep'] = ANN_Model_Sigmoid_history.history['root_mean_squared_error'][-1]
performance_val['sigmoid deep'] = ANN_Model_Sigmoid_history.history['val_root_mean_squared_error'][-1]

performance_test['sigmoid deep'] = ANN_Model_sigmoid.evaluate(X_test,Y_test, batch_size=100)


# MODELO TANH DENSO

ANN_Model_tanh = keras.Sequential([
    keras.layers.InputLayer(input_shape=(16,)),# input layer (1)
    keras.layers.Dense(100, activation='tanh'),
    keras.layers.Dense(60,activation='tanh'),
    keras.layers.Dense(30,activation='tanh'),
    keras.layers.Dense(30,activation='tanh'),
    keras.layers.Dense(1) # output layer (3)
])

ANN_Model_tanh.compile(loss=rmse,
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.metrics.RootMeanSquaredError()])

ANN_Model_tanh_history = ANN_Model_tanh.fit(X_train,Y_train,epochs=100,verbose=1,
                      validation_data=(X_valid,Y_valid))

tanh_loss_values = ANN_Model_tanh_history.history['loss']
tanh_epoch_values = range(1, len(tanh_loss_values) + 1)


performance_train['tanh deep'] = ANN_Model_tanh_history.history['root_mean_squared_error'][-1]
performance_val['tanh deep'] = ANN_Model_tanh_history.history['val_root_mean_squared_error'][-1]

performance_test['tanh deep'] = ANN_Model_tanh.evaluate(X_test,Y_test, batch_size=100)


# MODELO RELU DENSO

ANN_Model_relu = keras.Sequential([
    keras.layers.InputLayer(input_shape=(16,)),# input layer (1)
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(60,activation='relu'),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dense(1) # output layer (3)
])

ANN_Model_relu.compile(loss=rmse,
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.metrics.RootMeanSquaredError()])

ANN_Model_relu_history = ANN_Model_relu.fit(X_train,Y_train,epochs=100,verbose=1,
                      validation_data=(X_valid,Y_valid))

relu_loss_values = ANN_Model_relu_history.history['loss']
relu_epoch_values = range(1, len(relu_loss_values) + 1)


performance_train['relu deep'] = ANN_Model_relu_history.history['root_mean_squared_error'][-1]
performance_val['relu deep'] = ANN_Model_relu_history.history['val_root_mean_squared_error'][-1]

performance_test['relu deep'] = ANN_Model_relu.evaluate(X_test,Y_test, batch_size=100)

#GRAFICAS DE LOSS VS EPOCHS

plt.figure()
plt.plot(sigmoid_epoch_values, sigmoid_loss_values, 'r-')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSE)')
plt.ylim(0,max(sigmoid_loss_values))
plt.title(f'Loss vs Epochs during training: sigmoid, lr = {learning_rate}')

plt.figure()
plt.plot(tanh_epoch_values, tanh_loss_values, 'r-')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSE)')
plt.ylim(0,max(sigmoid_loss_values))
plt.title(f'Loss vs Epochs during training: tanh, lr = {learning_rate}')

plt.figure()
plt.plot(relu_epoch_values, relu_loss_values, 'r-')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSE)')
plt.ylim(0,max(sigmoid_loss_values))
plt.title(f'Loss vs Epochs during training: relu, lr = {learning_rate}')

# GRAFICAS DE ERROR
x = np.arange(len(performance_test))
width = 0.15

train_mae = list(performance_train.values())
val_mae = list(performance_val.values())
test_mae = [v[1] for v in performance_test.values()]

plt.figure()
plt.title(f'Deep nets performance ')
plt.bar(x - 0.17, train_mae, width, label='train')
plt.bar(x , val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.axhline(y=performance_test['Lineal'][1], color='b', linestyle='--')
plt.xticks(ticks=x, labels=performance_test.keys(),
           rotation=45)
plt.ylabel(f'Mean Absolute Error')
_ = plt.legend()
plt.show()