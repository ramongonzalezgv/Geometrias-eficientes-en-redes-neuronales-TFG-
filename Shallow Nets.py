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
from keras.callbacks import LearningRateScheduler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse
import tensorflow as tf
from tensorflow import keras


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
metrics=[tf.metrics.MeanSquaredError()]
epocas = 100
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error',
                                                    patience=6,
                                                    mode='min')


# DECAIMIENTO DE LA TASA DE APRENDIZAJE

def decay_lr(epoch, learning_rate):
    decay_rate = 0.1
    decay_step = 5
    if epoch % decay_step == 0 and epoch != 0:
        return learning_rate * decay_rate
    return learning_rate

lr_decay_callback = LearningRateScheduler(decay_lr)

# MODELO LINEAL

Lineal_Model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(16,))
])

Lineal_Model.compile(loss=tf.losses.MeanAbsoluteError(),
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.metrics.MeanAbsoluteError()])

Lineal_Model_history = Lineal_Model.fit(X_train,Y_train,epochs=epocas,verbose=1,
                      validation_data=(X_valid,Y_valid))


performance_train['Lineal'] = Lineal_Model_history.history['mean_absolute_error'][-1]
performance_val['Lineal'] = Lineal_Model_history.history['val_mean_absolute_error'][-1]

performance_test['Lineal'] = Lineal_Model.evaluate(X_test,Y_test, batch_size=100)

# MODELO SIGMOIDE

ANN_Model_Sigmoid_sh = keras.Sequential([
    keras.layers.InputLayer(input_shape=(16,)),# input layer (1)
    keras.layers.Dense(4, activation='sigmoid'),
    keras.layers.Dense(1) # output layer (3)
])

ANN_Model_Sigmoid_sh.compile(loss=tf.losses.MeanAbsoluteError(),
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.metrics.MeanAbsoluteError()])

ANN_Model_Sigmoid_sh_history = ANN_Model_Sigmoid_sh.fit(X_train,Y_train,epochs=epocas,verbose=1,
                      validation_data=(X_valid,Y_valid))

Sigmoid_sh_loss_values = ANN_Model_Sigmoid_sh_history.history['loss']
Sigmoid_sh_epoch_values = range(1, len(Sigmoid_sh_loss_values) + 1)


performance_train['sigmoid'] = ANN_Model_Sigmoid_sh_history.history['mean_absolute_error'][-1]
performance_val['sigmoid'] = ANN_Model_Sigmoid_sh_history.history['val_mean_absolute_error'][-1]

performance_test['sigmoid'] = ANN_Model_Sigmoid_sh.evaluate(X_test,Y_test, batch_size=100)


# MODELO TANH

ANN_Model_tanh_sh = keras.Sequential([
    keras.layers.InputLayer(input_shape=(16,)),# input layer (1)
    keras.layers.Dense(4, activation='tanh'),
    keras.layers.Dense(1) # output layer (3)
])

ANN_Model_tanh_sh.compile(loss=tf.losses.MeanAbsoluteError(),
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.metrics.MeanAbsoluteError()])

ANN_Model_tanh_sh_history = ANN_Model_tanh_sh.fit(X_train,Y_train,epochs=epocas,verbose=1,
                      validation_data=(X_valid,Y_valid))

tanh_sh_loss_values = ANN_Model_tanh_sh_history.history['loss']
tanh_sh_epoch_values = range(1, len(tanh_sh_loss_values) + 1)


performance_train['tanh'] = ANN_Model_tanh_sh_history.history['mean_absolute_error'][-1]
performance_val['tanh'] = ANN_Model_tanh_sh_history.history['val_mean_absolute_error'][-1]

performance_test['tanh'] = ANN_Model_tanh_sh.evaluate(X_test,Y_test, batch_size=100)


# MODELO RELU

ANN_Model_relu_sh = keras.Sequential([
    keras.layers.InputLayer(input_shape=(16,)),# input layer (1)
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1) # output layer (3)
])

ANN_Model_relu_sh.compile(loss=tf.losses.MeanAbsoluteError(),
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.metrics.MeanAbsoluteError()])

ANN_Model_relu_sh_history = ANN_Model_relu_sh.fit(X_train,Y_train,epochs=epocas,verbose=1,
                      validation_data=(X_valid,Y_valid))

relu_sh_loss_values = ANN_Model_relu_sh_history.history['loss']
relu_sh_epoch_values = range(1, len(relu_sh_loss_values) + 1)

performance_train['relu'] = ANN_Model_relu_sh_history.history['mean_absolute_error'][-1]
performance_val['relu'] = ANN_Model_relu_sh_history.history['val_mean_absolute_error'][-1]

performance_test['relu'] = ANN_Model_relu_sh.evaluate(X_test,Y_test, batch_size=100)

#GRAFICAS DE LOSS VS EPOCHS

plt.figure()
plt.plot(Sigmoid_sh_epoch_values, Sigmoid_sh_loss_values, 'r-')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.title(f'Loss vs Epochs during training: sigmoid, lr = {learning_rate}')

plt.figure()
plt.plot(tanh_sh_epoch_values, tanh_sh_loss_values, 'r-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Loss vs Epochs during training: tanh, lr = {learning_rate}')

plt.figure()
plt.plot(relu_sh_epoch_values, relu_sh_loss_values, 'r-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Loss vs Epochs during training: relu, lr = {learning_rate}')

# GRAFICAS DE ERROR
x = np.arange(len(performance_test))
width = 0.15

train_mae = list(performance_train.values())
val_mae = list(performance_val.values())
test_mae = [v[1] for v in performance_test.values()]

plt.figure()
plt.title(f'Shallow nets performance')
plt.bar(x - 0.17, train_mae, width, label='train')
plt.bar(x , val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.axhline(y=performance_test['Lineal'][1], color='b', linestyle='--')
plt.xticks(ticks=x, labels=performance_test.keys(),
           rotation=45)
plt.ylabel(f'Mean Absolute Error')
_ = plt.legend()
plt.show()