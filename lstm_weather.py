# -*- coding: utf-8 -*-
"""LSTM_Weather.ipynb
After recording traffic count data I converted it into a traffic terms of flowrate and density.
The next step was to consolidate Weather, Calendar, Special Event and Traffic data togeather and predict traffic for the next hour using LSTM.
The MTL-LSTM model predicts traffic flowrate and density parallely.
"""

from math import sqrt
import numpy as np
import tensorflow as tf
import keras
import copy
import csv
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, TimeDistributed, Reshape, Input, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import concatenate
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

from numpy import cov

tf.random.set_seed(1234)

# specify the number of lag hours
look_back = 7
features = 19
hours_of_day = 12
no_of_examples = 0
checkpoint_path = './Fl_Dn_lstm_cp.h5'

for i in range(0,13):
  print(i)

plt.scatter(dataset_values[0],dataset_values[11])
plt.show()

_file = open('dataset.csv','a')
_writer = csv.writer(_file, delimiter=',',lineterminator='\n')

for i_row in dataset_values:
  _writer.writerow(i_row)

_file.close()

covariance = cov(dataset_values[0],dataset_values[1])
print(covariance)

def series_to_supervised(data, look_back = 1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    print('look_back :',look_back)

    # input sequence (t-n, ... t-1)
    for i in range(look_back, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg

dataset = read_csv('Weather_Traffic_3.csv', header=0, index_col=False)
#dataset = read_csv('All_Forecasted_Data.csv', header=0, index_col=False)

dataset_values = dataset.values

#----------------Remove less correlated parameters--------------------------
removed_features = [3,4,6,8,9,10,11,13]
# removed_features = []
dataset_values = np.delete(dataset_values, removed_features, axis=1)
# dataset_values = np.delete(dataset_values, [0,1,2,8,9,10,11], axis=1)

Flowrate_column = features - len(removed_features) - 1 - 1
Density_column =  features - len(removed_features) - 1

# ensure all data is float
dataset_values = dataset_values.astype('float32')

# normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset_values)
#--------------------------------------------------------------------

# frame as supervised learning
reframed = series_to_supervised(scaled_data, look_back * hours_of_day , 1)

print(reframed.head())
values = reframed.values

Y_ordered = copy.deepcopy(values[:,-2:])
# values[:,-2:] = 0
# X_ordered = values                # With weather data of hour to predict traffic
X_ordered = values[:,0:924]      # Without weather data of hour to predict traffic - 7 days look back
# X_ordered = values[:,0:1848]      # Without weather data of hour to predict traffic - 14 days look back

print(X_ordered.shape)

# X_ordered = X_ordered.reshape((X_ordered.shape[0], 1, X_ordered.shape[1]))      # With weather data of hour to predict traffic
X_ordered = X_ordered.reshape((X_ordered.shape[0], look_back, hours_of_day * 11))    # Without weather data of hour to predict traffic

# _file = open('dataset.csv','a')
# _writer = csv.writer(_file, delimiter=',',lineterminator='\n')

# for i_row in X_ordered:
#   _writer.writerow([*i_row[0],*i_row[1],*i_row[2],*i_row[3],*i_row[4],*i_row[5],*i_row[6]])
# _file.close()

Y_ordered_Fl = Y_ordered[:,0]
Y_ordered_Dn = Y_ordered[:,1]

# X_train, X_test, Y_Fl_train, Y_Fl_test, Y_Dn_train, Y_Dn_test = train_test_split (X_ordered, Y_ordered_Fl, Y_ordered_Dn, train_size=492, shuffle = False)
# X_train, X_val, Y_Fl_train, Y_Fl_val, Y_Dn_train, Y_Dn_val = train_test_split (X_train, Y_Fl_train, Y_Dn_train, test_size=0.20, shuffle = False)

X_train, X_test, Y_Fl_train, Y_Fl_test, Y_Dn_train, Y_Dn_test = train_test_split (X_ordered, Y_ordered_Fl, Y_ordered_Dn, test_size=0.180, shuffle = False)
X_train, X_val, Y_Fl_train, Y_Fl_val, Y_Dn_train, Y_Dn_val = train_test_split (X_train, Y_Fl_train, Y_Dn_train, test_size=0.23, shuffle = False)

# X_train, X_val, Y_Fl_train, Y_Fl_val, Y_Dn_train, Y_Dn_val = train_test_split (X_ordered, Y_ordered_Fl, Y_ordered_Dn, test_size=0.20, shuffle = False)
# X_train, X_test, Y_Fl_train, Y_Fl_test, Y_Dn_train, Y_Dn_test = train_test_split (X_train, Y_Fl_train, Y_Dn_train, test_size=0.25, shuffle = False)

# X_train, Y_Fl_train, Y_Dn_train = X_ordered, Y_ordered_Fl, Y_ordered_Dn
# X_test, Y_Fl_test, Y_Dn_test = X_train, Y_Fl_train, Y_Dn_train
# X_val, Y_Fl_val, Y_Dn_val = X_train, Y_Fl_train, Y_Dn_train
print(X_train.shape)

x_input_shape = X_train.shape[1], X_train.shape[2]

print("No of Training data: ",len(X_train))
print("No of Test data: ",len(X_test))
print("No of Validation data: ",len(X_val))

print(X_train.shape, Y_Fl_train.shape, X_test.shape, Y_Fl_test.shape, X_val.shape, Y_Dn_val.shape)
print(Y_Dn_train.shape, Y_Dn_test.shape, Y_Dn_val.shape)

print("Flowrate_column: ",Flowrate_column)
print("Density_column: ",Density_column)
scale_Fl = MinMaxScaler()
scale_Dn = MinMaxScaler()
scale_Fl.min_, scale_Fl.scale_ = scaler.min_[Flowrate_column], scaler.scale_[Flowrate_column]
scaled_inv = Y_Fl_test.reshape(len(Y_Fl_test),1)
Y_Fl_test_org = scale_Fl.inverse_transform(scaled_inv)
# print(Y_Fl_test_org)
scale_Dn.min_, scale_Dn.scale_ = scaler.min_[Density_column], scaler.scale_[Density_column]
scaled_inv = Y_Dn_test.reshape(len(Y_Dn_test),1)
Y_Dn_test_org = scale_Dn.inverse_transform(scaled_inv)
# print(Y_Dn_test_org)

print(sum(Y_Fl_test) / len(Y_Fl_test))
dataset = dataset.values

epochs = 1000
batch_size = 25
lr_rate = 0.001
momentum = 0.8

input = Input(shape=x_input_shape)
print(input.shape)
# x = TimeDistributed (Conv2D(64, kernel_size=(3, 3), activation='relu'), input_shape = x_input_shape)(image)
# x = TimeDistributed (MaxPooling2D(pool_size=(2, 2)))(x)
# x = TimeDistributed (Conv2D(64, (3, 3), activation='relu'))(x)

x = LSTM(units = 20, activation = 'sigmoid', recurrent_activation = 'sigmoid', kernel_initializer='glorot_uniform', 
         return_sequences = True)(input)
# x = LSTM(units = 2, activation = 'sigmoid', recurrent_activation = 'sigmoid', kernel_initializer='glorot_uniform', 
        #  return_sequences = False)(x)

x = Dropout(0.2)(x)
x = Flatten()(x)
# x = Dense(20, activation='relu')(x)
# x = Dense(5, activation='sigmoid')(x)
fLowrate_output = Dense(1, activation='linear', name="flowrate")(x)
# x = Dense(10, activation='relu')(x)
density_output = Dense(1, activation='linear', name="density")(x)

model = Model(inputs = input, outputs = [fLowrate_output, density_output])
# model = Model(inputs = input, outputs = [fLowrate_output])
model.summary()

SGD = tf.keras.optimizers.SGD(lr=lr_rate, momentum=momentum)
adam = tf.keras.optimizers.Adam(lr=lr_rate, beta_1=0.60, beta_2=0.90)
Adadelta = tf.keras.optimizers.Adadelta(lr=lr_rate, rho=0.85)
Nadam = tf.keras.optimizers.Nadam(learning_rate=lr_rate, beta_1=0.9, beta_2=0.999)
Adamax = tf.keras.optimizers.Adamax(learning_rate=lr_rate, beta_1=0.90, beta_2=0.99, epsilon=1e-07, name="Adamax")
Adagrad = tf.keras.optimizers.Adagrad(learning_rate=lr_rate, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")

model.compile(loss=keras.losses.mean_squared_error,
          # optimizer="Adadelta",
          optimizer="Adam",
          # optimizer="Adamax",
            metrics=['mae'])

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto', period=1)

history = model.fit(X_train, [Y_Fl_train, Y_Dn_train],
# history = model.fit(X_train, [Y_Fl_train],
          batch_size = batch_size,
          epochs = epochs,
          verbose = 0,
          validation_data = (X_val, [Y_Fl_val, Y_Dn_val]),
          # validation_data = (X_val, [Y_Fl_val]),
          callbacks=[checkpoint])

# plot history
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

model.save('Fl_Dn_lstm_cp.h5')

# model = load_model('Fl_Dn_lstm_cp_0.13.h5')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['flowrate_loss'], label='flowrate_loss')
plt.plot(history.history['density_loss'], label='density_loss')
plt.plot(history.history['val_flowrate_loss'], label='val_flowrate_loss')
plt.plot(history.history['val_density_loss'], label='val_density_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# model.load_weights(checkpoint_path)
score = model.evaluate(X_test, [Y_Fl_test, Y_Dn_test], verbose=0)

predictions = model.predict(X_test)

# Invert scaling of predicted
scaled_inv = predictions[0].reshape(len(predictions[0]),1) #Predicted traffic flowrate value
Y_Fl_Predicted = scale_Fl.inverse_transform(scaled_inv)

scaled_inv = predictions[1].reshape(len(predictions[1]),1) #Predicted traffic density value
Y_Dn_Predicted = scale_Dn.inverse_transform(scaled_inv)

# Model Score
print(model.metrics_names)
print('Test loss:', score[0])
print('Test Flowrate loss:', score[1])
print('Test Density loss:', score[2])
print('Test Flowrate MAE:', mean_absolute_error(Y_Fl_Predicted, Y_Fl_test_org))
print('Test Density MAE:' ,mean_absolute_error(Y_Dn_Predicted, Y_Dn_test_org))
Fl_avg = sum(Y_Fl_test_org) / len(Y_Fl_test_org)
Dn_avg = sum(Y_Dn_test_org) / len(Y_Dn_test_org)
print('Test Flowrate MAPE:', *mean_absolute_error(Y_Fl_Predicted, Y_Fl_test_org)/Fl_avg)
print('Test Density MAPE:', *mean_absolute_error(Y_Dn_Predicted, Y_Dn_test_org)/Dn_avg)
print('Test Flowrate MAPE:', np.mean(np.abs((Y_Fl_test_org - Y_Fl_Predicted) / Y_Fl_test_org)) * 100)
print('Test Density MAPE:', np.mean(np.abs((Y_Dn_test_org - Y_Dn_Predicted) / Y_Dn_test_org)) * 100)

print(sum(Y_Fl_test), len(Y_Fl_test))
print(sum(Y_Dn_test), len(Y_Dn_test))

# calculate RMSE
rmse = sqrt(mean_absolute_error(predictions[0], Y_Fl_test_org))

# model = load_model('Fl_Dn_lstm_cp_0.31.h5')

plt.plot(Y_Fl_test_org, label='Actual')
plt.plot(Y_Fl_Predicted, label='Predicted')
plt.title('Traffic Flowrate')
plt.xlabel('No of Examples')
plt.ylabel('Hourly Flowrate')
plt.legend()
plt.show()

plt.plot(Y_Dn_test_org, label='Actual')
plt.plot(Y_Dn_Predicted, label='Predicted')
plt.title('Traffic Density')
plt.xlabel('No of Examples')
plt.ylabel('Hourly Density')
plt.legend()
plt.show()

print("Actual_Flowrate Predicted_Flowrate Actual_Density Predicted_Density,")
for i in range(Y_Fl_Predicted.__len__()):
  print(*Y_Fl_test_org[i]," ",*Y_Fl_Predicted[i]," ",*Y_Dn_test_org[i]," ",*Y_Dn_Predicted[i],",")
  # print(Y_Fl_test[i],",",predictions[0][i],",",Y_Dn_test[i],",",predictions[1][i])
# for i in range(Y_Fl_Predicted.__len__()):
#   print(*Y_Fl_test_org[i]," ",*Y_Fl_Predicted[i],",")

predictions = model.predict(X_ordered)

# Invert scaling of predicted

Y_ordered_Fl = Y_ordered_Fl.reshape(len(Y_ordered_Fl),1)
Y_ordered_Fl = scale_Fl.inverse_transform(Y_ordered_Fl)

Y_ordered_Fl_Pred = predictions[0].reshape(len(predictions[0]),1)
Y_ordered_Fl_Pred = scale_Fl.inverse_transform(Y_ordered_Fl_Pred)

Y_ordered_Dn = Y_ordered_Dn.reshape(len(Y_ordered_Dn),1)
Y_ordered_Dn = scale_Dn.inverse_transform(Y_ordered_Dn)

Y_ordered_Dn_Pred = predictions[1].reshape(len(predictions[1]),1)
Y_ordered_Dn_Pred = scale_Dn.inverse_transform(Y_ordered_Dn_Pred)

plt.plot(Y_ordered_Fl, label='Actual')
plt.plot(Y_ordered_Fl_Pred, label='Predicted')
plt.title('Traffic Flowrate')
plt.xlabel('No of Examples')
plt.ylabel('Hourly Flowrate')
plt.legend()
plt.show()

plt.plot(Y_ordered_Dn, label='Actual')
plt.plot(Y_ordered_Dn_Pred, label='Predicted')
plt.title('Traffic Density')
plt.xlabel('No of Examples')
plt.ylabel('Hourly Density')
plt.legend()
plt.show()

for i in range(Y_ordered_Fl.__len__()):
  print(*Y_ordered_Fl[i]," ",*Y_ordered_Fl_Pred[i]," ",*Y_ordered_Dn[i]," ",*Y_ordered_Dn_Pred[i],",")

dataset = read_csv('All_Forecasted_Data.csv', header=0, index_col=False)

dataset_values = dataset.values

#----------------Remove less correlated parameters--------------------------
removed_features = [3,4,6,8,9,10,11,13]
# removed_features = []
dataset_values = np.delete(dataset_values, removed_features, axis=1)
# dataset_values = np.delete(dataset_values, [0,1,2,8,9,10,11], axis=1)

Flowrate_column = features - len(removed_features) - 1 - 1
Density_column =  features - len(removed_features) - 1

# ensure all data is float
dataset_values = dataset_values.astype('float32')

# normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset_values)
#--------------------------------------------------------------------

# frame as supervised learning
reframed = series_to_supervised(scaled_data, look_back * hours_of_day , 1)

print(reframed.head())
values = reframed.values

Y_ordered = copy.deepcopy(values[:,-2:])
# values[:,-2:] = 0
# X_ordered = values                # With weather data of hour to predict traffic
X_ordered = values[:,0:924]      # Without weather data of hour to predict traffic - 7 days look back
# X_ordered = values[:,0:1848]      # Without weather data of hour to predict traffic - 14 days look back

print(X_ordered.shape)

# X_ordered = X_ordered.reshape((X_ordered.shape[0], 1, X_ordered.shape[1]))      # With weather data of hour to predict traffic
X_ordered = X_ordered.reshape((X_ordered.shape[0], look_back, hours_of_day * 11))    # Without weather data of hour to predict traffic

# _file = open('dataset.csv','a')
# _writer = csv.writer(_file, delimiter=',',lineterminator='\n')

# for i_row in X_ordered:
#   _writer.writerow([*i_row[0],*i_row[1],*i_row[2],*i_row[3],*i_row[4],*i_row[5],*i_row[6]])
# _file.close()

Y_ordered_Fl = Y_ordered[:,0]
Y_ordered_Dn = Y_ordered[:,1]

X_train, X_test, Y_Fl_train, Y_Fl_test, Y_Dn_train, Y_Dn_test = train_test_split (X_ordered, Y_ordered_Fl, Y_ordered_Dn, test_size=0.180, shuffle = False)
X_train, X_val, Y_Fl_train, Y_Fl_val, Y_Dn_train, Y_Dn_val = train_test_split (X_train, Y_Fl_train, Y_Dn_train, test_size=0.23, shuffle = False)

# X_ordered = X_ordered.reshape((X_ordered.shape[0], look_back, X_ordered.shape[1]))      # With weather data of hour to predict traffic
# X_ordered = X_ordered.reshape((X_ordered.shape[0], seq_steps, hours_of_day))    # Without weather data of hour to predict traffic

print(X_train.shape)

x_input_shape = X_train.shape[1], X_train.shape[2]

print("No of Training data: ",len(X_train))
print("No of Test data: ",len(X_test))
print("No of Validation data: ",len(X_val))

print(X_train.shape, Y_Fl_train.shape, X_test.shape, Y_Fl_test.shape, X_val.shape, Y_Dn_val.shape)
print(Y_Dn_train.shape, Y_Dn_test.shape, Y_Dn_val.shape)

print("Flowrate_column: ",Flowrate_column)
print("Density_column: ",Density_column)
scale_Fl = MinMaxScaler()
scale_Dn = MinMaxScaler()
scale_Fl.min_, scale_Fl.scale_ = scaler.min_[Flowrate_column], scaler.scale_[Flowrate_column]
scaled_inv = Y_Fl_test.reshape(len(Y_Fl_test),1)
Y_Fl_test_org = scale_Fl.inverse_transform(scaled_inv)
# print(Y_Fl_test_org)
scale_Dn.min_, scale_Dn.scale_ = scaler.min_[Density_column], scaler.scale_[Density_column]
scaled_inv = Y_Dn_test.reshape(len(Y_Dn_test),1)
Y_Dn_test_org = scale_Dn.inverse_transform(scaled_inv)
# print(Y_Dn_test_org)

print(sum(Y_Fl_test) / len(Y_Fl_test))
dataset = dataset.values

predictions = model.predict(X_test)

# Invert scaling of predicted
Y_Fl_Predicted = predictions[0].reshape(len(predictions[0]),1)
Y_Fl_Predicted = scale_Fl.inverse_transform(Y_Fl_Predicted)

Y_Dn_Predicted = predictions[1].reshape(len(predictions[1]),1)
Y_Dn_Predicted = scale_Dn.inverse_transform(Y_Dn_Predicted)

scaled_inv = Y_Fl_test.reshape(len(Y_Fl_test),1)
Y_Fl_org = scale_Fl.inverse_transform(scaled_inv)
# print(Y_Fl_test_org)

scaled_inv = Y_Dn_test.reshape(len(Y_Dn_test),1)
Y_Dn_org = scale_Dn.inverse_transform(scaled_inv)

print(X_ordered.__len__())

_file = open('Prediction.csv','a')
_writer = csv.writer(_file, delimiter=',',lineterminator='\n')

for i in range(X_test.__len__()):
  print(*Y_Fl_org[i]," ",*Y_Fl_Predicted[i]," ",*Y_Dn_org[i]," ",*Y_Dn_Predicted[i],",")
  _writer.writerow([*Y_Fl_org[i],",",*Y_Fl_Predicted[i],",",*Y_Dn_org[i],",",*Y_Dn_Predicted[i]])

_file.close()

print('Test Flowrate MAE:', mean_absolute_error(Y_Fl_Predicted, Y_Fl_org))
print('Test Density MAE:' ,mean_absolute_error(Y_Dn_Predicted, Y_Dn_org))
Fl_avg = sum(Y_Fl_org) / len(Y_Fl_org)
Dn_avg = sum(Y_Dn_org) / len(Y_Dn_org)
print('Test Flowrate MAPE:', *mean_absolute_error(Y_Fl_Predicted, Y_Fl_org)/Fl_avg)
print('Test Density MAPE:', *mean_absolute_error(Y_Dn_Predicted, Y_Dn_org)/Dn_avg)
