# -*- coding: utf-8 -*-
"""VGG_CNN.ipynb
Model accepts two images (t-1 and t) at a time to predict number of new vehicles in the given frame (t).
The previous frame (t-1) is passed as a context to the current frame (t) to predict the number of new vehicles.
The model also accepts time differencein milliseconds since the last frame.
"""

#region  Import Libraries
from __future__ import print_function
import keras
import os
import cv2
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Reshape, concatenate, PReLU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix,classification_report
from keras.layers.normalization import BatchNormalization
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

random_seed = 2
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

IMG_HEIGHT, IMG_WIDTH  = 123, 277
CHANNELS=3
num_classes = 6
seq_steps = 2
input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
checkpoint_path = './cnn_cp.ckpt'

initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

!unzip Data.zip

DATASET_PATH ='FG 1.0'

def load_model_data(data_file):

    if os.path.exists(DATASET_PATH +'/'+ data_file):
        with open(DATASET_PATH +'/'+ data_file, newline='') as csvfile:
            labelsfile = list(csv.reader(csvfile))
    else:
        labelsfile = [[]]

    data_file = labelsfile          # For .csv file

    X_ordered = np.empty((len(data_file), seq_steps, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=float)
    time_diff_ordered = np.empty((len(data_file), seq_steps, 1), dtype=float)
    labels = np.empty((len(data_file), 1), dtype=float)
    images_lst = list()
    i = 0

    for i_row in data_file:
        # print(i_row[0],i_row[1])
        if i_row[0].endswith('.jpg') and i_row[1].endswith('.jpg'):
            
            my_file = Path(DATASET_PATH +'/'+ i_row[0])

            if not my_file.exists():
              print("File "+DATASET_PATH +'/'+ i_row[0]+" not Exist!")

            X_ordered[i][0] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ i_row[0], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image
            X_ordered[i][1] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ i_row[1], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image
    
            #X_ordered[i][0] = cv2.imread(DATASET_PATH +'/'+ i_row[1], cv2.IMREAD_COLOR)
            #X_ordered[i][1] = cv2.imread(DATASET_PATH +'/'+ i_row[2], cv2.IMREAD_COLOR)
    
            time_diff_ordered[i][0] = i_row[3]
            time_diff_ordered[i][1] = i_row[4]
            
            i+=1
        
    labels = [item[2] for item in data_file]
    labels = np.array(labels).astype(float)

    return X_ordered, labels, time_diff_ordered

X_test, Y_test, TD_test = load_model_data('Test_formatted.csv')
X_train, Y_train, TD_train = load_model_data('Train_formatted.csv')
X_val, Y_val, TD_val = load_model_data('Validation_formatted.csv')

# Set the seed value of the random number generator

# X_train, X_val, Y_train, Y_val, TD_train, TD_val = train_test_split(X_train_ordered, Y_train_ordered, TD_train_ordered, test_size=0.20, shuffle = True)
#X_train, X_test, Y_train, Y_test, TD_train, TD_test = train_test_split(X_train, Y_train, TD_train, test_size=0.25, shuffle = True)
# X_test, Y_test, TD_test = X_test_ordered, Y_test_ordered, TD_test_ordered

#X_train, Y_train = X_train_ordered, Y_train_ordered
#X_test, Y_test = X_train_ordered, Y_train_ordered
#X_val, Y_val = X_train_ordered, Y_train_ordered

print(X_train.shape)

X_train_T0 = X_train[:,0,:,:,:]
X_test_T0 = X_test[:,0,:,:,:]
X_val_T0 = X_val[:,0,:,:,:]
print(X_train_T0.shape)

X_train_T1 = X_train[:,1,:,:,:]
X_test_T1 = X_test[:,1,:,:,:]
X_val_T1 = X_val[:,1,:,:,:]
print(X_train_T1.shape)
plt.imshow(X_train_T1[1034])

TD_train_T0 = TD_train[:,0]
TD_test_T0 = TD_test[:,0]
TD_val_T0 = TD_val[:,0]
print(X_train_T0.shape)

TD_train_T1 = TD_train[:,1]
TD_test_T1 = TD_test[:,1]
TD_val_T1 = TD_val[:,1]

'''
X_T0_train = X_train.reshape(len(X_T0_train), IMG_HEIGHT, IMG_WIDTH, 3)
X_T0_test = X_test.reshape(len(X_T0_test), IMG_HEIGHT, IMG_WIDTH, 3)
X_T0_val = X_val.reshape(len(X_T0_val), IMG_HEIGHT, IMG_WIDTH, 3)

X_T1_train = X_train.reshape(len(X_T1_train), IMG_HEIGHT, IMG_WIDTH, 3)
X_T1_test = X_test.reshape(len(X_T1_test), IMG_HEIGHT, IMG_WIDTH, 3)
X_T1_val = X_val.reshape(len(X_T1_val), IMG_HEIGHT, IMG_WIDTH, 3)
'''

print("No of Training data: ",len(X_train))
print("No of Test data: ",len(X_test))
print("No of Validation data: ",len(X_val))

# convert class vectors to binary class matrices
#Y_train = keras.utils.to_categorical(Y_train, num_classes)
#Y_val = keras.utils.to_categorical(Y_val, num_classes)
#Y_test = keras.utils.to_categorical(Y_test, num_classes)

Y_train = Y_train.reshape(len(Y_train),1)
Y_test = Y_test.reshape(len(Y_test),1)
Y_val = Y_val.reshape(len(Y_val),1)

print('X_train shape:', X_train.shape, X_test.shape, X_val.shape)
print('Y_train shape:', Y_train.shape, Y_test.shape, Y_val.shape)

batch_size = 20
epochs = 700
lr_rate = 0.0001
momentum = 0.8
init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

def custom_activation(x):
    return (keras.activations.linear(x)*3)

def create_convolution_layers(input_img):

    x = Conv2D(filters=64, kernel_size=(2, 2), activation='relu', 
                 input_shape=(input_shape))(input_img)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(2, 2), activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(2, 2), activation='relu')(x)
    #x = Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
    #x = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
    #x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Flatten()(x)

    return x

frame_T0 = Input(shape=input_shape)
T0_model = create_convolution_layers(frame_T0)

frame_T1 = Input(shape=input_shape)
T1_model = create_convolution_layers(frame_T1)

x = concatenate([T0_model, T1_model])
x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)

#TD_T0_data = Input(shape=(1,))
#T0_model = concatenate([TD_T0_data, T0_model])

TD_T1_data = Input(shape=(1,))
#x = concatenate([x, TD_T1_data])

x = concatenate([TD_T1_data, x])

x = Dense(8596, activation='softplus')(x)
#x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
x = Dense(7100, activation='softplus')(x)
#x = BatchNormalization()(x)
x = Dense(2000, activation='softplus')(x)

output = keras.layers.Dense(1, activation='linear')(x)
#model.summary()   # Show a summary of the network architecture

model = Model(inputs=[TD_T1_data, frame_T0, frame_T1], outputs=[output])
model.summary()
# Stochastic Gradient Descent with momentum and a validation set to prevent overfitting

SGD = tf.keras.optimizers.SGD(lr=lr_rate, momentum=momentum)
adam = tf.keras.optimizers.Adam(lr=lr_rate, beta_1=0.60, beta_2=0.90)
Adadelta = tf.keras.optimizers.Adadelta(lr=lr_rate, rho=0.85)
Nadam = tf.keras.optimizers.Nadam(learning_rate=lr_rate, beta_1=0.9, beta_2=0.999)
Adamax = tf.keras.optimizers.Adamax(learning_rate=lr_rate, beta_1=0.90, beta_2=0.99, epsilon=1e-07, name="Adamax")
Adagrad = tf.keras.optimizers.Adagrad(learning_rate=lr_rate, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")

model.compile(loss=keras.losses.mean_squared_error,
#           optimizer=Adadelta,
           optimizer="Adamax",
#           optimizer=Adagrad,
            metrics=['mae'])

earlystopper = EarlyStopping(monitor='val_loss', patience=epochs, verbose=True, mode= 'min')
checkpoint = ModelCheckpoint('./vgg_cnn_modelcp.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto', period=1)
#checkpoint = tf.keras.callbacks.ModelCheckpoint('.vgg_cnn_modelcp.hdf5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='min', period=1)
# The history structure keeps tabs on what happened during the session

history = model.fit([TD_train_T1, X_train_T0, X_train_T1], Y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1,
          validation_data = ([TD_val_T1, X_val_T0, X_val_T1], Y_val),
          callbacks=[checkpoint])

#model.load_weights('./vgg_cnn_modelcp.ckpt')
#best_model = tf.keras.models.load_model('./vgg_cnn_modelcp.ckpt')
score = model.evaluate([TD_test_T1, X_test_T0, X_test_T1], Y_test, verbose=0)

print('Test loss:', score[0])
print('Test MAE:', score[1])

def get_prediction_interval(prediction, y_test, test_predictions, pi=.95):
    '''
    Get a prediction interval for a linear regression.

    INPUTS: 
        - Single prediction, 
        - y_test
        - All test set predictions,
        - Prediction interval threshold (default = .95) 
    OUTPUT: 
        - Prediction interval for single prediction
    '''

    #get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)

    #get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev

    #generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval

    return lower, prediction, upper


# get_prediction_interval(predictions[0], y_test, predictions)

model.load_weights('./vgg_cnn_modelcp.h5')
score = model.evaluate([TD_test_T1, X_test_T0, X_test_T1], Y_test, verbose=0)

print(X_test_T0.shape)
print(X_test_T1.shape)
print('Test loss:', score[0])
print('Test MAE:', score[1])

predictions = model.predict([TD_test_T1, X_test_T0, X_test_T1], verbose=0)

# cm = confusion_matrix(Y_test.argmax(axis=1), predictions.argmax(axis=1))
# print(cm)

# target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
# print(classification_report(Y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names))

for i in range(X_test_T0.__len__()):
    # subplt = plt.subplot(int(i / 10) + 1, 10, i + 1)
    # no sense in showing labels if they don't match the letter
    #predicted_cars = np.argmax(predictions[i])
    #actual_cars = np.argmax(Y_test[i])
    print(predictions[i], end = ',')
    print(Y_test[i])

print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig = plt.figure(figsize=(16,8), dpi= 100, facecolor='w', edgecolor='k')
plt.ylim(0,20)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Loss', 'val loss'], loc='upper left')
plt.show()
