# -*- coding: utf-8 -*-
"""CNN_LSTM.ipynb
CNN_LSTM model to predict number of new vehicle in the given frame. It will accept 3 frames at a time (t-2, t-1 and t). To predict number of new vehicle in the frame t.
Average vehicle takes 3 frames to pass the region of interest thus, I have pass 3 frames in the model.
"""

import sys
import os
import cv2
import keras
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, TimeDistributed, Reshape, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, LSTM, concatenate, AveragePooling2D 
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import os

print(tf.__version__)
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
print(cv2.__version__)
tf.random.set_seed(7)

#np.set_printoptions(threshold=np.inf)
batch_size = 25
num_classes = 1

seq_steps = 3
no_of_examples = 0
checkpoint_path = './cnn_lstm_cp.ckpt'

#img = cv2.imread(dts_path +'/'+ i_name, cv2.IMREAD_COLOR)  # uint8 image
#norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

DATASET_PATH ='FG 1.0'
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 123, 277, 3

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
        if i_row[0].endswith('.jpg') and i_row[1].endswith('.jpg') and i_row[2].endswith('.jpg'):
            
            X_ordered[i][0] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ i_row[0], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image
            X_ordered[i][1] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ i_row[1], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image
            X_ordered[i][2] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ i_row[2], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image
            
            #X_ordered[i][0] = cv2.imread(DATASET_PATH +'/'+ i_row[0], cv2.IMREAD_COLOR)
            #X_ordered[i][1] = cv2.imread(DATASET_PATH +'/'+ i_row[1], cv2.IMREAD_COLOR)
            #X_ordered[i][1] = cv2.imread(DATASET_PATH +'/'+ i_row[2], cv2.IMREAD_COLOR)

            
            time_diff_ordered[i][0] = i_row[4]
            time_diff_ordered[i][1] = i_row[5]
            time_diff_ordered[i][2] = i_row[6]

            i+=1

    labels = [item[3] for item in data_file]
    labels = np.array(labels).astype(float)

    return X_ordered, labels, time_diff_ordered


X_test_ordered, Y_test_ordered, TD_test_ordered = load_model_data('Test_formatted.csv')
X_train_ordered, Y_train_ordered, TD_train_ordered = load_model_data('Train_formatted.csv')

#time_diff /= 1000

X_train_ordered, X_val_ordered, Y_train_ordered, Y_val_ordered, TD_train_ordered, TD_val_ordered = train_test_split(X_train_ordered, Y_train_ordered, TD_train_ordered,test_size=0.20, shuffle = True)
#X_train_ordered, X_test_ordered, Y_train_ordered, Y_test_ordered, TD_train_ordered, TD_test_ordered = train_test_split(X_train_ordered, Y_train_ordered, TD_train_ordered, test_size=0.25, shuffle = False)

#X_train_ordered, Y_train_ordered, TD_train_ordered = X_train_ordered, Y_train_ordered, TD_train_ordered
#X_test_ordered, Y_test_ordered, TD_test_ordered = X_train_ordered, Y_train_ordered, TD_train_ordered
#X_val_ordered, Y_val_ordered, TD_val_ordered = X_train_ordered, Y_train_ordered, TD_train_ordered

print("Second Split")
print(len(X_train_ordered))
print(len(X_val_ordered))

X_train_ordered = X_train_ordered.reshape(len(X_train_ordered), seq_steps, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
X_test_ordered = X_test_ordered.reshape(len(X_test_ordered), seq_steps, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
X_val_ordered = X_val_ordered.reshape(len(X_val_ordered), seq_steps, IMG_HEIGHT, IMG_WIDTH, CHANNELS)

Y_train_ordered = Y_train_ordered.reshape(len(Y_train_ordered),1)
Y_test_ordered = Y_test_ordered.reshape(len(Y_test_ordered),1)
Y_val_ordered = Y_val_ordered.reshape(len(Y_val_ordered),1)

TD_train_ordered = TD_train_ordered.reshape(len(TD_train_ordered), seq_steps, 1)
TD_test_ordered = TD_test_ordered.reshape(len(TD_test_ordered), seq_steps, 1)
TD_val_ordered = TD_val_ordered.reshape(len(TD_val_ordered), seq_steps, 1)

print("Final")
print("No of Training data: ",len(X_train_ordered), len(TD_train_ordered))
print("No of Test data: ",len(X_test_ordered), len(TD_test_ordered))
print("No of Validation data: ",len(X_val_ordered), len(TD_val_ordered))

x_input_shape = (seq_steps, IMG_HEIGHT, IMG_WIDTH, CHANNELS)

X_train_ordered = X_train_ordered.astype('float32')
X_test_ordered = X_test_ordered.astype('float32')
X_val_ordered = X_val_ordered.astype('float32')

#X_train_ordered *= 1.50
#X_test_ordered *= 1.50
#X_val_ordered *= 1.50

plt.imshow(X_test_ordered[0,0])

print('X_train_ordered shape:', X_train_ordered.shape)
print('TD_train_ordered shape:', TD_train_ordered.shape)
print('Y_train_ordered shape:', Y_train_ordered.shape)

epochs = 500
lr_rate = 0.0001
momentum =  0.9

image = Input(shape=x_input_shape)
x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))(image)
x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))(x)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))(x)
x = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))(x)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
x = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))(x)
x = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))(x)
#x = TimeDistributed(Conv2D(filters=256, kernel_size=(1, 1), activation='relu'))(x)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
#x = TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))(x)
#x = TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))(x)
#x = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), activation='relu'))(x)
#x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
#x = TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))(x)
#x = TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))(x)
#x = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), activation='relu'))(x)
#x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
#x = TimeDistributed(Dropout(0.2))(x)
x = TimeDistributed(Flatten())(x)
x = TimeDistributed(Dense(4096, activation='relu'))(x)
#x = TimeDistributed(Dropout(0.2))(x)
x = TimeDistributed(Dense(2000, activation='relu'))(x)

time_diff_data = Input(shape=(seq_steps, 1))
x = concatenate([time_diff_data, x])
x = LSTM(units=140, return_sequences=True)(x)
x = LSTM(units=75, return_sequences=False)(x)

#x = Dense(30,activation='relu')(x)
out_put = Dense(num_classes)(x)
model = Model(inputs=[image, time_diff_data], outputs=out_put)
model.summary()

SGD = tf.keras.optimizers.SGD(lr=lr_rate, momentum=momentum)
adam = tf.keras.optimizers.Adam(lr=lr_rate, beta_1=0.75, beta_2=0.9)
Adadelta = tf.keras.optimizers.Adadelta(lr=lr_rate, rho=0.95)

model.compile(loss="mean_squared_error",
#              optimizer="Adadelta",
              optimizer="adam",
#            optimizer=SGD,
            metrics=['mae','acc'])

earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs, verbose=True, mode= 'min')
#checkpoint = tf.keras.callbacks.ModelCheckpoint('./modelcp.ckpt', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)

hist = model.fit(x = [X_train_ordered, TD_train_ordered], y = Y_train_ordered,
          batch_size = batch_size, epochs = epochs,verbose=2,
          validation_data=([X_val_ordered, TD_val_ordered], Y_val_ordered),
          validation_split=0.0,
          callbacks = [earlystopper])

best_model = tf.keras.models.load_model('./best_model.ckpt')
score = best_model.evaluate([X_test_ordered, TD_test_ordered], Y_test_ordered, verbose=0)

print('Test loss:', score[0])
print('Test MAE:', score[1])

print(model.metrics_names)
output = model.predict([X_test_ordered, TD_test_ordered])
#output = model.predict(X_test_orzdered)

np.set_printoptions(precision=3)

print("Predictions||Actual:")

for i in range(len(Y_test_ordered)):
    print(output[i], end = ',')
    print(Y_test_ordered[i])

#print(hist.history)

print('Test loss:', score[0])
print('Test MAE:', score[1])
#print('Test Accuracy:', score[2])
np.set_printoptions(precision=3)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Loss', 'val loss'], loc='upper left')
plt.show()

