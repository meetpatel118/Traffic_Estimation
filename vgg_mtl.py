# -*- coding: utf-8 -*-
"""VGG_MTL.ipynb
The multiple task learrning model accepts two images as an input to predict the number of new vehicles and the total number of vehicles.
The model also accepts time in milliseconds till the last frame.
This model has produced best accuracy and used to record traffic count from all image sequence data.
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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Reshape, concatenate
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix,classification_report
from keras.layers.normalization import BatchNormalization
from scipy.stats import norm
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

random_seed = 2
np.random.seed(random_seed)
DATASET_PATH ='FG 1.0'

IMG_HEIGHT, IMG_WIDTH = 123, 277
CHANNELS = 3
num_classes = 6
seq_steps = 2
batch_size = 25
input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
print(tf.__version__)

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

!unzip Data.zip

#-------------------------------------------Data load function and generator--------------------------------------------------
DATASET_PATH ='FG 1.0'

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataset, batch_size=25, dim=input_shape, n_channels=3,
                 shuffle=True):
        'Initialization'
        # print(dataset)
        self.dim = dim
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_steps = seq_steps
        self.indexes = dataset.index
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in indexes]
        # print(list_IDs_temp)
        # Generate data
        X_time_diff_T1, X_ordered_T0, X_ordered_T1, Y_new_count_labels, Y_count_labels = self.__data_generation(list_IDs_temp)

        return [X_time_diff_T1, X_ordered_T0, X_ordered_T1], [Y_new_count_labels, Y_count_labels]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X_ordered_T0 = np.empty((self.batch_size, *self.dim))
        X_ordered_T1 = np.empty((self.batch_size, *self.dim))
        X_time_diff_T1 = np.empty((self.batch_size, 1), dtype=float)
        Y_new_count_labels = np.empty((self.batch_size, 1), dtype=float)
        Y_count_labels = np.empty((self.batch_size, 1), dtype=float)

        i = 0
  
        # Generate data
        for i_row, ID in enumerate(list_IDs_temp):
            # Store sample
            img_1 = Path(DATASET_PATH +'/'+ self.dataset[ID][0])
            img_2 = Path(DATASET_PATH +'/'+ self.dataset[ID][1])
            if not img_1.exists():
              print("File "+DATASET_PATH +'/'+ self.dataset[ID][0]+" not Exist!")

            #print(self.dataset[ID])
            X_ordered_T0[i] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ self.dataset[ID][0], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image
            X_ordered_T1[i] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ self.dataset[ID][1], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image

            #time_diff_T0[i] = self.dataset[list_IDs_temp[i_row]][3]
            X_time_diff_T1[i] = self.dataset[ID][4]
        
            Y_new_count_labels[i] = self.dataset[ID][2]
            Y_new_count_labels = np.array(Y_new_count_labels).astype(int)

            Y_count_labels[i] = self.dataset[ID][5]
            Y_count_labels = np.array(Y_count_labels).astype(int)

            i+=1

        return X_time_diff_T1, X_ordered_T0, X_ordered_T1, Y_new_count_labels, Y_count_labels

def load_csv_data(data_file):
    
    if os.path.exists(DATASET_PATH +'/'+ data_file):
        with open(DATASET_PATH +'/'+ data_file, newline='') as csvfile:
            labelsfile = list(csv.reader(csvfile))
    else:
        labelsfile = [[]]

    data_file = labelsfile          # For .csv file

    return data_file

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

    new_count_labels = [item[2] for item in data_file]
    new_count_labels = np.array(new_count_labels).astype(int)

    count_labels = [item[5] for item in data_file]
    count_labels = np.array(count_labels).astype(int)

    return X_ordered, new_count_labels, count_labels, time_diff_ordered

#-------------------------------------------Load data from CSV file--------------------------------------------------
X_train = load_csv_data('Train_formatted.csv')
X_val = load_csv_data('Validation_formatted.csv')

# Set the seed value of the random number generator
# X_train, X_val = train_test_split(train_data, test_size=0.20, shuffle = True)

# Generators
training_generator = DataGenerator(dataset = X_train)
validation_generator = DataGenerator(dataset = X_val)

epochs = 1
lr_rate = 0.1
momentum = 0.8
checkpoint_filepath = './vgg_MTL_modelcp.h5'
initializer_1 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
initializer = tf.keras.initializers.Zeros()
#, kernel_initializer = zero_init
my_initializer = tf.keras.initializers.glorot_uniform(seed = 2)

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

    # x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
    # x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Flatten()(x)

    return x

frame_T0 = Input(shape=input_shape)
T0_model = create_convolution_layers(frame_T0)

frame_T1 = Input(shape=input_shape)
T1_model = create_convolution_layers(frame_T1)

x = concatenate([T0_model, T1_model])
# x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)

# TD_T0_data = Input(shape=(1,))
# T0_model = concatenate([TD_T0_data, T0_model])

TD_T1_data = Input(shape=(1,))
#x = concatenate([x, TD_T1_data])

x = concatenate([TD_T1_data, x])
#---------------------------------------------Number of new vehicle--------------------------------------------------------------------
new_veicle_x = Dense(2096, activation='softplus')(x)
new_veicle_x = Dense(1000, activation='softplus')(new_veicle_x)
new_veicle_output = Dense(1, activation='linear', name="new_veicle_count")(new_veicle_x)
#---------------------------------------------Total number of vehicle------------------------------------------------------------------
count_x = Flatten()(T1_model)
count_x = Dense(500, activation='relu')(count_x)
count_x = Dense(300, activation='relu')(count_x)
count_output = Dense(1, activation='linear', name="total_veicle_count")(count_x)
#---------------------------------------------------------------------------------------------------------------------------------------

model = Model(inputs = [TD_T1_data, frame_T0, frame_T1], outputs = [new_veicle_output, count_output])
model.summary()

# Stochastic Gradient Descent with momentum and a validation set to prevent overfitting
SGD = tf.keras.optimizers.SGD(lr=lr_rate, momentum=momentum)
adam = tf.keras.optimizers.Adam(lr=lr_rate, beta_1=0.60, beta_2=0.90)
Adadelta = tf.keras.optimizers.Adadelta(lr=lr_rate, rho=0.85)
Adamax = tf.keras.optimizers.Adamax(learning_rate=lr_rate, beta_1=0.90, beta_2=0.99, epsilon=1e-07, name="Adamax")
Adagrad = tf.keras.optimizers.Adagrad(learning_rate=lr_rate, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")

model.compile(loss=keras.losses.mean_squared_error,
#           optimizer="Adadelta",
           optimizer="Adamax",
#           optimizer=Adagrad,
            metrics=['mae'])

earlystopper = EarlyStopping(monitor='val_loss', patience=350, verbose=True, mode= 'min')
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
#checkpoint = tf.keras.callbacks.ModelCheckpoint('.vgg_cnn_modelcp.hdf5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='min', period=1)

# history = model.fit_generator(generator = training_generator,
#           epochs = epochs,
#           verbose = 1,
#           callbacks=[checkpoint],
#           validation_data = validation_generator)
#use data generator at fit function, can also use data generator for validation data

#model.load_weights('./vgg_MTL_modelcp.ckpt')
#best_model = tf.keras.models.load_model('./vgg_cnn_modelcp.ckpt')
score = model.evaluate([TD_test_T1, X_test_T0, X_test_T1], [Y_new_count_labels, Y_test_count], verbose=0)

print('Test loss:', score[0])
print('Test new_veicle_count loss:', score[1])
print('Test total_veicle_count loss:', score[2])
print('Test new_veicle_count MAE:', score[3])
print('Test total_veicle_count MAE:', score[4])

model = tf.keras.models.load_model('/content/drive/My Drive/Model_Checkpoint/vgg_MTL_modelcp_CC_New.h5')
model.summary()

X_test_T0 = X_test[:,0,:,:,:]
X_test_T1 = X_test[:,1,:,:,:]

TD_test_T0 = TD_test[:,0]
TD_test_T1 = TD_test[:,1]

Y_test_new_count = Y_test_new_count.reshape(len(Y_test_new_count),1)
Y_test_count = Y_test_count.reshape(len(Y_test_count),1)

print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test_new_count.shape, Y_test_count.shape)
#--------------------------------------------------------------------------------------------------------------------

def get_prediction_interval(prediction, y_test, test_predictions, pi=.10):
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
    z_score = norm.ppf(ppf_lookup)
    interval = z_score * stdev

    #generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval

    print(lower, prediction, upper)

score = model.evaluate([TD_test_T1, X_test_T0, X_test_T1], [Y_test_new_count, Y_test_count], verbose=0)

print(X_test_T0.shape)
print(X_test_T1.shape)
# print('Test loss:', score[0])
# print('Test MAE:', score[1])

predictions = model.predict([TD_test_T1, X_test_T0, X_test_T1], verbose=0)

_file = open('Counts_test_CC.csv','w')
_writer = csv.writer(_file,delimiter=',',lineterminator='\n')

for i in range(X_test_T0.__len__()):
    # print(predictions[i])
    _writer.writerow((Y_test_new_count[i], predictions[0][i], Y_test_count[i], predictions[1][i]))

_file.close()

print('Test loss:', score[0])
print('Test new_vehicle_count loss:', score[1])
print('Test total_vehicle_count loss:', score[2])
print('Test new_vehicle_count MAE:', score[3])
print('Test total_vehicle_count MAE:', score[4])
print('Test new_vehicle_count MAPE:', score[3]/2.5)
print('Test total_vehicle_count MAPE:', score[4]/4.11)

plt.ylim(0,60)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_new_veicle_count_loss'])
plt.plot(history.history['val_total_veicle_count_loss'])
# plt.plot(history.history['val_new_veicle_count_loss'])
# plt.plot(history.history['val_total_veicle_count_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Loss', 'val loss'], loc='upper left')
plt.show()

model.save('vgg_MTL_modelcp.h5')
