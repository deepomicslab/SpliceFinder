from __future__ import print_function
import numpy as np
import time
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import LSTM, GRU
from keras.applications import *
from sklearn.model_selection import train_test_split

def load_data(n):

    labels = np.loadtxt('label_'+str(n)+'.txt')
    encoded_seq = np.loadtxt('encoded_seq_'+str(n)+'.txt')

    x_train,x_test,y_train,y_test = train_test_split(encoded_seq, labels, test_size=0.2)

    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

def rnn():

    model = Sequential()
    model.add(LSTM(50,return_sequences=True,activation='relu',input_shape=(1,1600)))
    model.add(Dense(100))

    model.add(Dense(3))
    model.add(Activation('softmax'))


    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def cnn():

    model = Sequential()
        

    model.add(keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, 1, 1600), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    
    model.add(Dropout(0.3))
    model.add(Dense(3,activation='softmax'))
    
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def training_process(x_train,y_train,x_test,y_test,j):

    x_train = x_train.reshape(-1, 1, 1600)
    y_train = np_utils.to_categorical(y_train, num_classes=3)
    x_test = x_test.reshape(-1, 1, 1600)
    y_true = y_test
    y_test = np_utils.to_categorical(y_test, num_classes=3)

    print("======================")
    print('Recurent Neural Network')
    start_time = time.time()
    model = rnn()
    model.fit(x_train, y_train, epochs=20, batch_size=50)
    model.save('RNN_'+str(j)+'.h5')
    loss,accuracy = model.evaluate(x_test,y_test)
    print('testing accuracy: {}'.format(accuracy))
    print('training took %fs'%(time.time()-start_time))

def main():
    for j in [0,2,14,27,102,104,122,173,191,200,205]: 
        
        x_train,y_train,x_test,y_test = load_data(j)

        training_process(x_train,y_train,x_test,y_test,j)


if __name__ == '__main__':
    main()
