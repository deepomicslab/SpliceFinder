from __future__ import print_function
import numpy as np
import time
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.applications import *
import linecache

x_test = np.loadtxt('encoded_seq')
x_test = x_test.reshape(-1,400, 4)
y_test = np.loadtxt('y_label')
y_true = y_test
y_test = np_utils.to_categorical(y_test, num_classes=3)

model = load_model('CNN.h5')
print(model.summary())
loss,accuracy = model.evaluate(x_test,y_test)
print('testing accuracy: {}'.format(accuracy))

predict = model.predict_classes(x_test).astype('int')

positive=0
positive_predict=0
true_positive=0

for i in range(len(y_true)):
    if y_true[i]!=2:
        positive = positive+1
        print(i,end=' ')
        print(y_true[i],end='')
        print(':',end='')
        print(predict[i])
        
    if predict[i]!=2:
        positive_predict = positive_predict+1

    if y_true[i]!=2 and predict[i]!=2:
        true_positive = true_positive+1
print('# of real positive is %f'%positive)
print('# of predicted positive is %f'%positive_predict)
print('# of true positive is %f'%true_positive)

falsepositive = open('fp_seq','w')

for i in range(len(y_true)):
    if (predict[i]!=2) and (y_true[i]==2):
        seq = linecache.getline(r'encoded_seq',i+1)
        falsepositive.write(seq)

        


