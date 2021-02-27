import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
from keras.utils import np_utils


labels = np.loadtxt('label_205.txt')
encoded_seq = np.loadtxt('encoded_seq_205.txt')

X_train, X_test, Y_train, Y_test = train_test_split(encoded_seq,labels,test_size=0.2)

classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=100,
                                         activation_function='sigmoid',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)
classifier.save('dbn.pkl')

Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
