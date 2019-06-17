from __future__ import print_function
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.utils import np_utils
from dbn.tensorflow import SupervisedDBNClassification
from sklearn.model_selection import train_test_split

def load_data():
    labels = np.loadtxt('label.txt')
    encoded_seq = np.loadtxt('encoded_seq.txt')
    x_train,x_test,y_train,y_test = train_test_split(encoded_seq,labels,test_size=0.2)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def svm_classifier_rbf(train_x, train_y):
    model = SVC(kernel='rbf', gamma=0.01, C=1)
    model.fit(train_x, train_y)
    return model


def svm_classifier_linear(train_x, train_y):
    model = SVC(kernel='linear', gamma=0.01, C=1)
    model.fit(train_x, train_y)
    return model


def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


def decision_tree_classifier(train_x, train_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


def random_forest_classifier(train_x, train_y, num_classes=3):
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    return model


def ada_boost_classifier(train_x, train_y, num_classes=3):
    model = AdaBoostClassifier()
    model.fit(train_x, train_y)
    return model

def dbn(train_x, train_y, num_classes=3):
    model = SupervisedDBNClassification(learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    model.fit(train_x, train_y)
    return model

def training_process(classifier, x_train, y_train, x_test, y_test):
    print("======================")
    print('Classifier: {}'.format(classifier))
    start_time = time.time()
    temp_model = eval(classifier)(x_train, y_train)
    y_train_predict = temp_model.predict(x_train)
    training_accuracy = metrics.accuracy_score(y_train, y_train_predict)
    print('training accuracy: {}'.format(training_accuracy))
    y_predict = temp_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print('testing accuracy: {}'.format(accuracy))
    print('training took %fs' % (time.time() - start_time))


    prob = temp_model.predict(x_test)
    prob = np_utils.to_categorical(prob, num_classes=3)
    predict = temp_model.predict(x_test)
    predict = np_utils.to_categorical(predict, num_classes=3)
    y_true = np_utils.to_categorical(y_test, num_classes=3)
    true_0 = y_true[:,0]
    prob_0 = prob[:,0]
    predict_0 = predict[:,0]
    auc = metrics.roc_auc_score(true_0,prob_0)
    precision = metrics.precision_score(true_0,predict_0)
    recall = metrics.recall_score(true_0,predict_0)
    f1 = metrics.f1_score(true_0,predict_0)
    print('acceptor AUC of :%f'%auc)
    print('acceptor precision of :%f'%precision)
    print('acceptor recall of :%f'%recall)
    print('acceptor f1 of :%f'%f1)

    
    true_1 = y_true[:,1]
    prob_1 = prob[:,1]
    predict_1 = predict[:,1]
    auc = metrics.roc_auc_score(true_1,prob_1)
    precision = metrics.precision_score(true_1,predict_1)
    recall = metrics.recall_score(true_1,predict_1)
    f1 = metrics.f1_score(true_1,predict_1)
    print('donor AUC of :%f'%auc)
    print('donor precision of :%f'%precision)
    print('donor recall of :%f'%recall)
    print('donor f1 of :%f'%f1)

def main():
    x_train, y_train, x_test, y_test = load_data()
    
    training_process('svm_classifier_linear',x_train,y_train,x_test,y_test)
    training_process('svm_classifier_rbf',x_train,y_train,x_test,y_test)
    training_process('logistic_regression_classifier', x_train, y_train, x_test, y_test)
    training_process('decision_tree_classifier', x_train, y_train, x_test, y_test)
    training_process('random_forest_classifier', x_train, y_train, x_test, y_test)
    training_process('ada_boost_classifier', x_train, y_train, x_test, y_test)
    training_process('dbn',x_train,y_train,x_test,y_test)
                     
if __name__ == '__main__':
    main()