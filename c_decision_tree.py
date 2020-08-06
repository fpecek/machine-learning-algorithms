import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from b_logistic_regression import visualise_dataset_sample, train_test_split

if __name__ == "_main__":

    #load the visualisze the iris dataset
    features, targets = datasets.load_iris(return_X_y=True)
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']   
    class_names = ['iris-setosa', 'iris-versicolour', 'iris-virginica']

    visualise_dataset_sample(features, targets, class_names, feature_names, 20)
    
    test_split_ratio = 0.2
    features_train, targets_train, features_test, targets_test = train_test_split(features, targets, \
      class_names, test_split_ratio)

    #convert to binary classification problem (iris-setosa)
    #targets_train[targets_train != 0] = 1
    #targets_test[targets_test != 0] = 1

    #convert to binary classification problem (iris-versicolour)
    targets_train[targets_train != 1] = 0
    targets_test[targets_test != 1] = 0

    # Create decision tree classifier and fit to training data
    clf = DecisionTreeClassifier()
    clf.fit(features_train, targets_train)

    #predict the classes for test data
    predictions_test = clf.predict(features_test)

    #number of samples in the test set with 
    print('Accuracy : {}, Precision : {}, Recall: {}'.format(
        accuracy_score(targets_test, predictions_test), \
        precision_score(targets_test, predictions_test), \
        recall_score(targets_test, predictions_test)))
    
    