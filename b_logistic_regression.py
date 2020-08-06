import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, recall_score, precision_score

def visualise_dataset_sample(features, targets, class_names, feature_names, num_samples = 20):
    fig, axes = plt.subplots(4, 4)
    versicolour_mask = targets == 1
    for i in range(4):
        for j in range(0, 4):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 3:
                ax.set_xlabel(feature_names[j])
            if j == 0:
                ax.set_ylabel(feature_names[i])
            if i == j:
                continue
            for class_index, (class_name, color) in enumerate(zip(class_names, ['red', 'green', 'blue'])):
                features_i_class = features[targets == class_index][:, i]
                features_j_class = features[targets == class_index][:, j]
                ax.scatter(features_i_class, features_j_class, color=color)
    plt.show()

def train_test_split(features, targets, class_names, test_split_ratio):
    # Split the features into training/testing sets
    features_train = []
    features_test = []
    targets_train = []
    targets_test = []
    for class_index, class_name in enumerate(class_names):
      class_mask = targets == class_index
      class_features = features[class_mask]
      class_num_test_samples = int(class_features.shape[0] * test_split_ratio)
      class_num_train_samples = class_features.shape[0] - class_num_test_samples
      class_indices = list(range(class_features.shape[0]))
      np.random.shuffle(class_indices)
      test_indices = class_indices[:class_num_test_samples]
      train_indices = class_indices[class_num_test_samples:]
      features_train.append(class_features[train_indices])
      features_test.append(class_features[test_indices])
      targets_train.append([class_index]*class_num_train_samples)
      targets_test.append([class_index]*class_num_test_samples)

    features_train = np.concatenate(features_train)
    features_test = np.concatenate(features_test)
    targets_train = np.concatenate(targets_train)
    targets_test = np.concatenate(targets_test)

    return features_train, targets_train, features_test, targets_test

if __name__ == "_main__":
    #load the visualisze the iris dataset
    features, targets = datasets.load_iris(return_X_y=True)
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']   

    visualise_dataset_sample(features, targets, class_names, feature_names, 20)
    class_names = ['iris-setosa', 'iris-versicolour', 'iris-virginica']

    #create the train/test split
    test_split_ratio = 0.2
    features_train, targets_train, features_test, targets_test = train_test_split(features, targets, \
      class_names, test_split_ratio)

    #convert to binary classification problem (iris-setosa)
    targets_train[targets_train != 0] = 1
    targets_test[targets_test != 0] = 1

    #convert to binary classification problem (iris-versicolour)
    #targets_train[targets_train != 1] = 0
    #targets_test[targets_test != 1] = 0

    # Create logistic regression model and fit to training data
    logreg = linear_model.LogisticRegression()
    logreg.fit(features_train, targets_train)

    #predict the classes for test data
    predictions_test = logreg.predict(features_test)

    #number of samples in the test set with 
    print('Accuracy : {}, Precision : {}, Recall: {}'.format(
        accuracy_score(targets_test, predictions_test), \
        precision_score(targets_test, predictions_test), \
        recall_score(targets_test, predictions_test)))
