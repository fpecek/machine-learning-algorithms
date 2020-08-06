import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def visualise_dataset_sample(features, targets, num_samples = 20):
    indices = list(range(features.shape[0]))
    num_samples = min(num_samples, len(indices))
    sample_indices = np.random.choice(indices, num_samples)
    sample_features = features[sample_indices]
    sample_targets = targets[sample_indices]
    plt.xlabel("BMI")
    plt.ylabel("Progression")
    plt.scatter(sample_features, sample_targets, color='black');plt.show()

if __name__ == "_main__":

    #load the diabetes dataset and plot random sample for BMI feature
    features_all, targets = datasets.load_diabetes(return_X_y=True)
    visualise_dataset_sample(features_all[:, np.newaxis, 3], targets)
    features = features_all[:, np.newaxis, 3]

    # Split the features into training/testing sets
    features_train = features[:-100]
    features_test = features[-100:]

    # Split the targets into training/testing sets
    targets_train = targets[:-100]
    targets_test = targets[-100:]

    # Create linear regression object and fit to training data
    regr = linear_model.LinearRegression()
    regr.fit(features_train, targets_train)

    #predict the progression values for the test features
    predictions_test = regr.predict(features_test)

    print('Mean squared error: %.2f'
      % mean_squared_error(targets_test, predictions_test))

    # Plot outputs
    plt.scatter(features_test, targets_test,  color='black')
    plt.plot(features_test, predictions_test, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()
