"""
You are allowed to use the `sklearn` package for SVM.

See the documentation at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
from sklearn.svm import SVC

from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt

class SVM(object):
    """
    SVM method.
    """

    def __init__(self, C, kernel, gamma=1., degree=1, coef0=0.):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            C (float): the weight of penalty term for misclassifications
            kernel (str): kernel in SVM method, can be 'linear', 'rbf' or 'poly' (:=polynomial)
            gamma (float): gamma prameter in rbf and polynomial SVM method
            degree (int): degree in polynomial SVM method
            coef0 (float): coef0 in polynomial SVM method
        """
        """
        In the method we are creating the object of SVC class, where we have to point:
        kernel: kernel we want to use for our SVM method, default = 'rbf',
        C: is the regularization parameter which describe the tradeoff between size of minimum margin and correct classification,
        gamma: used in 'poly', 'rbf', and 'sigmoid' to specify the kernel's coefficients = weights, higher values fits the dataset better, which is some cases can lead to overfitting,
        degree: used in 'poly' kernel to specify the degree of polynomial function,
        coef0: used in 'poly' and 'sigmoid' as an independent term.
        Then we are training our model using loss function and gradient updates to maximize margin between data.
        At the end we are invoking the method SVM.predict(training_data), to obtain the predicted values for training data.

        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
    def fit(self, training_data, training_labels):
        """
        Trains the model by SVM, then returns predicted labels for training data.
        We are checking which side of the boundary that we find during the training sample is. 

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.clf_linear = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        self.clf_linear.fit(training_data, training_labels)

        return self.predict(training_data)
    
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = self.clf_linear.predict(test_data)
        return pred_labels
    
    def graph(self, training_data, training_labels):
        # Set x-axis ticks and labels
        param_range = np.logspace(-1, 3, 5)
        # Create an instance of the SVM model
        train_scores, test_scores = validation_curve(
        SVC(kernel=self.kernel),
        training_data,
        training_labels,
        param_name="C",
        param_range=param_range,
        scoring="accuracy",
        n_jobs=2,
        )
        # Get the mean and standard deviation of accuracy
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Set labels
        plt.title("Validation Curve with SVM")
        plt.xlabel(r"C")
        plt.ylabel("Test Set Accuracy")
        plt.ylim(0.0, 1.1)
        lw = 2

        # Create a training set plot using log function
        plt.semilogx(
            param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
        )
        plt.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.2,
            color="darkorange",
            lw=lw,
        )
        # Create a testing set plot using log function
        plt.semilogx(
            param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
        )
        plt.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.2,
            color="navy",
            lw=lw,
        )
        plt.legend(loc="best")
        plt.show()