import numpy as np
from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn
import matplotlib.pyplot as plt


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        
    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        scores = np.dot(data, W)
        max_scores = np.max(scores, axis=1, keepdims=True)
        scores -= max_scores
        exp_result = np.exp(scores)
        return (exp_result/np.sum(exp_result, axis = 1, keepdims = True))

    def loss_logistic_multi(data, labels, w):
        """ 
        Loss function for multi class logistic regression, i.e., multi-class entropy.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value 
        """
        return -np.sum((labels*np.log(f_softmax(data,w))))
    
    def gradient_logistic_multi(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        return data.T @ (self.f_softmax(data, W) - labels)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = training_data.shape[1]
        labels = label_to_onehot(training_labels)
        C = labels.shape[1]
        self.W = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            gradient = self.gradient_logistic_multi(training_data, labels, self.W)
            self.W -= self.lr * gradient

            predictions = self.predict(training_data)
            if accuracy_fn(predictions, onehot_to_label(labels)) == 1:
                break
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        Y = self.f_softmax(test_data, self.W)
        labels = np.argmax(Y, axis=1)
        return labels
    
    def graph(self,learning_rates,max_iters,xtrain,ytrain,xtest,ytest):
        
        learning_rates = [0.000001,0.00001,0.0001,0.001, 0.01]
        test_accuracies = []

        for lr in learning_rates:
            # Create an instance of the logistic regression model with the current learning rate
            method_obj = LogisticRegression(lr=lr, max_iters=max_iters)
            # Fit the model on the training data
            preds_train = method_obj.fit(xtrain, ytrain)
            # Predict on the test data
            preds = method_obj.predict(xtest)
            # Compute accuracy on the test set
            acc = accuracy_fn(preds, ytest)
            # Append the test set accuracy to the list
            test_accuracies.append(acc)

        # Set x-axis ticks and labels
        xticks = [0.000001,0.00001,0.0001,0.001, 0.01]  # Specify the x-axis tick values
        num_ticks = 5
        #xticks = np.linspace(min(xticks), max(xticks), num_ticks)

        xtick_labels = ['1e-6','1e-5','1e-4','1e-3', '1e-2']  # Specify the x-axis tick labels
        plt.xticks(xticks, xtick_labels)  # Set the x-axis ticks and labels
        plt.xscale('log')
        print(learning_rates)
        print(test_accuracies)
        plt.plot(learning_rates, test_accuracies)
        plt.xlabel('Learning Rate')
        plt.ylabel('Test Set Accuracy')
        plt.title('Test Set Accuracy vs. Learning Rate')
        plt.grid(True)
        plt.show()
