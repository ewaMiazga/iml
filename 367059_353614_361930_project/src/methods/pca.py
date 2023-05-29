import numpy as np
import matplotlib.pyplot as plt

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        X = training_data
        self.mean = np.mean(X, axis=0)
        # Center the data with the mean
        X_tilde = X - self.mean
        # Create the covariance matrix
        C = X_tilde.T @ X_tilde / X_tilde.shape[0]

        # Compute the eigenvectors and eigenvalues. Hint: look into np.linalg.eigh()
        eigvals, eigvecs = np.linalg.eigh(C)
        # Choose the top d eigenvalues and corresponding eigenvectors. 
        # Hint: sort the eigenvalues (with corresponding eigenvectors) in decreasing order first.
        ind = eigvals.argsort()[::-1][:self.d]

        eg = eigvals[ind]
        self.W = eigvecs[:, ind]

        
        # Compute the explained variance
        exvar = np.sum(eg) / np.sum(eigvals) * 100

        # Compute the explained variance ratio
        explained_variance_ratio = eg / np.sum(eg)
    
        # Compute the cumulative explained variance ratio
        cumulative_variance = np.cumsum(explained_variance_ratio)

        self.cumulative_explain_variance_plot(cumulative_variance)

        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        self.find_principal_components(data)

        X = data
        # Center the data with the mean
        X_tilde = X - self.mean

        # project the data using W
        data_reduced = X_tilde.dot(self.W)
        return data_reduced
        
    def cumulative_explain_variance_plot(self, cumulative_variance):
        # Assuming you have your data in the variable 'X'

        # Compute the cumulative explained variance ratio

        # Plot the cumulative explained variance ratio
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio')
        plt.grid(True)
        plt.show()

