import numpy as np
import matplotlib.pyplot as plt
from src.utils import accuracy_fn


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters
    
    def _compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.
        
        Arguments:    
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """

        N = data.shape[0]
        K = centers.shape[0]

        distances = np.zeros((N, K))
        for k in range(K):
            center = centers[k]
            distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))
            
        return distances
    
    def _find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.
        
        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """

        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments
    
    def _compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments: 
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        
        centers = np.zeros((self.K, data.shape[1]))
        for k in range(self.K):
            cluster_points = data[cluster_assignments == k]
            mean = np.mean(cluster_points, axis=0)
            centers[k,:] = mean
        return centers



    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """

        random_idx = np.random.permutation(data.shape[0])[:self.K]
        centers = data[random_idx[:self.K]]

        for i in range(max_iter):
            if ((i+1) % 10 == 0):
                print(f"Iteration {i+1}/{max_iter}...")
            old_centers = centers.copy()  # keep in memory the centers of the previous iteration
            
            
            distances = self._compute_distance(data, centers)
            cluster_assignments = self._find_closest_cluster(distances)
            centers = self._compute_centers(data, cluster_assignments)

            # End of the algorithm if the centers have not moved (hint: use old_centers and look into np.all)
            if np.all(old_centers == centers):
                print(f"K-Means has converged after {i+1} iterations!")
                break
    
        return centers, cluster_assignments
    
    def _assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments: 
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns: 
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """

        cluster_center_label = np.zeros(centers.shape[0])
        for i in range(len(centers)):
            label = np.argmax(np.bincount(true_labels[cluster_assignments == i]))
            cluster_center_label[i] = label
        return cluster_center_label
    
    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        self.final_centers, cluster_assignments = self.k_means(training_data)

        self.cluster_center_label = self._assign_labels_to_centers(self.final_centers, cluster_assignments, training_labels)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        distances = self._compute_distance(test_data, self.final_centers)
        cluster_assignments = self._find_closest_cluster(distances)
        
        pred_labels = self.cluster_center_label[cluster_assignments]
        return pred_labels

    def graph(number_of_clusters, xtrain, ytrain, xtest, ytest, y_pred):
        """
        Returns a graph of the number of clusters vs the accuracy of the model.
        The goal is to find the best K for K-means clustering.
        
        Arguments:
            number_of_clusters: the number of clusters to test
            xtrain: training data of shape (N,D)
            ytrain: labels of shape (N,)
            xtest: test data of shape (N,D)
            ytest: labels of shape (N,)
            y_pred: predicted labels of shape (N,)
        """
        accuracies = []
        for k in range(1, number_of_clusters):
            KmeansObj = KMeans(K=k)
            KmeansObj.fit(xtrain, ytrain)
            y_pred = KmeansObj.predict(xtest)
            acc = accuracy_fn(ytest, y_pred)
            accuracies.append(acc)
        print("Best K: ", np.argmax(accuracies), "with accuracy: ", np.max(accuracies))
        plt.plot(range(1, number_of_clusters), accuracies)
        plt.title('Validation for Kmeans')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Accuracy')
        plt.show()
        pass