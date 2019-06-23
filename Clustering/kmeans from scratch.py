import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        self.means = features[np.random.choice(features.shape[0], 2, replace=False), :]
        new_means = np.copy(self.means)
        print(self.means)
        distances = self.euclidean_distances(features, self.means)
        label_vector = np.argmin (distances, axis=-1)

        new_means[0] = np.mean(features[label_vector == 0], axis = 0)
        new_means[1] = np.mean(features[label_vector == 1], axis = 0)
        while (np.all(new_means!=self.means)):
            self.means = np.copy(new_means)
            print(self.means)
            distances = self.euclidean_distances(features, self.means)
            label_vector = np.argmin(distances, axis=-1)
            new_means[0] = np.mean(features[label_vector == 0], axis = 0)
            new_means[1] = np.mean(features[label_vector == 1], axis = 0)

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        distances = self.euclidean_distances(features, self.means)
        label_vector = np.argmin(distances, axis=1)
        return (label_vector)

    def euclidean_distances(self, X, Y):
        """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
        and Y (shape NxK). The output of this function is a matrix of shape MxN containing
        the Euclidean distance between two rows.

        Arguments:
            X {np.ndarray} -- First matrix, containing M examples with K features each.
            Y {np.ndarray} -- Second matrix, containing N examples with K features each.

        Returns:
            D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.

            format of returned array: columns (x dimension) represent index of rows in matrix Y, rows (Y dimension) represents index of rows in matrix X.
                for example,
                returned array:
                    [d(y1,x1), d(y2,x1)]
                    [d(y1,x2), d(y2,x2)]
        """
        #better way
        return np.sqrt(-2*np.dot(X, Y.T) + np.linalg.norm(X, ord=2, axis=1)[:, np.newaxis]**2 + np.linalg.norm(Y, ord=2, axis=1)[np.newaxis, :]**2)

    
