import numpy as np

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """

        self.most_common_class = 1 if targets.sum()/len(targets)>.5 else 0
        #print(counts[counts[:] == np.argmax(counts)])

        #counts = np.histogram(targets, bins = 10)
        #print(counts)
        #print(counts[1][np.argmax(counts[0])])
        #if (np.argmax(counts[0]) == len(counts[0])-1):
        #    self.most_common_class = counts[1][np.argmax(counts[0])+1]
        #else:
        #    self.most_common_class = counts[1][np.argmax(counts[0])]

    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """

        #if self.most_common_class >.5:
        #    predictions = np.ones((len(data)))
        #else:
        #    predictions = np.zeros((len(data)))

        return np.ones([len(data)])*self.most_common_class
