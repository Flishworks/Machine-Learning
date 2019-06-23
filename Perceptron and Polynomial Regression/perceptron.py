import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    #seperate circular data
    means = np.mean(features, axis = 0)
    features = features - means
    #transform
    for i,x in enumerate(features):
        if (x[0]**2+x[1]**2)**.5 < 50:
            features[i,1] = 100
        else:
            features[i,1] = -100
    return features

def transform_data_xor(features):
    #seperate XOR style data
    #center at origin
    x = features
    means = np.mean(x, axis = 0)
    x = x - means
    #transform
    row_product = np.prod(x, axis = 1)
    return np.absolute(x)*row_product.reshape(len(row_product),1)


class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the
        line are one class and points on the other side are the other class.

        """
        self.max_iterations = max_iterations
        self.w = 0


    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for
        the perceptron learning algorithm:

            begin initialize weights
                while not converged or not exceeded max_iterations
                    for each example in features
                        if example is misclassified using weights
                        then weights = weights + example * label_for_example
                return weights
            end

        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm.

        Args:
            max_iterations (int): the perceptron learning algorithm stops after
            this many iterations if it has not converged.
        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """

        self.w = np.array([0,0,0])
        ones = np.ones(len(features[:,0]))
        features = np.concatenate((ones.reshape(len(ones),1),features),axis=1)
        w = np.array([1,1,1])

        #perform perceptron fitting
        iterations = 0
        while np.any(self.w != w) & (iterations < self.max_iterations):
            iterations += 1
            self.w = w
            for i, example in enumerate(features):
                if np.matmul(example, w.T)/np.absolute(np.matmul(example, w.T)) != targets[i]:
                    w = w + example * targets[i]

        if iterations == self.max_iterations:
            print("Max iterations reached without convergence.")
        else:
            print("Solution found in ", iterations, " iterations.")
            

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        ones = np.ones(len(features[:,0]))
        features = np.concatenate((ones.reshape(len(ones),1),features),axis=1)
        return np.matmul(features,self.w.T)/np.absolute(np.matmul(features,self.w.T))

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        self.fit(features, targets)
        predicted = self.predict(features)
        x2 = np.linspace(np.min(features), np.max(features), 100).reshape(100,1)
        ones = np.ones(100).reshape(100,1)
        x2 = np.concatenate((ones, x2), axis=1)

        #convert to y=mx+b format
        w2 = np.array([-1*self.w[0]/self.w[2],-1*self.w[1]/self.w[2]])
        y2=np.matmul(x2, w2)
        plt.scatter(features[:,0], features[:,1], c = targets)
        plt.plot(x2[:,1], y2)
        plt.savefig("visualization.svg", format="svg")

    def visualize_realtime(self, features, targets):
        """
        This plots each iteration and waits for user input to continue
        """
        self.w = np.array([0,0,0])
        ones = np.ones(len(features[:,0]))
        x2 = np.concatenate((ones.reshape(len(ones),1),features),axis=1)
        w = np.array([1,1,1])

        #perform perceptron fitting
        iterations = 0
        while np.any((self.w != w)) & (iterations < self.max_iterations):
            iterations += 1
            self.w = w
            for i, example in enumerate(x2):
                if np.matmul(example, w.T)/np.absolute(np.matmul(example, w.T)) != targets[i]:
                    w = w + example * targets[i]

            x3 = np.linspace(np.min(features), np.max(features), 100).reshape(100,1)
            ones = np.ones(100).reshape(100,1)
            x3 = np.concatenate((ones, x3), axis=1)
            #convert to y=mx+b format
            w2 = np.array([-1*self.w[0]/self.w[2],-1*self.w[1]/self.w[2]])
            y2=np.matmul(x3, w2)
            plt.scatter(features[:,0], features[:,1], c = targets)
            plt.plot(x3[:,1], y2)
            #plt.savefig("visualization.svg", format="svg")
            plt.show()
            input("Press Enter to continue...")
