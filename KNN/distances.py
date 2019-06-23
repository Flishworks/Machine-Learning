import numpy as np
def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    #L2 norm
    # L2 norm (euclidean distance)
    ##the inneficient way
    L2 = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            distance = 0
            for k in range(x.shape[0]):
                distance += (abs(y[k])-abs(x[k]))**2
            distance = distance ** .5
            L2[i,j] = distance
    return L2

    #better way
    #return np.sqrt(-2*np.dot(X, Y.T) + np.linalg.norm(X, ord=2, axis=1)[:, np.newaxis]**2 + np.linalg.norm(Y, ord=2, axis=1)[np.newaxis, :]**2)
    #check
    #return sk.metrics.pairwise.euclidean_distances(X, Y)


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    #L1 norm (manhattan distance)
    L2 = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            distance = 0
            for k in range(x.shape[0]):
                distance += abs((abs(y[k])-abs(x[k])))
            L2[i,j] = distance
    return L2

    #check
    #return sk.metrics.pairwise.manhattan_distances(X, Y)

def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    #cosine distance
    #L1
    #X_norm = X/np.sum(np.abs(X),axis=1)[:, np.newaxis] #np.linalg.norm(X, ord=1, axis=1)[:, np.newaxis]
    #Y_norm = Y/np.sum(np.abs(Y),axis=1)[:, np.newaxis] #np.linalg.norm(Y, ord=1, axis=1)[:, np.newaxis]
    #L2
    #X_norm = X/np.linalg.norm(X, ord=2, axis=1)[:, np.newaxis]
    #Y_norm = Y/np.linalg.norm(Y, ord=2, axis=1)[:, np.newaxis]
    #dot = np.dot(X_norm,Y_norm.T)
    #print(dot)
    #another way with one line
    return 1 - np.dot(X, Y.T)/(np.linalg.norm(X, ord=2, axis=1)[:, np.newaxis] * np.linalg.norm(Y, ord=2, axis=1)[np.newaxis, :])

    #check
    #print(sk.metrics.pairwise.cosine_similarity(X, Y))
