import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO
    CLASSES (binary).

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    true = predictions[actual[:] == predictions[:]]
    false = predictions[actual[:] != predictions[:]]

    confusion = np.array([len(true[true[:] == 0]),len(false[false[:] == 1]), len(false[false[:] == 0]), len(true[true[:] == 1])])
    return confusion.reshape(2,2)

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    assert len(predictions) > 0, "prediction vector cannot be of length zero"

    return len(predictions[actual[:] == predictions[:]])/len(predictions)

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion = confusion_matrix(actual, predictions)
    #[true_negatives, false_positives],
    #[false_negatives, true_positives]
    tp = confusion[1,1]
    tp = 0 if np.isnan(tp) else tp
    fp = confusion[0,1]
    fp = 0 if np.isnan(fp) else fp
    fn = confusion[1,0]
    fn = 0 if np.isnan(fn) else fn
    if (tp+fp)>0:
        precision = tp/(tp+fp)
    else:
        precision = 0
    if (tp+fn)>0:
        recall = tp/(tp+fn)
    else:
        recall = 0

    return precision, recall

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)
    if (precision+recall) > 0:
        return 2*precision*recall/(precision+recall)
    else:
        return 0
