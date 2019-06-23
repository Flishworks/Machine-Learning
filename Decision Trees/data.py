import numpy as np
import os
import csv

def load_data(data_path):
    """
    Associated test case: tests/test_data.py

    Reading and manipulating data is a vital skill for machine learning.

    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of features.

    data_path leads to a csv comma-delimited file with each row corresponding to a
    different example. Each row contains binary features for each example
    (e.g. chocolate, fruity, caramel, etc.) The last column indicates the label for the
    example how likely it is to win a head-to-head matchup with another candy
    bar.

    This function reads in the csv file, and reads each row into two numpy arrays.
    The first array contains the features for each row. For example, in candy-data.csv
    the features are:

    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus

    The second array contains the targets for each row. The targets are in the last
    column of the csv file (labeled 'class'). The first row of the csv file contains
    the labels for each column and shouldn't be read into an array.

    Example:
    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus,class
    1,0,1,0,0,1,0,1,0,1

    should be turned into:

    [1,0,1,0,0,1,0,1,0] (features) and [1] (targets).

    This should be done for each row in the csv file, making arrays of size NxK and Nx1.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size Nx1 containing the 1 feature.
        attribute_names (list): list of strings containing names of each attribute
            (headers of csv)
    """
    raw_data=[]
    with open(data_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            raw_data.append(row)

    num_cols = len(raw_data[1]) #get number of columns
    num_rows = len(raw_data) #get number of rows

    #build attribute and class arrays
    myArray=np.array(raw_data)
    features = myArray[1:,0:num_cols-1].astype(float)
    targets = myArray[1:,num_cols-1].astype(float)
    attribute_names = list(myArray[0,0:num_cols-1])
    return features, targets, attribute_names

def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing, randomly. N points from the data
    sampled for training and (features.shape[0] - N) points for testing. Where N:

        N = int(features.shape[0] * fraction)

    Returns train_features (size NxK), train_targets (Nx1), test_features (size MxK
    where M is the remaining points in data), and test_targets (Mx1).

    Args:
        features (np.array): numpy array containing features for each example
        targets (np.array): numpy array containing labels corresponding to each example.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns
        train_features: subset of features containing N examples to be used for training.
        train_targets: subset of targets corresponding to train_features containing targets.
        test_features: subset of features containing N examples to be used for testing.
        test_targets: subset of targets corresponding to test_features containing targets.
    """
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')

    num_rows = features.shape[0]
    N = int(num_rows * fraction)
    all_rows = np.array(range(num_rows))
    scrambled_rows = np.random.choice(all_rows, size=num_rows, replace=False) #scramble elements from leftover_rows
    train_rows = scrambled_rows[:N]
    test_rows = scrambled_rows[N:]
    train_features = features[train_rows,:]
    train_targets = targets[train_rows]
    test_features = features[test_rows,:]
    test_targets = targets[test_rows]
    return train_features, train_targets, test_features, test_targets
