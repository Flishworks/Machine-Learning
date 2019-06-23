import csv
import numpy as np
import os

def load_movielens_data(data_folder_path):
    """
    The MovieLens dataset is contained at data/ml-100k.zip. This function reads the
    unzipped content of the MovieLens dataset into a numpy array. The file to read in
    is called ```data/ml-100k/u.data``` The description of this dataset is:

    u.data -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of
	          user id | item id | rating | timestamp.
              The time stamps are unix seconds since 1/1/1970 UTC

    Return a numpy array that has size 943x1682, with each item in the matrix (i, j)
    containing the rating user i had for item j. If a user i has no rating for item j,
    you should put 0 for that entry in the matrix.

    Args:
        data_folder_path {str}: Path to MovieLens dataset (given at data/ml-100).
    Returns:
        data {np.ndarray}: Numpy array of size 943x1682, with each item in the array
            containing the rating user i had for item j. If user i did not rate item j,
            the element (i, j) should be 0.
    """
    # This is the path to the file you need to load.
    data_file = os.path.join(data_folder_path, 'u.data')
    data = []
    with open(data_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
             data.append(row[0:3]) #dont need timestanps

    transformed_array = np.zeros((943, 1682))
    for row in np.array(data).astype(int):
        transformed_array[row[0]-1, row[1]-1] = row[2]


    return transformed_array
