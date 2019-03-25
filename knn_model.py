import numpy as np
import pandas as pd
from classifier import Classifier

import scipy.spatial.distance

class KNNModel(Classifier):

     """Simple K-nearest-neighbors model in the environmental
        space.
        Orders classes by decreasing frequency along the neighbors labels, then by lowest distances.
    """
    def __init__(self, window_size=4, k=10):
        """
           :param window_size: the size of the pixel window to calculate the
            mean value for each layer of a tensor
        """
        self.train_vectors = None # vectors for the training dataset
        self.window_size = window_size # window size for vector values
        self.k = k # number of neighbors

    def fit(self, dataset):

        """Builds for each point in the training set a K-dimensional vector,
           K being the number of layers in the env. tensor
           :param dataset: the GLCDataset training set
        """
        self.train_set = dataset
        self.train_vectors = dataset.tensors_to_vectors(self.window_size)

    def predict(self, dataset, ranking_size=30):

        """For each point in the dataset, returns the labels of the 30 closest points in the training set,
           ranked by distance.
           It only keeps the closests training points of different species.
        """
        predictions = []
        test_vectors = dataset.tensors_to_vectors(self.window_size)

        for j in range(len(dataset)):

            vector_j = test_vectors[j]
            # euclidean distances from the test point j to all training points i
            distances = np.array([scipy.spatial.distance.euclidean(vector_j,vector_i)
                                  for vector_i in self.train_vectors
                                 ])
            argsort = np.argsort(distances)
            argsort_k = argsort[:k]
            # build list of species, along k closest points
            y_closest = [self.train_set.get_label(idx) for idx in argsort]

            # get unique labels and their counts, then predicts the species:
            # - first sort the labels by decreasing frequencies
            # - for labels equal in frequency, then sort by closest to further
            unique_labels,counts = np.unique(y_closest,return_counts=True)
            unique_counts, c_counts = np.unique (counts, return_counts=True)

            y_predicted = unique_labels[np.argsort(counts)]
            predictions.append(y_predicted)

        return predictions
