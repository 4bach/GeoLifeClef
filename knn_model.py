import numpy as np
import pandas as pd
from classifier import Classifier

from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighborsModel(Classifier):

    """Simple K-nearest-neighbors model in the vectors representation space
        Orders classes by decreasing frequency along the neighbors labels, then by lowest distances.
    """
    def __init__(self, n_neighbors=5, weights='uniform', p=2, metric='minkowski', ranking_size=30):
        """
           :param n_neighbors: the number of neighbors for predicting class probabilities
           :param metric: the distance metric used, should be something like
            - 'euclidean' for regular euclidean distance
            - 'manhattan'
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine': the cosinus similarity
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.ranking_size = ranking_size
        # the Scikit-learn K neighbors classifier
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                            metric=metric,
                                            weights=weights,
                                            n_jobs=-1)

    # def predict(self, dataset, ranking_size=30):

    #     """For each point in the dataset, returns the labels of the 30 nearest points in the training set,
    #        ranked by distance.
    #        It only keeps the nearests training points of different species.
    #     """
    #     predictions = []
    #     test_vectors = dataset.tensors_to_vectors(self.window_size, self.repr_space)

    #     for j in range(len(dataset)):

    #         vector_j = test_vectors[j]
    #         # euclidean distances from the test point j to all training points i

    #         distances = np.array([np.sqrt(np.sum(vect_j-vect_i)) for vect_i in self.train_vectors])

    #         # consider points which are the k nearest neighbors only
    #         argsort = np.argsort(distances)[: k]
    #         # build list of unique labels, along k nearest points
    #         y_nearest = self.train_set.labels[ argsort ]
    #         y_unique_nearest, counts = np.unique(y_nearest, return_counts=True)
    #         for iy in np.argsort(counts):
    #             same_counts = counts[counts == counts[iy]]

    #         # get unique labels and their counts, then predicts the species:
    #         # - first sort the labels by decreasing frequencies
    #         # - for labels equal in frequency, then sort by nearest to further
    #         tosort = np.array([(counts[i],argsort[i] for i in range(len(y_unique_nearest)))],
    #                           dtype=[('count','i4'),('dist','f8')])
    #         for i in range(len(y_unique_nearest)):
    #             tosort[i] = (argsort)
    #         tosort =
    #         tosort = np.array([(count,dist) for count,dist in zip()])
    #         y_sorted = [y for y in y_unique_nearest[np.argsort(counts)]][:ranking_size]

    #         unique_labels,counts = np.unique(y_nearest,return_counts=True)
    #         unique_counts, c_counts = np.unique (counts, return_counts=True)

    #         y_predicted = unique_labels[np.argsort(counts)]
    #         predictions.append(y_predicted)

    #     return predictions

if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from glcdataset import build_environmental_data
    from sklearn.preprocessing import StandardScaler

    # for reproducibility
    np.random.seed(42)

    # working on a subset of Pl@ntNet Trusted: 2500 occurrences
    df = pd.read_csv('example_occurrences.csv',
                 sep=';', header='infer', quotechar='"', low_memory=True)

    df = df[['Longitude','Latitude','glc19SpId','scName']]\
           .dropna(axis=0, how='all')\
           .astype({'glc19SpId': 'int64'})

    # target pandas series of the species identifiers (there are 505 labels)
    target_df = df['glc19SpId']

    # building the environmental data
    env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir='example_envtensors')
    X = env_df.values
    y = target_df.values
    # Standardize the features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Evaluate as the average accuracy on one train/split random sample:
    print("Test KNN model")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = KNearestNeighborsModel(n_neighbors=150, weights='distance',metric='euclidean')
    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')
    print('Params:',classifier.get_params())

    print("Example of predict proba:")
    print(f"occurrence:\n{X_test[12]}")
    y_pred, y_probas = classifier.predict(X_test[12].reshape(1,-1), return_proba=True)
    print(f'predicted labels:\n{y_pred}')
    print(f'predicted probas:\n{y_probas}')

