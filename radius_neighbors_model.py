import numpy as np
import pandas as pd
from classifier import Classifier

from sklearn.neighbors import RadiusNeighborsClassifier

## NEEDS DEBUGGING : REWRITE FIT METHOD TO PREDICT PROBABILITIES
class RadiusNeighborsModel(Classifier):

    """Simple K-nearest-neighbors model in the vectors representation space
        Orders classes by decreasing frequency along the neighbors labels, then by lowest distances.
    """
    def __init__(self, radius=1.0, weights='uniform', p=2, metric='minkowski', how_outliers='most_common', ranking_size=30):
        """
           :param n_neighbors: the number of neighbors for predicting class probabilities
           :param metric: the distance metric used, should be something like
            - 'euclidean' for regular euclidean distance
            - 'manhattan'
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine': the cosinus similarity
        """
        self.radius = radius
        self.weights = weights
        self.p = p
        self.metric = metric
        # how are isolated examples, with no neighbors withing a given radius,
        # treated : return the most common label or return a random label
        self.how_outliers=how_outliers
        self.ranking_size = ranking_size
        # the Scikit-learn K neighbors classifier
        self.clf = RadiusNeighborsClassifier(radius=radius,
                                        weights=weights,
                                        p=p,
                                        metric=metric,
                                        outlier_label=None,
                                        n_jobs=-1
                                        )

        def fit(self, X, y):

            # set the returned labels for outliers examples
            if outlier_label == 'most_common':
                y_unique,counts = np.unique(y, return_counts=True)
                outlier_label = y_unique[np.argmax(counts)]
                self.set_params(outlier_label=outlier_label)
                self.clf.set_params(outlier_label=outlier_label)
            elif outlier_label == 'random':
                self.set_params(outlier_label=None)
                self.clf.set_params(outlier_label=None)

            super().fit(X, y)

        def predict(self, X, y)
        ## TODO : define a predict_proba method since it's not available with
        # the radius neighbors classifier from Scikit-learn
        pass

if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from glcdataset import build_environmental_data

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

    # Evaluate as the average accuracy on two train/split random sample:
    print("Test radius neighbors model, euclidean metric")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = RadiusNeighborsModel(radius=10, weights='uniform',
                                        metric='euclidean',
                                        how_outliers=None)
    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')

    print(classifier.clf.get_params())

if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from glcdataset import build_environmental_data

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

    # Evaluate as the average accuracy on two train/split random sample:
    print("Test radius neighbors model, euclidean metric")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = KNearestNeighborsModel(n_neighbors=150, weights='distance',metric='euclidean')
    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')

    print(classifier.clf.get_params())
