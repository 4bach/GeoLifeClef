
import numpy as np
import pandas as pd
from classifier import Classifier

# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import pairwise_distances

class VectorModel(Classifier):

    """Simple vector model based on nearest-neighbors in the vector space.
       Returns the nearest-neighbor class, then the 2-nd nearest neighbor class, and so on
       INCONVENIENTS ARE:
       - prone to overfitting? because based on 1-nearest-neighbor
       - slow in prediction, because instance-based and quadratic
       - doesn't use filtering methods

       UPGRADES TO THIS MODEL:

       - a k-nn classifier over *k nearest neighbors* in the vector space :
         ordering classes by decreasing frequency along the neighbors labels (then by lowest distances)

       - and : filtering neighbors by location first, then distances in the environmental space
       - or, inversely: filtering neighbors by environment, then distances in the environmental space
       - use of co-occurences? Co-occurences matrix of species in a s-size location window?
         => revealed groups/patterns of species aggregates over space/environment
         => k-means over groups of species (event. filtered in space or environment), both hard and soft (GMM?)
         => logistic regression, or svm in this new representation space (hybrid clustering/classif)

      - different distances metrics
    """
    def __init__(self, metric='euclidean', ranking_size=30):
        """
        :param metric: the distance metric used, should be something like
            - 'euclidean' for regular euclidean distance
            - 'manhattan'
            - 'haversine' for distances between (latitude,longitude) points only
            - 'cosine'
        """
        self.metric = metric
        self.ranking_size = ranking_size

    def fit(self, X, y):

        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):

        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        # compute all distances, in parallel if possible
        all_distances = pairwise_distances(X, self.X_,
                                           metric=self.metric, n_jobs=-1)
        # get index of the sorted points' distances
        all_argsorts = [np.argsort(dst) for dst in all_distances]
        y_predicted = list()
        # selecting closests points distinct labels
        for argsort in all_argsorts:

            # get closests labels
            y_closest = self.y_[argsort]
            # predict distinct labels
            y_found = set()
            y_pred = list()
            for y in y_closest:
                if len(y_pred)>= self.ranking_size:
                    break
                if y not in y_found:
                  y_pred.append(y)
                  y_found.add(y)

            y_predicted.append(y_pred)
            # THIS DOESN'T WORK: don't know why it return duplicates
            # y_pred = self.y_[np.sort(y_indexes)][:self.ranking_size]
        return y_predicted

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
    print("Test vector model, euclidean metric")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    classifier = VectorModel(metric='euclidean')
    classifier.fit(X_train,y_train)
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')
