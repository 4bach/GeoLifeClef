import numpy as np
import pandas as pd
from classifier import Classifier
import sys
# Scikit-learn validation tools
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import pairwise_distances
#from sklearn.model_selection import train_test_split
from glcdataset import build_environmental_data
from sklearn.preprocessing import StandardScaler
# TODO : RETURN PROBABILITIES IN PREDICT
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

    def predict(self, X, with_proba=False):

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
        return y_predicted
    
    
def run(train_csv, train_tensor,test_csv,test_tensor, metric='euclidean'):
    print("K means\n")
    
    """
        Construction du dataset train.
    """
    df = pd.read_csv(train_csv,sep=';', header='infer', quotechar='"', low_memory=True)
    
    df = df[['Longitude','Latitude','glc19SpId','scName']]\
       .dropna(axis=0, how='all')\
       .astype({'glc19SpId': 'int64'})
     # target pandas series of the species identifiers (there are 505 labels)
    target_df = df['glc19SpId']
    
    # building the environmental data
    env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir=train_tensor)
    X_train = env_df.values
    y_train = target_df.values 
       
    """
        Construction du dataset test.
    """
    df = pd.read_csv(test_csv,sep=';', header='infer', quotechar='"', low_memory=True)
    
    df = df[['Longitude','Latitude','glc19SpId','scName']]\
       .dropna(axis=0, how='all')\
       .astype({'glc19SpId': 'int64'})
     # target pandas series of the species identifiers (there are 505 labels)
    target_df = df['glc19SpId']
    
    # building the environmental data
    env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir=test_tensor)
    X_test = env_df.values
    y_test = target_df.values
    
    """
        Entrainement modèle.
    """
    # Standardize the features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    classifier = VectorModel(metric=metric)
    classifier.fit(X_train,y_train)
    """
         Évaluation et Calcul de score.
    """
    
    y_predicted = classifier.predict(X_test)
    print(f'Top30 score:{classifier.top30_score(y_predicted, y_test)}')
    print(f'MRR score:{classifier.mrr_score(y_predicted, y_test)}')
    print('Params:',classifier.get_params())

    
if __name__ == '__main__':

    # examplecsv = '../example_occurrences.csv'
    # dir_tens = '../examples/ex_csv/'
    if len(sys.argv) == 3:

        run(sys.argv[1], sys.argv[2],'../../data/occurrences/test.csv','/local/karmin/env_tensors/test/')
    else:
        run('../examples/example_occurrences_train.csv', '../examples/ex_csv_train/','../examples/example_occurrences_test.csv','../examples/ex_csv_test/')
        # print("Donner le fichier csv en premier argument et le dossier des tenseurs en deuxième argument.")
   