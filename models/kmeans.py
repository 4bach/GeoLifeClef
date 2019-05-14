import numpy as np
import pandas as pd
import sys
from classifier import Classifier
from glcdataset import GLCDataset
from sklearn.cluster import KMeans


class Kmeans(Classifier):

    """
    K-Means Model
    
    
    """
    def __init__(self, dataset, window_size=4, nb_cluster=4, test_size=0.2):
        """
            param window_size: the size of the pixel window to calculate the
            mean value for each layer of a tensor
           param nb_cluster : number of clusters for the k means.
           test_size: percentage of test in the dataset.
        """
        #dataset.split_dataset(test_size=test_size)
        
        self.nb_cluster = nb_cluster
        self.window_size = window_size
        self.species_matrix = []
        self.train_frequency = []
        self.kmeans = []

    def tensors_to_vectors(self, dataset):

        """Builds a vector out of a env. tensor for each datapoint
            On a besoin de dataset pour prendre les bons patchs.
           X -> soit dataset.train soit dataset.test 
           :return: the list of vectors for each datapoint
        """
        vectors = np.zeros((len(dataset.data),33))
        i=0
        for index in dataset.data.index:
            x_c, y_c = 32,32 # (64,64)/2 
            vectors[i] = np.array([layer[x_c - self.window_size//2: x_c + self.window_size//2, y_c - self.window_size//2: y_c + self.window_size//2].mean() for layer in dataset[index]['patch']])
            i += 1
        return vectors

    def species_vectors(self, dataset):
        """
            Construct a characteristic vector for each species.
            We do the average of all the spatial vectors where the species appears.
            dataset -> train
        """
        nb_unique_spec = dataset.labels.nunique()
        self.done = np.zeros((nb_unique_spec,)) # Va également nous servir à retrouver les ID d'espèces 
        vectors = self.tensors_to_vectors(dataset)
        self.species_matrix = np.zeros((nb_unique_spec,33))
        j = 0
        for i in range(len(dataset.labels)):
            if dataset.labels.values[i] not in self.done:
                self.done[j] = dataset.labels.values[i]
                mask = pd.Index(dataset.labels).get_loc(dataset.labels.values[i])
                self.species_matrix[j] = np.mean(vectors[mask], axis=0)
                j += 1

    def fit(self, dataset):

        """
            Now we have our training species_vectors, we can cluster it with the sklearn methods KMeans.
        """
        self.set_frequence(dataset)
        self.species_vectors(dataset)
        self.which_clusters = KMeans(n_clusters=self.nb_cluster, random_state=0).fit_predict(self.species_matrix)
        self.kmeans = KMeans(n_clusters=self.nb_cluster, random_state=0).fit(self.species_matrix)
        return (self.which_clusters,self.kmeans,self.done)
    
    def predict(self, dataset, ranking_size=30):

        """
            Transform the test set into environnement vectors, then it predict in which clusters it goes.
            dataset -> test
        """
        prediction = np.zeros((len(dataset.data),ranking_size))
        vectors = self.tensors_to_vectors(dataset)
        self.cluster_test = self.kmeans.predict(vectors)
        for i in range(len(prediction)):
            clust = self.cluster_test[i] # On récupère le cluster 
            species = self.done[np.where(self.which_clusters==clust)]# On récupère les indices ( donc les espèces) qui ont ce cluster.
            for j in range(len(species)):
                prediction[i][j] = species[j]
        return prediction
        """
            Avec which_clusters on a pour chaque espèce(du train), dans quel cluster il est. 
            prediction nous donne les clusters pour les nouvelles espèces 
            maintenant il faut selectionner les 30 espèces les plus fréquentes du cluster.
            problème: s'il y a moins que 30 espèces ? Prendre toutes les espèces du cluster et
            ensuite prendre au hasard
            s'il y a plus ? Prendre par ordre de fréquence.
            problème : dans which cluster on a pas les ID des espèces 
            
        """

    def set_frequence(self, dataset):
        """
            Return a series with all the labels of the training set and the number of time it occurres. 
            dataset -> train
        """
        self.train_frequency = dataset.labels.value_counts(normalize=False, sort=True, ascending=False)
    
    
def run(train_csv, train_tensor,test_csv,test_tensor, window_size=4, nb_cluster=8, test_sizes=0.2):
    print("K means\n")
    
    """
        Construction du dataset train.
    """
    df = pd.read_csv(train_csv, sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset_train = GLCDataset(df[['Longitude', 'Latitude']], df['glc19SpId'], scnames=df[['glc19SpId', 'scName']], patches_dir=train_tensor)
    """
        Construction du dataset test.
    """
    df = pd.read_csv(test_csv, sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset_test = GLCDataset(df[['Longitude', 'Latitude']], df['glc19SpId'], scnames=df[['glc19SpId', 'scName']], patches_dir=test_tensor)
    
    """
        Entrainement modèle.
    """
    
    kmeans = Kmeans(glc_dataset_train, window_size=4, nb_cluster=8, test_size=0.2)
    kmeans.fit(glc_dataset_train)
    """
        Calcul de score.
    """
    print('Top30 score:', kmeans.top30_score(glc_dataset_test))
    #print('MRR score:', kmeans.mrr_score(glc_dataset))
    
    
if __name__ == '__main__':

    # examplecsv = '../example_occurrences.csv'
    # dir_tens = '../examples/ex_csv/'
    if len(sys.argv) == 3:

        run(sys.argv[1], sys.argv[2],'../../data/occurrences/test.csv','/local/karmin/env_tensors/test/')
    else:
        run('../examples/example_occurrences_train.csv', '../examples/ex_csv_train/','../examples/example_occurrences_test.csv','../examples/ex_csv_test/')
        # print("Donner le fichier csv en premier argument et le dossier des tenseurs en deuxième argument.")
