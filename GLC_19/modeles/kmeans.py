
import numpy as np
import pandas as pd
import sys
import scipy.spatial.distance
from classifier import Classifier
from glcdataset import GLCDataset
from sklearn.cluster import KMeans

class Kmeans(Classifier):

    """
    K-Means Model
    
    
    """
    def __init__(self,dataset,windows_size=4,nb_cluster=4,test_size=0.2):
        """
            param window_size: the size of the pixel window to calculate the
            mean value for each layer of a tensor
           param nb_cluster : number of clusters for the k means.
           test_size: percentage of test in the dataset.
        """
        dataset.split_dataset(test_size=test_size)
        self.nb_cluster = nb_cluster
        self.windows_size = windows_size
    
    def tensors_to_vectors(self,dataset,X):

        """Builds a vector out of a env. tensor for each datapoint
          
           :return: the list of vectors for each datapoint
        """
        vectors = np.zeros((len(X),33))
        i=0
        for index in X.index:
            x_c, y_c = 32,32 # (64,64)/2 
            vectors[i]= np.array([layer[x_c - self.window_size//2: x_c + self.window_size//2,y_c - self.window_size//2: y_c + self.window_size//2].mean() for layer in dataset[index]['patch']])
            i+=1
        
        return vectors
    def species_vectors(self,dataset):
        """
            Construct a characteristic vector for each species.
            We do the average of all the spatial vectors where the species appears.
        """
        done = []
        vectors = self.tensors_to_vector(dataset,dataset.xtrain)
        self.species_matrix = []
        for i in range(len(dataset.ytrain)):
            if dataset.ytrain.values[i] not in done:
                done.append(dataset.ytrain.values[i])
                mask = pd.Index(dataset.ytrain).get_loc(dataset.ytrain.values[i])
                self.species_matrix[i] = np.mean(vectors[mask],axis=0)
        
        self.species_matrix = np.array(self.species_matrix)
    def fit(self, dataset):

        """
            Now we have our training species_vectors, we can cluster it with the sklearn methods KMeans.
        """
        self.set_frequence(dataset)
        self.species_vectors(dataset)
        self.kmeans = KMeans(n_clusters=self.nb_cluster, random_state=0).fit(self.species_matrix)
        

    def predict(self, dataset, ranking_size=30):

        """
            Transform the test set into environnement vectors, then it predict in which clusters it goes.
        """
        vectors = self.tensors_to_vector(dataset,dataset.xtest)
        prediction = self.kmeans.predict(vectors)
        
    
    def set_frequence(self,dataset):
        """
            Return a series with all the labels of the training set and the number of time it occurres. 
        """
        self.train_frequency=dataset.ytrain.value_counts(normalize=False, sort=True, ascending=False)
    
    
def run(file_csv,dir_tensor,test_size=0.2):
    
    
    print("K means \n")
    df = pd.read_csv(file_csv, sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
                             scnames=df[['glc19SpId','scName']],patches_dir=dir_tensor)

    kMeans = Kmeans(glc_dataset,window_size=4,nb_cluster=8,test_size=0.2)
    kMeans.fit(glc_dataset)
    print("Top30 score:",kMeans.top30_score(glc_dataset))
    print("MRR score:", kMeans.mrr_score(glc_dataset))
    
    
if __name__ == '__main__':
    #examplecsv = '../example_occurrences.csv'
    #dir_tens = '../examples/ex_csv/'
    if len(sys.argv)==3:
    
        run(sys.argv[1],sys.argv[2],test_size=0.1)
    else:
        run('../example_occurrences.csv','../examples/ex_csv/',test_size=0.1)
        #print("Donnez le fichier csv en premier argument et le dossier des tenseurs en deuxième argument.")
