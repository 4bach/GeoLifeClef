
import numpy as np
import pandas as pd
from classifier import Classifier
import sys
import scipy.spatial.distance
from glcdataset import GLCDataset

class VectorModel(Classifier):

    """Simple vector model based on nearest-neighbors in the environmental
       space
       It can be seen as a 1-nearest-neighbor classifier, applied to multi-class:
       the nearest neighbor predicts the 1st ranked label, the 2nd nearest predicts
       the 2nd ranked label and so on.
       INCONVENIENTS ARE:
       - prone to overfitting? because based on 1-nearest-neighbor
       - inconvenients of space/environment representation
       - slow in prediction (bcause non-parametric clf)
       - doesn't use filtering methods
       UPGRADES TO THIS MODEL:
       - a k-nn classifier over *k nearest neighbors* in space and/or environment:
         ordering classes by decreasing frequency along the neighbors labels (then by lowest distances)
       - and : filtering neighbors by location first, then distances in the environmental space
       - or, inversely: filtering neighbors by environment, then distances in the environmental space
       - use of co-occurences? Co-occurences matrix of species in a s-size location window?
         => revealed groups/patterns of species aggregates over space/environment
         => k-means over groups of species (event. filtered in space or environment), both hard and soft (GMM?)
         => logistic regression, or svm in this new representation space (hybrid clustering/classif)
    """
    def __init__(self,dataset,window_size=4,test_size=0.2):
        """
           :param window_size: the size of the pixel window to calculate the
            mean value for each layer of a tensor
        """
        dataset.split_dataset(test_size=test_size)
        self.window_size = window_size

    def fit(self, dataset):

        """Builds for each point in the training set a K-dimensional vector,
           K being the number of layers in the env. tensor
           :param dataset: the GLCDataset training set
        """
        #self.train_set = dataset
        self.train_vectors = self.tensors_to_vectors(dataset,dataset.xtrain)

    def predict(self, dataset, ranking_size=30):

        """For each point in the dataset, returns the labels of the 30 closest points in the training set.
           It only keeps the closests training points of different species.
           Modification de l'algo pour calculer un produit matriciel 
        """
        test_vectors = self.tensors_to_vectors(dataset,dataset.xtest)
        d = scipy.spatial.distance.cdist(test_vectors,self.train_vectors , metric='euclidean')
        argsort = np.array([np.argsort(d[i])[:ranking_size] for i in range(np.shape(d)[0])])        
        prediction = np.array([[dataset.ytrain[dataset.ytrain.index[argsort[j][i]]] for i in range(np.shape(argsort)[1])] for j in range(np.shape(argsort)[0])])
        return prediction
    
    def tensors_to_vectors(self,dataset,X):

        """Builds a vector out of a env. tensor for each datapoint
           :param window_size: the size of the pixel window to calculate the
            mean value for each layer of a tensor
           :return: the list of vectors for each datapoint
        """
        vectors = np.zeros((len(X),33))
        i=0
        for index in X.index:
            x_c, y_c = 32,32 # (64,64)/2 
            vectors[i]= np.array([layer[x_c - self.window_size//2: x_c + self.window_size//2,y_c - self.window_size//2: y_c + self.window_size//2].mean() for layer in dataset[index]['patch']])
            i+=1
        return vectors
    
    
def run(file_csv,dir_tensor,test_size=0.2):
    
    
    print("Vector model \n")
    df = pd.read_csv(file_csv, sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
                             scnames=df[['glc19SpId','scName']],patches_dir=dir_tensor)

    vectormodel = VectorModel(glc_dataset,window_size=4)
    vectormodel.fit(glc_dataset)
    print("Top30 score:",vectormodel.top30_score(glc_dataset))
    print("MRR score:", vectormodel.mrr_score(glc_dataset))
    vectormodel = VectorModel(glc_dataset,window_size=10)
    vectormodel.fit(glc_dataset)
    print("Top30 score:",vectormodel.top30_score(glc_dataset))
    print("MRR score:", vectormodel.mrr_score(glc_dataset))
    
if __name__ == '__main__':
    #examplecsv = '../example_occurrences.csv'
    #dir_tens = '../examples/ex_csv/'
    if len(sys.argv)==3:
    
        run(sys.argv[1],sys.argv[2],test_size=0.1)
    else:
        run('../example_occurrences.csv','../examples/ex_csv/',test_size=0.1)
        #print("Donnez le fichier csv en premier argument et le dossier des tenseurs en deuxième argument.")

   