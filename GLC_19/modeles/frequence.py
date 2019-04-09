import numpy as np
import pandas as pd
from classifier import Classifier
from glcdataset import GLCDataset
import sys

class FrequenceModel(Classifier):

    """
    Frequency model. Return all the time the most common specie in the train set.
    
    """
    def __init__(self, dataset,test_size=0.2):
        
        dataset.split_dataset(test_size=test_size)
         # species ranked from most to least common in the training set
        self.all_labels_by_frequency = None

    def fit(self, dataset):

        self.train_frequency=dataset.ytrain.value_counts(normalize=False, sort=True, ascending=False)
        
    def predict(self, dataset, ranking_size=30):

        """
            Allways the same array is return
        """
        predictions = np.array([[self.train_frequency.index[i] for i in range(ranking_size)] for j in range(len(dataset.ytest))])
       
        """
            À optimiser, je crée une matrice de predictions de la taille de test, mais on pourrait juste garder un vecteur
            le problème c'est que ça bug pour le calcul du score.
        """

        return predictions
    
def run(file_csv,dir_tensor,test_size=0.2):
    
    
    print("Frequence model \n")
    df = pd.read_csv(file_csv, sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
                             scnames=df[['glc19SpId','scName']],patches_dir=dir_tensor)

    frequencemodel = FrequenceModel(glc_dataset,test_size=test_size)

    frequencemodel.fit(glc_dataset)
    
    
    
    print("Top30 score:",frequencemodel.top30_score(glc_dataset))
    print("MRR score:", frequencemodel.mrr_score(glc_dataset))

if __name__ == '__main__':
    #examplecsv = '../example_occurrences.csv'
    #dir_tens = '../examples/ex_csv/'
    if len(sys.argv)==3:
    
        run(sys.argv[1],sys.argv[2],test_size=0.2)
    else:
        print("Donnez le fichier csv en premier argument et le dossier des tenseurs en deuxième argument.")
    