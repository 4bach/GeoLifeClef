import numpy as np
import pandas as pd
import sys
from classifier import Classifier
from glcdataset import GLCDataset

class RandomModel(Classifier):

    """Simple vector model based on nearest-neighbors in the environmental
       space
    """
    def __init__(self,dataset,test_size=0.2):

        dataset.split_dataset(test_size=test_size)

    def fit(self, dataset):

        self.all_labels = pd.Series(dataset.ytrain.unique())
        
        
        
    def predict(self, dataset, ranking_size=30):

        
        predictions = np.array([np.random.choice(self.all_labels.values, size=ranking_size) for j in range(len(dataset.ytest))])

        return predictions




def run(file_csv,dir_tensor,test_size=0.2):
    
    
    print("Random model \n")
    df = pd.read_csv(file_csv, sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
                             scnames=df[['glc19SpId','scName']],patches_dir=dir_tensor)

    randommodel = RandomModel(glc_dataset)

    randommodel.fit(glc_dataset)
    
  

    print("Top30 score:",randommodel.top30_score(glc_dataset))
    print("MRR score:", randommodel.mrr_score(glc_dataset))
    
if __name__ == '__main__':
    #examplecsv = '../example_occurrences.csv'
    #dir_tens = '../examples/ex_csv/'

    if len(sys.argv)==3:
    
        run(sys.argv[1],sys.argv[2],test_size=0.2)
    else:
        print("Donnez le fichier csv en premier argument et le dossier des tenseurs en deuxième argument.")
