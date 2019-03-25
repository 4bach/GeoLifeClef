from frequence import FrequenceModel
from vector_model import VectorModel
from randommodel import RandomModel
from glcdataset import GLCDataset
import pandas as pd

"""
    Fichier pour tester nos différents modèles et évaluer nos performances. 
    
"""
if __name__ == '__main__':
    print("Vector model tested on train set\n")
    df = pd.read_csv('../../data/occurrences/PL_trusted.csv', sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')
    df = df.astype({'glc19SpId': 'int64'})
    glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
                             scnames=df[['glc19SpId','scName']],patches_dir='/local/karmin/env_tensors/PL_trusted')

    
    
    vectormodel = VectorModel(window_size=4)

    vectormodel.fit(glc_dataset)
    #predictions = vectormodel.predict(glc_dataset)
    #scnames = vectormodel.train_set.scnames
    #for idx in range(4):

        #y_predicted = predictions[idx]
        #print("Occurrence:", vectormodel.train_set.data.iloc[idx].values)
        #print("Observed specie:", scnames.iloc[idx]['scName'])
        #print("Predicted species, ranked:")

        #print([scnames[scnames.glc19SpId == y]['scName'].iloc[0] for y in y_predicted[:10]])
        #print('\n')

    print("Top30 score:(train)",vectormodel.top30_score(glc_dataset))
    print("MRR score: (train)", vectormodel.mrr_score(glc_dataset))
    print("Cross validation score top 30:", vectormodel.cross_validation(glc_dataset, 4, shuffle=False, evaluation_metric='top30'))
    print("\nCross validation score MRR:", vectormodel.cross_validation(glc_dataset, 4, shuffle=False, evaluation_metric='mrr'))
   
    # FREQUENCE 
    print("\nFrequence model tested \n")
    
    frequencemodel = FrequenceModel()

    frequencemodel.fit(glc_dataset)
    print("Top30 score:(train)",frequencemodel.top30_score(glc_dataset))
    print("MRR score:(train)", frequencemodel.mrr_score(glc_dataset))
    print("Cross validation score: MRR", frequencemodel.cross_validation(glc_dataset, 4, shuffle=False, evaluation_metric='mrr'))
    print("\nCross validation score: Top30", frequencemodel.cross_validation(glc_dataset, 4, shuffle=False, evaluation_metric='top30'))
    
    # RANDOM
    
    print("Top30 score:",randommodel.top30_score(glc_dataset))
    print("MRR score:", randommodel.mrr_score(glc_dataset))
    print("Cross validation score:", randommodel.cross_validation(glc_dataset, 4, shuffle=False, evaluation_metric='mrr'))
    
    
    
    
    