import pandas as pd
import numpy as np

def build_environmental_data(df, patches_dir, mean_window_size=None):

    """This function builds a dataset containing all the latitude,
       longitude, and vectors of the environmental tensors associated saved
       in a directory.
       Used to fit to Scikit-Learn models.
       If the environmental tensors are just row vectors (i.e the env. variables
       values at the location) then it loads them in a new dataframe.
       Otherwise, the tensors are flattened as long row vectors;
       that's when the tensors are the env. variables values around the location.
       :param df: the locations dataframe, containing (Latitude,Longitude)
           columns
       :param patches_dir: the directory where the env. patches are saved
       :param mean_window_size: if not None, takes the mean value of each
       raster on the provided window size
       :return: a new dataframe containing the locations concatenated with
           their env. vectors
    """
    # import the names of the environmental variables
    from environmental_raster_glc import raster_metadata

    env_array = list()
    # number of values per channel, 1 if patches are vector
    n_features_per_channel = 1
    for idx in range(len(df)):

        # get the original index used to write the patches on disk
        true_idx = df.index[idx]
        # find the name of the file
        patch_name = patches_dir + '/' + str(true_idx)+'.npy'
        # reads the file
        patch = np.load(patch_name)
        # build the row vector
        lat, lng = df.loc[true_idx,'Longitude'], df.loc[true_idx,'Latitude']

        if mean_window_size:
            try:
                patch = np.array([ ch[ch.shape[0]//2 - mean_window_size//2:
                                      ch.shape[0]//2 + mean_window_size//2,
                                      ch.shape[1]//2 - mean_window_size//2:
                                      ch.shape[1]//2 + mean_window_size//2
                                     ].mean() for ch in patch
                                 ])
                assert(len(patch.shape)==1)
            except IndexError:
                raise Exception("Channels don't have two dimensions!")
        else:
            if len(patch.shape) > 1:
                n_features_per_channel = patch.shape[0]*patch.shape[1]
            elif len(patch.shape) ==2 :
                raise Exception("Channel of dimension one: should only be a scalar\
                                 or a two dimensional array")
        # flatten to build row vector
        env_array.append(np.concatenate(([lat,lng],patch),axis=None))

    rasters_names = sorted(raster_metadata.keys())
    if n_features_per_channel == 1:
        header_env = rasters_names
    else:
        header_env = []
        for name in rasters_names:
            header_env.extend([name+f'__{i}' for i in range(n_features_per_channel)])
    header = ['Latitude','Longitude'] + header_env
    env_df = pd.DataFrame(env_array, columns=header, dtype='float64')
    return env_df

def get_taxref_names(self, y, taxonomic_names):
    """Returns the taxonomic names which corresponds to the list of
       species ids
       :param y: the list of species
       :return: the list of taxonomic names
    """
    return [taxonomic_names[taxonomic_names['glc19SpId']==spid]['taxaName'].iloc[0] for spid in y]

if __name__ == '__main__':

    from IPython.display import display

    # random seed for reproducibility
    np.random.seed(42)

    # working on a subset of Pl@ntNet Trusted: 2500 occurrences
    df = pd.read_csv('../example_occurrences.csv',
                     sep=';', header='infer', quotechar='"', low_memory=True)

    df = df[['Longitude','Latitude','glc19SpId','scName']]
    df = df.dropna(axis=0, how='all') #drop nan lines
    df = df.astype({'glc19SpId': 'int64'})
    # target pandas series of the species identifiers (there are 505 labels)
    target_df = df['glc19SpId']
    # correspondence table between ids and the species taxonomic names
    # (Taxref names with year of discoverie)
    taxonomic_names = pd.read_csv('../../data/occurrences/taxaName_glc19SpId.csv',
                                  sep=';',header='infer', quotechar='"',low_memory=True)

    # glc_dataset = GLCDataset(df[['Longitude','Latitude']], df['glc19SpId'],
    #                          scnames=df[['glc19SpId','scName']],patches_dir='example_envtensors')

    print(len(df), 'occurrences in the dataset')
    print(len(target_df.unique()), 'number of species\n')
    display(df.head(5))
    duplicated_df = df[df.duplicated(subset=['Latitude','Longitude'],keep=False)]
    print(f'There are {len(duplicated_df)} entries observed at interfering locations :')
    display(duplicated_df.head(5))
    env_df = build_environmental_data(df[['Latitude','Longitude']],patches_dir='../examples/ex_csv')
    assert(len(df) == len(env_df))

    X = env_df.values
    y = target_df.values