import os
import sys
sys.path.extend(['.', '..'])
import pandas as pd
import numpy as np

from environmental_raster_glc import PatchExtractor, raster_metadata

# This script exports the patches on disk, in command line mode
# See the readme for how to use

if __name__ == '__main__':
    """Exports the patches on disk"""

    import argparse

    parser = argparse.ArgumentParser(description='extract environmental patches to disk')
    parser.add_argument('rasters', type=str, help='the path to the raster directory')
    parser.add_argument('dataset', type=str, help='the dataset in CSV format')
    parser.add_argument('destination', type=str,
                        help='The directory where the patches will be exported')

    parser.add_argument('--size', dest='size', type=int, help='size of the final patch (default : 64)', default=64)
    parser.add_argument('--normalized', dest='norm', type=bool, help='true if patch normalized (False by default)',
                        default=False)

    args = parser.parse_args()

    # Reads the csv file containing the occurences

    df = pd.read_csv(args.dataset, sep=';', header='infer', quotechar='"', low_memory=True)
    df = df.dropna(axis=0, how='all')

    ## MODIFIED : Now all the files are saved in the same directory and their
    ## names are just the index of the row in the dataframe.

    batch_size = 10000  # number of patch to extract simultaneously
    # testing destination directory
    if not os.path.isdir(args.destination):
        os.mkdir(args.destination)

    ext = PatchExtractor(args.rasters, size=args.size, verbose=True)

    positions = []
    # exception = ('proxi_eau_fast','alti', 'clc') # add rasters that don't fit into memory
    exception = tuple()
    # testing destination directory
    if not os.path.isdir(args.destination):
        os.mkdir(args.destination)

    export_idx = 0
    for idx, occurrence in enumerate(df.iterrows()):
        # adding an occurrence latitude and longitude
        positions.append((occurrence[1].Latitude, occurrence[1].Longitude))

        # if the batch is full, extract and export
        if len(positions) == batch_size or idx == len(df) - 1:
            variables = []
            for i, raster in enumerate(sorted(raster_metadata.keys())):
                if raster in exception:
                    continue
                ext.clean()
                ext.append(raster, normalized=args.norm)
                variable = np.stack([ext[p] for p in positions])

                variables.append(variable)

            variables = np.concatenate(variables, axis=1)
            # the shape of variables is (batch_size, nb_rasters, size, size)

            for p_idx in range(variables.shape[0]):

                np.save(args.destination + '/' + str(export_idx), variables[p_idx])

                export_idx += 1
            # resetting positions for new batch
            positions = []
    print('done!')
