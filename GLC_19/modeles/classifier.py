from glcdataset import GLCDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
class Classifier(object):

    """Generic class for a classifier
    """
    def __init__(self):

        self.train_set = None
        pass

    def fit(self,dataset):
        """Trains the model on the dataset
           :param dataset: the GLCDataset training set
        """
        raise NotImplementedError("fit not implemented!")

    def predict(self, dataset, ranking_size=30):
        """Predict the list of labels most likely to be observed
           for the data points given
        """
        raise NotImplementedError("predict not implemented!")

    def mrr_score(self, dataset):
        """Computes the mean reciprocal rank from a test set provided:
           It finds the inverse of the rank of the actual class along
           the predicted labels for every row in the test set, and
           calculate the mean.
           :param dataset: the test set
           :return: the mean reciprocal rank, from 0 to 1 (perfect prediction)
        """
        predictions = self.predict(dataset)
        mrr = 0.
        for idx,y_predicted in enumerate(predictions):
            try:
                rank = np.where(y_predicted==dataset.get_label(dataset.ytest.index[idx]))
                mrr += 1./(rank[0][0]+1)
            except IndexError: # the actual specie is not predicted
                mrr += 0.
        return 1./len(dataset)* mrr

    def top30_score(self, dataset):
        """It is the accuracy based on the first 30 answers:
           The mean of the function scoring 1 when the good species is in the 30
           first answers, and 0 otherwise, over all test test occurences.
        """
        predictions = self.predict(dataset)
        #predictions = [y_predicted[:30] for y_predicted in predictions] # keep 30 first results
        top30score = 0.
        for y in range(len(predictions)):
            top30score += (dataset.get_label(dataset.ytest.index[y]) in predictions[y])

        return 1./len(dataset)* top30score

    def cross_validation(self, dataset, n_folds, shuffle=True, evaluation_metric='top30'):
        # NEEDS DEBUGGING !!!
        """Cross validation prodedure to evaluate the classifier
           :param n_folds: the number of folds for the cross validation
           :idx_permutation: the permutation over indexes to use before cross validation
           :evaluation_metric: the evaluation metric function
           :return: the mean of the metric over the set of folds
        """
        if evaluation_metric == 'top30':
            metric = self.top30_score
        elif evaluation_metric == 'mrr':
            metric = self.mrr_score
        else:
            raise Exception("Evaluation metric is not known")
        if shuffle:
            idx_random = np.random.permutation(len(dataset))
        else:
            idx_random = np.arange(len(dataset))
        fold_size = len(dataset)//n_folds
        idx_folds = []

        # split the training data in n folds
        for k in range(0,n_folds-1):
            idx_folds.append(idx_random[fold_size*k : fold_size*(k+1)])
        idx_folds.append(idx_random[fold_size*(n_folds-1): ])

        # for each fold:
        # train the classifier on all other fold
        # validation score on the current fold
        scores = []
        for k in range(0,n_folds):

            idx_train = [i for idx_fold in idx_folds[:k]+idx_folds[k+1:] for i in idx_fold]
            idx_test = [i for i in idx_folds[k]]

            train_set = GLCDataset(dataset.data.iloc[idx_train], dataset.labels.iloc[idx_train], None, dataset.patches_dir)
            test_set = GLCDataset(dataset.data.iloc[idx_test], dataset.labels.iloc[idx_test], None, dataset.patches_dir)
            self.fit(train_set)
            scores.append(metric(test_set))
        print(scores)
        return np.mean(scores)
    
    def cross_validation_sklearn(self,dataset,n_folds=5,random=42,test_size=0.1,evaluation_metric='top30'):
        if evaluation_metric == 'top30':
            metric = self.top30_score
        elif evaluation_metric == 'mrr':
            metric = self.mrr_score
        else:
            raise Exception("Evaluation metric is not known")
        
        
        for k in range(n_folds):
            X_train,X_test, y_train, y_test = self.split_train_test(dataset,test_size,random_state=random)
            self.fit()
    
    def split_train_test(self,dataset,test_size=0.2):
        """
            Sépare un dataset en données de train & test. 
            test_size permet de choisir la taille des données de test.
            retourne : X_train,X_test, y_train, y_test.
        """
        return train_test_split(dataset.data, dataset.label, test_size=test_size, random_state=0)

        