# Author: Haoguang Liu
# Date: 2022.03.28 22:00 PM
# E-mail: liu.gravity@gmail.com

import numpy as np

class PCA():
    ''' Principal Component Analysis '''
    def __init__(self, feature_map, contribution):
        '''
        args:
            feature_map: feature martix for dimensionality reduction
            contribution: or principal component contribution value size
        '''
        self.feature_map = feature_map
        self.contribution = contribution

    def featureNorm(self):
        ''' zero mean processing matrix '''
        row_mean = np.expand_dims(self.feature_map.mean(axis=1), axis=1)
        feature_map_norm = np.subtract(self.feature_map, row_mean)
        return feature_map_norm

    def getCovMatrix(self, feature_map_norm):
        ''' Calculate the covariance matrix '''
        X = feature_map_norm
        n = X.shape[1]
        return (X.T @ X) / n

    def getEigens(self, covMartix):
        ''' Calculate the eigenvalues and eigenvectors of the covariance matrix '''
        value, vector = np.linalg.eig(covMartix)
        return value, vector

    def dimReduction(self, value, vector):
        ''' Dimensionality reduction '''
        valSum = value.sum()
        tempVal = 0.
        subDim = 0
        for i in range(len(value)):
            tempVal += value[i]
            subDim += 1
            if tempVal / valSum > self.contribution:
                break
        P = vector[:subDim][:]
        return subDim, P

    def pca(self):
        ''' forward function for PCA '''
        feature_map_norm = self.featureNorm()
        covMartix = self.getCovMatrix(feature_map_norm)
        val, vec = self.getEigens(covMartix)
        subDim, P = self.dimReduction(val, vec)
        return subDim, P, feature_map_norm