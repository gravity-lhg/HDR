# Author: Haoguang Liu
# Data: 2022.04.05 16:34 PM
# E-mail: liu.gravity@gmail.com

import numpy as np
import copy

class Bayes():
    ''' Bayesian Binary Classifier '''
    def __init__(self):
        self.priori_probability = 0.1
        
    def area_statistics(self, feature_map):
        ''' Small Area Binary Statistics '''
        mask_pos = feature_map / 16.0 > 0.8
        feature_map[mask_pos] = 1
        feature_map[~mask_pos] = 0
        return feature_map

    def area_probability(self, feature_map, counter):
        ''' Regional probability calculation '''
        index_num = 0
        ap_matrix = np.zeros((10, 7, 7), dtype=np.float32)
        for i in range(10):
            ap_matrix[i] = (feature_map[index_num:index_num+counter[str(i)], :, :].sum(axis=0) + 1) / (counter[str(i)] + 2)
            index_num += counter[str(i)]
        return ap_matrix

    def class_conditional_probability(self, feature_map, ap_matrix, mode='single'):
        ''' Class Conditional Probability Computation '''
        if mode == 'dataset':
            ccp_matrix = np.zeros((10000, 10), dtype=np.float32)
        elif mode == 'single':
            ccp_matrix = np.zeros((1, 10), dtype=np.float32)
        mask_pos = feature_map == 1
        feature_map_copy = copy.deepcopy(feature_map)
        feature_map_copy[mask_pos] = 0
        feature_map_copy[~mask_pos] = 1
        for i in range(10):
            ap_matrix_part = np.expand_dims(ap_matrix[i], axis=0)
            sum1_matrix = feature_map * ap_matrix_part
            sum1_matrix[~mask_pos] = 1
            sum1 = np.prod(sum1_matrix, axis=(1, 2))

            sum2_matrix = feature_map_copy * (1 - ap_matrix_part)
            sum2_matrix[mask_pos] = 1
            sum2 = np.prod(sum2_matrix, axis=(1, 2))

            ccp_matrix[:,i] = sum1 * sum2
        return ccp_matrix

    def posterior_probability(self, ccp_matrix):
        ''' Posterior probability calculation '''
        all_matrix = (self.priori_probability * ccp_matrix).sum(axis=1)
        pp_matrix = (self.priori_probability * ccp_matrix) / np.expand_dims(all_matrix, axis=-1)
        return pp_matrix


    def bayes(self):

        pass