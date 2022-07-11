# Author: Haoguang Liu
# Date: 2022.03.29 21:26 PM
# E-mail: liu.gravity@gmail.com

import os
from PIL import Image
import numpy as np
from pca import PCA
from bayes import Bayes
from cnn import Eval


img_map = {
    '0' : [], '1' : [], '2' : [], '3' : [], '4' : [],
    '5' : [], '6' : [], '7' : [], '8' : [], '9' : [],
}
train_count = {
    '0': 5923, '1': 6742, '2': 5958, '3': 6131, '4': 5842,
    '5': 5421, '6': 5918, '7': 6265, '8': 5851, '9': 5949,
    }
test_count = {
    '0': 980, '1': 1135, '2': 1032, '3': 1010, '4': 982,
    '5': 892, '6': 958,  '7': 1028, '8': 974,  '9': 1009,
    }

ROOT_PATH = '/Users/lhg/work/pytorch-learn/mnist_data/'
DATASET_PATH = ROOT_PATH + 'MNIST/images'

labels = np.zeros([60000])
index_num = 0
for i in range(10):
    for _ in range(train_count[str(i)]):
        labels[index_num] = i
        index_num += 1

def load_feature(mode, task):
    img_path = os.path.join(DATASET_PATH, f'{mode}_images')
    imageList = os.listdir(img_path)
    # Put image data into dict
    for item in imageList:
        img = np.array(Image.open(os.path.join(img_path, item)))
        itemList = item.split('_')
        img_map[itemList[0]].append(img)

    # Concatenate all types of image data
    img_sequence = tuple((img_map[str(i)] for i in range(10)))
    all_img_narray = np.concatenate(img_sequence, axis=0)

    for i in range(10):
        img_map[str(i)] = []

    print(f'Load {mode} dataset ===> Done !!!')

    # Initialize the feature matrix
    if mode == 'train':
        feature_map = np.zeros((60000, 7, 7), dtype=np.float32)
    elif mode == 'test':
        feature_map = np.zeros((10000, 7, 7), dtype=np.float32)

    # Extract features from each image
    for j in range(7):
        for k in range(7):
            feature_map[:, j, k] = (all_img_narray[:, 4*j:4*(j+1), 4*k:4*(k+1)] < 100).sum(axis=(1, 2))
    
    if task == 'bayes':
        return feature_map

    elif task == 'pca':    
        if mode == 'train':
            feature_map = feature_map.reshape(60000, -1)
        elif mode == 'test':
            feature_map = feature_map.reshape(10000, -1)
        return feature_map

def singleImage(path, task, mode):
    if mode == 'choice':
        img = Image.open(path)
    elif mode == 'graph':
        img_np = np.array(path)
        img_np = 255 - img_np
        img = Image.fromarray(img_np.astype('uint8'))
    img_np = np.expand_dims(np.array(img), axis=0)
    feature_map = np.zeros((1, 7, 7), dtype=np.int32)
    for j in range(7):
        for k in range(7):
            feature_map[:, j, k] = (img_np[:, 4*j:4*(j+1), 4*k:4*(k+1)] < 100).sum(axis=(1, 2))
    if task == 'pca':
        feature_map = feature_map.reshape(1, -1)
        feature_norm = np.subtract(feature_map, np.expand_dims(feature_map.mean(axis=1), axis=1))
        subMartix = np.load('npy/subMatrix.npy')
        sub_feature_map = feature_norm @ subMartix.T
        subTrainFM = np.load('npy/subTrainFM.npy')
        dis = euclideanDis(subTrainFM, sub_feature_map)
        index = np.argsort(dis, axis=0)
        return int(labels[index[0][0]])
    elif task == 'bayes':
        bayes = Bayes()
        feature_map = bayes.area_statistics(feature_map)
        ap_matrix = np.load('npy/ap_matrix.npy')
        ccp_matrix = bayes.class_conditional_probability(feature_map, ap_matrix, 'single')
        pp_martix = bayes.posterior_probability(ccp_matrix)
        index = pp_martix.argmax()
        return index
    elif task == 'cnn':
        eval = Eval(ROOT_PATH)
        img_np = np.transpose(img_np, (1, 2, 0))
        pred = eval.evalImg(img_np)
        return pred

def euclideanDis(X, Y):
    ''' get the Euclidean Distance between to matrix 
        math: X^2 + Y^2 -2XY
    '''
    (rowx, colx) = X.shape
    (rowy, coly) = Y.shape
    assert colx == coly, 'colx must be equal with coly'
    xy = np.dot(X, Y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(X, X), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(Y, Y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis

def toPCA(feature_map, contribution):
    pca = PCA(feature_map, contribution)
    subDim, P, train_feature_map_norm = pca.pca()
    sub_train_feature_map = train_feature_map_norm @ P.T
    return subDim, P, sub_train_feature_map

def toBayes(feature_map):
    bayes = Bayes()
    feature_map = bayes.area_statistics(feature_map)
    ap_martix = bayes.area_probability(feature_map, train_count)
    return ap_martix

def bayseTest(test_feature_map, ap_martix):
    bayes = Bayes()
    test_feature_map = bayes.area_statistics(test_feature_map)
    ccp_matrix = bayes.class_conditional_probability(test_feature_map, ap_martix, 'dataset')
    pp_matrix = bayes.posterior_probability(ccp_matrix)
    return pp_matrix

def train(task):
    info = []
    if task == 'pca':
        if os.path.exists('npy/subMatrix.npy') & os.path.exists('npy/subTrainFM.npy'):
            subDim = np.load('npy/subMatrix.npy').shape[0]
            info.append('PCA training is done !')
            info.append(f'SubDim is {subDim}')
        else:
            f_train = load_feature('train', 'pca')
            subDim, P, sub_train_feature_map = toPCA(f_train, 0.95)
            np.save('npy/subMatrix.npy', P)
            np.save('npy/sunTrainFM.npy', sub_train_feature_map)
            info.append('PCA training is done !')
            info.append(f'SubDim is {subDim}')

    elif task == 'bayes':
        if os.path.exists('npy/ap_matrix.npy'):
            info.append('Bayes training is done !')
        else:
            f_train = load_feature('train', 'bayes')
            ap_matrix = toBayes(f_train)
            np.save('npy/ap_matrix.npy', ap_matrix)
            info.append('Bayes training is done !')
    
    elif task == 'cnn':
        info.append('Training is done ! ')

    return info

def test(task):
    info = []
    res_count = np.zeros([10])
    if task == 'pca':
        if os.path.exists('npy/index_disMatrix.npy'):
            index = np.load('npy/index_disMatrix.npy')
        else:
            f_test = load_feature('test', 'pca')
            P = np.load('npy/subMatrix.npy')
            sub_train_feature_map = np.load('npy/subTrainFM.npy')
            test_feature_map_norm = np.subtract(f_test, np.expand_dims(f_test.mean(axis=1), axis=1))
            sub_test_feature_map = test_feature_map_norm @ P.T
            disMatrix = euclideanDis(sub_train_feature_map, sub_test_feature_map)
            index = np.argsort(disMatrix, axis=0)
            np.save('npy/index_disMatrix.npy', index)

        count = 0
        origin_count = []
        for i in range(10):
            for _ in range(test_count[str(i)]):
                if labels[index[0][count]] == i:
                    res_count[i] += 1
                count += 1
            origin_count.append(test_count[str(i)])
        res_precent = []
        res_precent = res_count / origin_count
        info.append('PCA testing is done !')
        str1 = ''
        for i in range(len(res_precent)):
            str1 += str('%.4f'%res_precent[i])
            if i < len(res_precent) - 1:
                if i == 4:
                    str1 += '\n'
                else:
                    str1 += ',  '
        info.append(str1)
        info.append('Average acc: ' + str('%.4f'%res_precent.mean()))

    elif task == 'bayes':
        f_test = load_feature('test', 'bayes')
        ap_matrix = np.load('npy/ap_matrix.npy')
        pp_matrix = bayseTest(f_test, ap_matrix)
        max_index = np.argsort(pp_matrix)[:,::-1]

        count = 0
        origin_count = []
        for i in range(10):
            for _ in range(test_count[str(i)]):
                if max_index[count][0] == i:
                    res_count[i] += 1
                count += 1
            origin_count.append(test_count[str(i)])
        res_precent = []
        res_precent = res_count / origin_count
        info.append('Bayes testing is done !')
        str1 = ''
        for i in range(len(res_precent)):
            str1 += str('%.4f'%res_precent[i])
            if i < len(res_precent) - 1:
                if i == 4:
                    str1 += '\n'
                else:
                    str1 += ',  '
        info.append(str1)
        info.append('Average acc: ' + str('%.4f'%res_precent.mean()))
    
    elif task == 'cnn':
        eval = Eval(ROOT_PATH)
        eval_acc = eval.eval()
        info.append('CNN testing is done !')
        info.append('Test acc: ' + str('%.4f'%eval_acc))

    return info

if __name__ == '__main__':
    # f_train = load_feature('train')
    # subDim, P, sub_train_feature_map = toPCA(f_train, 0.95)
    # print(f"subDim is {subDim}.")
    # np.save('subMatrix.npy', P)
    # np.save('sunTrainFM.npy', sub_train_feature_map)
    # f_test = load_feature('test')
    # test_feature_map_norm = np.subtract(f_test, np.expand_dims(f_test.mean(axis=1), axis=1))
    # sub_test_feature_map = test_feature_map_norm @ P.T
    # disMatrix = euclideanDis(sub_train_feature_map, sub_test_feature_map)
    # index = np.argsort(disMatrix, axis=0)
    # np.save('index_disMatrix.npy', index)
    # print('Save indexMatrix is successed !!!')
    # index = np.load('index_disMatrix.npy')
    
    # count = 0
    # origin_count = []
    # for i in range(10):
    #     for _ in range(test_count[str(i)]):
    #         if labels[index[0][count]] == i:
    #             res_count[i] += 1
    #         count += 1
    #     origin_count.append(test_count[str(i)])
    # res_precent = res_count / origin_count
    # print(res_precent)
    # print(res_precent.mean())

    # path = '/Users/lhg/work/pytorch-learn/mnist_data/MNIST/images/test_images/2_167.bmp'
    # res = singleImage(path, 'pca', 'choice')
    # print(res)

    # f_train = load_feature('train', 'bayes')
    # f_test = load_feature('test', 'bayes')
    # pp_matrix = toBayes(f_train, f_test)
    # max_index = np.argsort(pp_matrix)[:,::-1]
    # print(max_index[0])
    # print(max_index[:2500,0])

    # count = 0
    # origin_count = []
    # for i in range(10):
    #     for _ in range(test_count[str(i)]):
    #         if max_index[count][0] == i:
    #             res_count[i] += 1
    #         count += 1
    #     origin_count.append(test_count[str(i)])
    # res_precent = res_count / origin_count
    # print(res_precent)
    # print(res_precent.mean())
    pass