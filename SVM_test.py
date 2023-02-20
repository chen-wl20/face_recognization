import torch
from sklearn.decomposition import PCA
from sklearn import svm
import joblib
from skimage import io
import numpy as np
import random as rd
import os
from torch.utils.data import Dataset
from torch.utils import data
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import cv2
import dlib
import math
import utils

try_batchs = [1, 2, 3, 4, 5, 6, 7]
result_dir_SVM = './result/SVM'
kernels = ['linear', 'rbf', 'poly']
feature = 0.25
for kernel in kernels:
    for try_batch in try_batchs:
        face_recognizer = svm.SVC(kernel=kernel, decision_function_shape="ovo")
        face_recognizer = joblib.load('./svm/svm_{}_{}.m'.format(kernel, try_batch))
        for i in range(600):
            A = cv2.imread('../../data/test_pair/{}/A.jpg'.format(str(i)))
            B = cv2.imread('../../data/test_pair/{}/B.jpg'.format(str(i)))
            A = utils.face_alignment(A, is_gray=True).reshape(1, -1)
            B = utils.face_alignment(B, is_gray=True).reshape(1, -1)
            
            pca = PCA(feature)
            DATA_ = pca.fit_transform(A.reshape(1, -1))
            
            labelA, confidenceA = face_recognizer.predict(A)
            labelB, confidenceB = face_recognizer.predict(B)
            
            if not os.path.exists(result_dir_SVM):
                os.makedirs(result_dir_SVM)
            fp = open(result_dir_SVM+'/SVM{}_{}.txt'.format(kernel,try_batch), 'a')
            if labelA==labelB:
                fp.write('1\n')
            else:
                fp.write('0\n')
            fp.close()