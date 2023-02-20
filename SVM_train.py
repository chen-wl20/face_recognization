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

#device='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
seed = 10
epochs = 12
def setup_seeds(seed=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rd.seed(seed)
setup_seeds(seed)
train_data_path = '~/cwl/code/rgb_set'
img_data = torchvision.datasets.ImageFolder(train_data_path,transform=(transforms.ToTensor()))#,transform=transforms.Compose([transforms.Resize(256),
                                                                                    #transforms.CenterCrop(224),
                                                                                   #transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(img_data, batch_size=2000, shuffle=False)

'''train for LBPH'''
feature = 0.25
try_batchs = [1, 2, 3, 4, 5, 6, 7]
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    face_recognizer = svm.SVC(kernel=kernel, decision_function_shape="ovo")
    for step,(x, y) in enumerate(data_loader):
        print(step)
        pca = PCA(feature)
        x_copy = pca.fit_transform((0.299*x[:,:,:,0]+0.587*x[:,:,:,1]+0.114*x[:,:,:,2]).reshape(x.shape[0],-1))
        x = x.permute(0, 2, 3, 1).numpy()
        y = y.numpy()
        face_recognizer.fit(x_copy, y)
        if not os.path.exists('./svm'):
            os.makedirs('./svm')
        joblib.dump(face_recognizer, './svm/svm_{}_{}.m'.format(kernel, step+1))

