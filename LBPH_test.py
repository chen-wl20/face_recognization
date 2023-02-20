import torch
import sklearn
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

try_batchs = [1, 2, 3, 4, 5, 6, 7]
'''face_recognizer = cv2.face.LBPHFaceRecognizer_create()
for step,(x, y) in enumerate(data_loader):
    print(step)
    x = x.permute(0, 2, 3, 1).numpy()
    y = y.numpy()
    face_recognizer.train(0.299*x[:,:,:,0]+0.587*x[:,:,:,1]+0.114*x[:,:,:,2], y)
    face_recognizer.write('./lbp{}.yml'.format(step+1))
'''
'''test for LBPH in training data'''

result_dir_LBPH = './result/LBPH'
for try_batch in try_batchs:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('./lbp{}.yml'.format(try_batch))
    '''error = 0
    total = 0
    not_confident = 0
    for step, (x, y) in enumerate(data_loader):
        print('step:',step)
        if step==try_batch:
            break
        x = x.permute(0, 2, 3, 1).numpy()
        y = y.numpy()
        for i in range(len(y)):
            y_predict, confidence = face_recognizer.predict(0.299*x[i,:,:,0]+0.587*x[i,:,:,1]+0.114*x[i,:,:,2])
            #print(y[i], y_predict)
            error = error + 1-(y[i]==y_predict)
            if confidence>80:
                not_confident = not_confident + 1
            #break
        total = total + len(y)
    print('{}  total:{},error:{},acc{},not_confident:{}'.format(try_batch, total, error, 1.-1.*error/total, not_confident))
    '''
    '''answer for LBPH'''
    for i in range(600):
        A = cv2.imread('../../data/test_pair/{}/A.jpg'.format(str(i)))
        B = cv2.imread('../../data/test_pair/{}/B.jpg'.format(str(i)))
        labelA, confidenceA = face_recognizer.predict(utils.face_alignment(A, is_gray=True))
        labelB, confidenceB = face_recognizer.predict(utils.face_alignment(B, is_gray=True))
        
        if not os.path.exists(result_dir_LBPH):
            os.makedirs(result_dir_LBPH)
        fp = open(result_dir_LBPH+'/LBPH{}.txt'.format(try_batch), 'a')
        if labelA==labelB:
            fp.write('1\n')
        else:
            fp.write('0\n')
        fp.close()