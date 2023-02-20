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
from torchvision import transforms, utils
import cv2
import dlib
import math
import os
def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
def shape_to_np(shape, dtype="int"): # 将包含68个特征的的shape转换为numpy array格式
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized
def face_alignment(face, is_gray=False):
    face_shape = (face.shape[0], face.shape[1])
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        if(x<0 or y<0):
            continue
        #print(face.shape, y, y+h, x, x+w)
        face = face[y:y+h, x:x+w]
        #print(face.shape, face_shape, (x, y, w, h))
    
        face = cv2.resize(face, face_shape)
        break
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rec = dlib.rectangle(0, 0, face.shape[0],face.shape[1])
    shape = predictor(face, rec)
    order = [36, 45, 30, 48, 54]
    for j in order:
        x = shape.part(j).x
        y = shape.part(j).y
        face = np.ascontiguousarray(face, dtype=np.uint8)
        cv2.circle(face, (x,y), 2, (0, 0, 255),-1)
    eye_center = ((shape.part(36).x+shape.part(45).x)*1./2,
                 (shape.part(36).y+shape.part(45).y)*1./2)
    dx = (shape.part(45).x-shape.part(36).y)
    dy = (shape.part(45).y-shape.part(36).y)
    angle = math.atan2(dy,dx)*180./math.pi
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    RotImg = cv2.warpAffine(face, RotateMatrix,face_shape)
    if is_gray:
        gray = cv2.cvtColor(RotImg, cv2.COLOR_BGR2GRAY)
        return gray
    else:
        return RotImg
'''
raw_data_path = "~/data/training_set"
raw_data = torchvision.datasets.ImageFolder(raw_data_path, transform=(transforms.ToTensor()))
data_loader = torch.utils.data.DataLoader(raw_data, batch_size=200,shuffle = False)

gray_data_path = "gray_set/"
rgb_data_path = "rgb_set/"
for step, (x, y) in enumerate(data_loader):
    print(step)
    for i in range(x.shape[0]):
        print(i, x[i], y[i], step*200+i)
        if step*200+i<1531:
            continue
        face_gray = face_alignment((x[i].permute(1, 2, 0).numpy()*255).astype("uint8"), is_gray=True)
        if(not os.path.exists(gray_data_path+str(y[i])+'/')):
            os.mkdir(gray_data_path+str(y[i])+'/')
        if(not os.path.exists(rgb_data_path+str(y[i])+'/')):
            os.mkdir(rgb_data_path+str(y[i])+'/')
        cv2.imwrite(gray_data_path+str(y[i])+'/'+f'{step*200+i}.jpg', face_gray)
        face_rgb = face_alignment((x[i].permute(1, 2, 0).numpy()*255).astype("uint8"), is_gray=False)
        cv2.imwrite(rgb_data_path+str(y[i])+'/'+f'{step*200+i}.jpg', face_rgb)
'''