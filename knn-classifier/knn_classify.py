# -*- coding: UTF-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Author      : waynamigo
#   Created date: 19-6-3 下午2:16
#   Usage : python knn_classifier.py -fp imagename
#
#   more details are printed.
# ================================================================
import cv2
import os
import numpy as np
import argparse
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
def identifyby_vgg16_feature(imgbyvgg):
    feature_gen = VGG16(
        weights= 'imagenet',
        include_top = False)
    feature_gen.summary()
    imagelist = list(os.listdir('data/'))
    imagelist = imagelist[:20]
    features = []
    labels =[]
    for imagename in imagelist:
        # load the image and extract the class label (assuming that our
        path = 'data/' + imagename
        print path
        label = getlabel_splitedbydot(imagename)
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(feature_gen.predict(x))
        labels.append(label)
    features = np.array(features)
    print(features[0].shape)
    labels = np.array(labels)
    # imgbyvgg = 'horse.1.jpg'
    img = image.load_img('data/'+imgbyvgg, target_size=(224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    d = select_max(np.array(feature_gen.predict(x)),features, labels, k=6)
    print d

def euclidean_distance(TestIMGs, TrainIMGs):
    testnum = TestIMGs.shape[0]
    trainnum = TrainIMGs.shape[0]
    distances = np.zeros((testnum, trainnum))
    for i in range(testnum):
        for j in range(trainnum):
            distances[i,j]=np.sqrt(np.sum((TestIMGs[i,:]-TrainIMGs[j,:])**2))
    print('distance',distances)
    return distances
def predict(TestIMGs, TrainIMGs, Train_Label, k ):
    distances = euclidean_distance(TestIMGs, TrainIMGs)
    sample_test = TestIMGs.shape[0]
    prediction = np.zeros(sample_test)
    for i in range(sample_test):
        test_row = distances[i,:]
        sorted_row = np.argsort(test_row)
        print('sorted distance subscript:',sorted_row)
        closet_y =Train_Label[sorted_row[0:k]]
        print('closed k',closet_y)
        closet_y.astype(np.int64)
        prediction[i] = np.argmax(np.bincount(closet_y))
    return prediction
def select_set(TestIMGs, Test_Label, TrainIMGs, Train_Label, k):
    prediction = predict(TestIMGs, TrainIMGs, Train_Label, k)
    num_correct = np.sum(prediction == Test_Label)
    print('correct',num_correct)
    accuracy = np.mean(prediction == Test_Label)
    d = {"k": k,
         "predictionlist": prediction,
         "accuracy": accuracy}
    return d
def select_max(requiredIMG, TrainIMGs, Train_Label, k):
    Y_prediction = predict(requiredIMG, TrainIMGs, Train_Label, k)
    if Y_prediction[0]==0:
        print 'this is a cat'
    elif Y_prediction[0]==1:
        print 'this is a dog'
    elif Y_prediction[0] ==2:
        print 'this is a horse'
    else:
        print 'this is a rabbit'

def getlabel_splitedbydot(imgpath):
    if imgpath.split()[-1].split('.')[0] == 'cat':
        return 0
    elif imgpath.split()[-1].split('.')[0] == 'dog':
        return 1
    elif imgpath.split()[-1].split('.')[0] == 'horse':
        return 2
    elif imgpath.split()[-1].split('.')[0] == 'rabbit':
        return 3
def knn_train_testset():
    imagelist = list(os.listdir('data/'))
    imagelist = imagelist[:767]
    features = []
    labels = []
    for imagename in imagelist:
        path = 'data/' + imagename
        img = cv2.imread(path)
        label = getlabel_splitedbydot(imagename)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, (24, 24, 24), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist.flatten()
        features.append(hist), labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    (trainFeat, testFeat, trainLabels, testLabels) = \
        train_test_split(features, labels, test_size=0.20, random_state=30)
    print trainFeat.shape, testFeat.shape, testLabels.shape
    d = select_set(testFeat,testLabels, trainFeat,trainLabels,k=5)
    print d
def identifier(imgpath):
    imagelist = list(os.listdir('data/'))
    imagelist = imagelist[:760]
    features = []
    labels = []
    for imagename in imagelist:
        img = cv2.imread('data/'+imagename)
        label = getlabel_splitedbydot(imagename)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, (24, 24, 24), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist.flatten()
        features.append(hist), labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    (TrainIMGs, TestIMGs, TrainLabels, TestLabels) = \
        train_test_split(features, labels, test_size=0, random_state=30)
    inputimage  = cv2.imread(imgpath)
    hsv = cv2.cvtColor(inputimage, cv2.COLOR_BGR2HSV)
    '''
    cv2.calcHist(params[]) 
    params  H/S/V,mask ,null ,HSV corresponding bin(s)   ,pixels'range (0-180,0-256,0-256)
    '''
    img2histogram = cv2.calcHist([hsv], [0, 1, 2], None, (24,24,24), [0, 180, 0, 256, 0, 256])
    cv2.normalize(img2histogram, img2histogram)
    img2histogram.flatten()
    imgfeature = np.array(img2histogram)
    print imgfeature.shape
    select_max(imgfeature, TrainIMGs,TrainLabels, k=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", dest="imagename",
                        help="input filename and dealwith it plainly")
    parser.add_argument("-vgg16", dest="imgbyvgg",
                        help="input filename and extract features with vgg16")
    args = parser.parse_args()
    if args.imagename:
        imgpath  = 'data/'+ args.imagename
        print imgpath
        identifier(imgpath)
    elif args.imgbyvgg:
        identifyby_vgg16_feature(args.imgbyvgg)
    else:
        #default running on train/test_dataset
        knn_train_testset()