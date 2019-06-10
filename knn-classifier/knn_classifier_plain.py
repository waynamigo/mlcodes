# -*- coding: UTF-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Author      : waynamigo
#   Created date: 19-6-3 下午2:16
#   Description : python knn_classifier.py
#                   more details are printed.
# ================================================================
import cv2
import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def Euclidean_Distance(TestIMGs, TrainIMGs):
    sample_test = TestIMGs.shape[0]
    sample_train = TrainIMGs.shape[0]
    # print sample_train,sample_test
    distances = np.zeros((sample_test, sample_train))
    for i in range(sample_test):
        for j in range(sample_train):
            distances[i,j]=np.sqrt(np.sum((TestIMGs[i,:]-TrainIMGs[j,:])**2))
    print distances.shape
    return distances
def predict(TestIMGs, TrainIMGs, Train_Label, k ):
    distances = Euclidean_Distance(TestIMGs, TrainIMGs)
    sample_test = TestIMGs.shape[0]
    Y_prediction = np.zeros(sample_test)
    for i in range(sample_test):
        closet_y =[]
        test_row = distances[i,:]
        sorted_row = np.argsort(test_row)
        closet_y =Train_Label[sorted_row[0:k]]
        closet_y.astype(np.int64)
        Y_prediction[i] = np.argmax(np.bincount(closet_y))
    return Y_prediction
def model(TestIMGs, Test_Label, TrainIMGs, Train_Label, k):
    Y_prediction = predict(TestIMGs, TrainIMGs, Train_Label, k)
    num_correct = np.sum(Y_prediction == Test_Label)
    print('correct',num_correct)
    accuracy = np.mean(Y_prediction == Test_Label)
    d = {"k": k,
         "Y_prediction": Y_prediction,
         "accuracy": accuracy}
    return d
def getlabel_splitedbydot(imgpath):
    if imgpath.split()[-1].split('.')[0] == 'cat':
        return 0
    elif imgpath.split()[-1].split('.')[0] == 'dog':
        return 1
    elif imgpath.split()[-1].split('.')[0] == 'horse':
        return 2
    else:
        return 3
def knn():
    imagelist = list(os.listdir('data/'))
    imagelist = imagelist[:767]
    # print imagelist
    features = []  # histogram
    labels = []  # class
    for imagename in imagelist:
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        path = 'data/' + imagename
        image = cv2.imread(path)
        if imagename.split()[-1].split(".")[0]=='cat':
            label = 0
        # elif imagename.split()[-1].split(".")[0]=='rabbit':
        #     label = 1
        else:
            label = 2
        pixels = cv2.resize(image, (32, 32)).flatten()
        # extract  BGR color feature to HSV colorarea
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # make the histogram normalized
	    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
        hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist.flatten()
        # append each vector
        features.append(hist), labels.append(label)
    # array them
    features = np.array(features)
    labels = np.array(labels)
    (trainFeat, testFeat, trainLabels, testLabels) = \
        train_test_split(features, labels, test_size=0.20, random_state=42)
    print trainFeat.shape, testFeat.shape, testLabels.shape
    d = model(testFeat, testLabels, trainFeat,trainLabels,k=8)
    print d
def identifier(imgpath):
    imagelist = list(os.listdir('data/'))
    imagelist = imagelist[:767]
    # print imagelist
    features = []  # histogram
    labels = []  # class
    for imagename in imagelist:
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        path = 'data/' + imagename
        image = cv2.imread(path)
        label = getlabel_splitedbydot(imagename)
        # if imagename.split()[-1].split(".")[0]=='cat':
        #     label = 0
        # elif imagename.split()[-1].split(".")[0]== 'dog':
        #     label = 1
        # elif imagename.split()[-1].split(".")[0] == 'horse':
        #     label = 2
        # else:
        #     label = 3
        pixels = cv2.resize(image, (32, 32)).flatten()
        # extract  BGR color feature to HSV colorarea
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # make the histogram normalized
	    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
        hist = cv2.calcHist([hsv], [0, 1, 2], None, (24, 24, 24), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist.flatten()
        # append each vector
        features.append(hist), labels.append(label)
    # array them
    features = np.array(features)
    labels = np.array(labels)
    (TrainIMGs, TestIMGs, TrainLabels, TestLabels) = \
        train_test_split(features, labels, test_size=0, random_state=30)
    inputimage  = cv2.imread(imgpath)
    hsv = cv2.cvtColor(inputimage, cv2.COLOR_BGR2HSV)
    # cv2.imshow("image1", hsv),cv2.waitKey()
    img2histogram = cv2.calcHist([hsv], [0, 1, 2], None, (24, 24, 24), [0, 180, 0, 256, 0, 256])
    cv2.normalize(img2histogram, img2histogram)
    img2histogram.flatten()
    imgfeature = np.array(img2histogram)
    print imgfeature.shape
    label = getlabel_splitedbydot(imgpath)
    d = model(imgfeature, np.array(label), TrainIMGs,TrainLabels, k=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", dest="imagename",
                        help="input image name in data")
    args = parser.parse_args()
    if args.imagename:
        imgpath  = 'data/'+ args.imagename
        print imgpath
        identifier(imgpath)
    else:
        #dataset_test_splited
        knn()