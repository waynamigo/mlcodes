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

def predict(TestIMGs, TrainIMGs, Train_Label, k = 1):
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
    print Y_prediction
    return Y_prediction

def model(TestIMGs, Test_Label, TrainIMGs, Train_Label, k = 1, print_correct = False):
    Y_prediction = predict(TestIMGs, TrainIMGs, Train_Label, k)
    num_correct = np.sum(Y_prediction == Test_Label)
    accuracy = np.mean(Y_prediction == Test_Label)
    if print_correct:
        print('Correct %d/%d: The test accuracy: %f' % (num_correct, TestIMGs.shape[1], accuracy))
    d = {"k": k,
         "Y_prediction": Y_prediction,
         "accuracy": accuracy}
    return d

def knn():
    imagelist = list(os.listdir('./datasets/train/'))
    imagelist = imagelist[:300]
    # print imagelist
    rawImages = []  # pixels
    features = []  # histogram
    labels = []  # class
    for imagename in imagelist:
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        path = './datasets/train/' + imagename
        image = cv2.imread(path)
        if imagename.split()[-1].split(".")[0]=='cat':
            label = 0
        else:
            label = 1
        pixels = cv2.resize(image, (32, 32)).flatten()
        # extract  BGR color feature to HSV colorarea
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # make the histogram normalized
	    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
        hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist.flatten()
        # append each vector
        rawImages.append(pixels), features.append(hist), labels.append(label)
    # array them
    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    # (train_images_raw, test_images_raw, train_lables, test_lables) = \
    #     train_test_split(rawImages, labels, test_size=0.20, random_state=42)
    # print rawImages.shape, features.shape, labels.shape
    (trainFeat, testFeat, trainLabels, testLabels) = \
        train_test_split(features, labels, test_size=0.20, random_state=42)
    # print train_images_raw.shape , test_images_raw.shape ,train_lables.shape
    # print("[INFO] evaluating raw pixel accuracy...")
    # d = model(test_images_raw, test_lables, train_images_raw,train_lables, k=1)
    # print d
    print trainFeat.shape, testFeat.shape, testLabels.shape
    print("[INFO] evaluating raw pixel accuracy...")
    d = model(testFeat, testLabels, trainFeat,trainLabels, k=3)
    print d
def identifier(imgpath):
    imagelist = list(os.listdir('./datasets/train/'))
    imagelist = imagelist[:100]
    # print imagelist
    rawImages = []  # pixels
    features = []  # histogram
    labels = []  # class
    for imagename in imagelist:
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        path = './datasets/train/' + imagename
        image = cv2.imread(path)
        if imagename.split()[-1].split(".")[0]=='cat':
            label = 0
        else:
            label = 1
        pixels = cv2.resize(image, (32, 32)).flatten()
        # extract  BGR color feature to HSV colorarea
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # make the histogram normalized
	    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
        hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist.flatten()
        # append each vector
        rawImages.append(pixels), features.append(hist), labels.append(label)
    # array them
    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    inputimage  = cv2.imread(imgpath)
    hsv = cv2.cvtColor(inputimage, cv2.COLOR_BGR2HSV)
    img2histogram = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(img2histogram, img2histogram)
    img2histogram.flatten()
    imgfeature = np.array(img2histogram)
    print
    if imgpath.split()[0].split('/')[3].split('.')[0] == 'cat':
        label = 0
    else:
        label = 1
    print("[INFO] evaluating histogram is ")
    d = model(imgfeature, np.array(label), features,labels, k=1)
    print d
if __name__ == '__main__':
    # identifier('./datasets/train/cat.0.jpg')
    knn()