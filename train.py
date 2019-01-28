
import os
import cv2
from skimage.measure import label, regionprops
from skimage import data, util
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import math
from data_loader import data_loader

class LR_Model(object):

    def __init__(self,wsize,stride,lr=0.0001):
        '''
            initialize logistic regression model
            Inputs:
				wsize - window size
                stride - stride length
                lr - learning rate
        '''
        self.wsize = wsize #window size
        self.wsize2 = wsize//2
        self.stride = stride #strides
        self.lr = lr #learning rate
        self.iters = 5 #no. of iterations
        self.cost_log = [] #cost log to be graphed later
        self.weights = np.random.rand(3) #model weights
        self.bias = np.random.rand(1) #model bias


    @staticmethod    
    def sigmoid(z):
        '''
            compute sigmoid
        '''
        return 1/(1+np.exp(-z))

    def predict(self,features):
        '''
            Returns 1D array of probabilities
            that the class label == 1
            Inputs:
				features - vector of features
			Outputs:
				predictions - sigmoid of dot product of features with weights
        '''
        z = np.dot(features, self.weights) + self.bias
        return self.sigmoid(z)

    def cost_function(self, predictions, labels):
        '''
            Returns cost
            Inputs:
                predictions - probability of features being 1 or 0
                labels - true 1s and 0s of features
            Outputs:
                cost - float value of cost
        '''
        #Take the error when label=1
        cost = -labels*np.log(predictions+0.00001) - (1-labels)*np.log(1-predictions+0.00001)
        return cost

    def update_weights(self, features, labels):
        '''
            Update weights with gradient descent
            Inputs:
                features - feature vector
                labels - true 1s and 0s of features
            Outputs:
                predictions - probability of predicted value of feature
        '''
        N = len(features)
        predictions = self.predict(features) #sigmoidal function
        gradient = labels - predictions
        #5 - Subtract from our weights to minimize cost
        self.weights += self.lr*gradient*np.transpose(features)
        self.bias += self.lr*gradient
        return predictions
        
    def train(self, training_set):
        '''
            Train a binary classifier with Logistic Regression
            Inputs:
                training_set - generator for training sampels from dataloader class
            Outputs:
                None
        '''
        for i in range(self.iters):
            j = 0
            for feature,label in training_set.generate(self.wsize2,self.stride):
                #update weights
                prediction = self.update_weights(feature, label)
                #Calculate error
                j+=1
                cost = self.cost_function(prediction,label)
            #Log Progress
            self.cost_log.append((cost+0.0)/j)
            print("iter: "+str(i) + " cost: "+str(cost))

    def decision_boundary(self,prob):
        return 1 if prob >= 0.5 else 0

    def test(self,img):
        '''
            test model on image
            Inputs: 
                img - Cv2 read image 
            Outputs:
                mask_img - predicted binary mask of image showing barrel using model
        '''
        #use YCRCB color space
        img = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        avg_img = np.array([np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])])
        std_img = np.array([np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2])])
        [x,y,z] = img.shape
        mask_img = np.zeros([x,y])
        #slide window, take average of window as feature vector
        for i in range(x):
            for j in range(y):
                feature = img[i,j,:]
                #standardize
                feature = (img[i,j,:] - avg_img)/std_img
                #get mask of predictions
                mask_img[i,j] = self.decision_boundary(self.predict(feature))
        mask_img = mask_img.astype(int)
        return mask_img
    
    def load(self,weights):
        '''
            Load pre-trained weights into model
            Input:
                weights - name of file containing weights
            Output:
                None
        '''
        with open(weights,'rb') as f:
            parameters = pickle.load(f)
        self.weights = parameters[0]
        self.bias = parameters[1]

wsize = 10
stride = 4
#make training set by drawing boxes
folder = "trainset"
dataset = data_loader()
makedata = input('make a dataset? Y/N: ')
if makedata == 'Y':
    dataset.save_masks(folder)
model = LR_Model(wsize,stride,lr=0.0001)
#Train by opening dataset and running Logistic Regression
maketrain = input('train? Y/N: ')
if maketrain == 'Y':
    model.train(dataset)
    #save weights
    with open('weights.pickle', 'wb') as f:
        pickle.dump((model.weights,model.bias), f)
    #plot cost
    plt.plot(model.cost_log,label='Cost')
    plt.xlabel('iterations')
    plt.ylabel('Cost')
    plt.show()

