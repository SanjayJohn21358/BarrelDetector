
import os
import cv2
from skimage.measure import label, regionprops
from skimage import data, util
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import math
#comment out data_loader for autograder
#from data_loader import data_loader
#comment new

class LR_Model(object):

    def __init__(self):
        '''
            initialize logistic regression model
            Inputs:
                wsize - window size
                stride - stride length
                lr - learning rate
        '''
        self.wsize = 15 #window size
        self.wsize2 = self.wsize//2
        self.stride = 7 #strides
        self.lr = 0.02 #learning rate
        self.iters = 3000 #no. of iterations
        self.cost_log = [] #cost log to be graphed later
        self.weights = np.zeros([2*(self.wsize**2),2]) #model weights
        self.bias = np.zeros([1,2]) #model bias


    @staticmethod    
    def sigmoid(z):
        '''
            compute sigmoid
        '''
        return 1/(1+np.exp(-z))

    @staticmethod
    def softmax(z):
        '''
            compute softmax (for multi-class)
        '''
        z = np.exp(z - np.max(z,axis=1).reshape(-1,1))
        return z/np.sum(z,axis=1).reshape(-1,1)

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
        return self.softmax(z)

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
        cost = -np.mean(labels*np.log(predictions+0.00001))
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
        #subtract from our weights to minimize cost
        self.weights += (self.lr/N)*(np.dot(np.transpose(features),gradient))
        self.bias += self.lr*np.mean(gradient,axis=0)
        return predictions

    '''
    def train(self,dataset):
        Train a multi-class classifier with Logistic Regression
            Inputs:
                training_set - generator for training samples from dataloader class
            Outputs:
                None
        features,labels = dataset.generate(self.wsize2,self.stride)
        features = features.reshape(-1,(self.wsize**2)*2)
        labels = labels.reshape(-1,2)
        #make validation set
        N = len(features)
        split = int(N*0.9)
        validation_features = features[split:,:]
        validation_labels = labels[split:,:]
        features = features[:split,:]
        labels = labels[:split,:]
        previous_cost = 1000
        new_cost = 0
        print('Created training set!')
        for i in range(1,self.iters):
            #update weights
            prediction = self.update_weights(features, labels)
            #Calculate error
            cost = self.cost_function(prediction,labels)
            #Log Progress
            if i%5 == 0:
                self.cost_log.append(cost+0.0)
            print("iter: "+str(i) + " cost: "+str(cost))
            #validate
            prediction = self.predict(validation_features)
            new_cost = self.cost_function(prediction,validation_labels)
            if new_cost < previous_cost:
                previous_cost = new_cost
            else:
                return None
    '''

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
        R = (img[:,:,0] - avg_img[0])/std_img[0]
        B = (img[:,:,2] - avg_img[2])/std_img[2]
        #slide window, take average of window as feature vector
        for i in range(self.wsize2,x-self.wsize2):
            for j in range(self.wsize2,y-self.wsize2):
                #take standardized window to be feature vector
                R_new = R[i-self.wsize2:i+self.wsize2+1,j-self.wsize2:j+self.wsize2+1].reshape(1,-1)
                B_new = B[i-self.wsize2:i+self.wsize2+1,j-self.wsize2:j+self.wsize2+1].reshape(1,-1)
                feature = np.append(R_new,B_new).reshape(1,-1)
                #get mask of predictions
                predictions = self.predict(feature)
                mask_img[i,j] = predictions[0,1]
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


if __name__ == '__main__':
    #make training set by drawing boxes
    folder = "trainset"
    dataset = data_loader()
    makedata = input('draw regions? Y/N: ')
    if makedata == 'Y':
        dataset.save_masks(folder)   
    model = LR_Model()
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