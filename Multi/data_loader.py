import os
import cv2
#from roipoly import RoiPoly
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pickle

fig = plt.Figure()
done = False

class Callback(object):
    """
        Class to add buttons on roipoly module
        Allows for adding different classes in GUI
    """

    has_next = True
    def __init__(self,img):
        """
            initialize class
            Inputs:
                img - input image for roipoly
            Outputs:
                None
        """
        self.greyimg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        self.ROI = np.zeros(self.greyimg.shape)

    def addBarrelBlue(self, event):
        """
            Adds button for adding barrel blue class
            Inputs:
                event - button click
            Outputs:
                None
        """
        # let user draw second ROI
        ROI = RoiPoly(color='b') #let user draw ROI
        plt.show(block=False)
        mask = ROI.get_mask(self.greyimg)
        self.ROI += mask

    def addNonBarrelBlue(self, event):
        """
            Adds button for adding non barrel blue class
            Inputs:
                event - button click
            Outputs:
                None
        """
        # let user draw second ROI
        ROI = RoiPoly(color='r') #let user draw ROI
        plt.show(block=False)
        mask = ROI.get_mask(self.greyimg)
        mask = mask*2
        self.ROI += mask

    def finish(self, event):
        """
            closes current image
            Inputs:
                event - button click
            Outputs:
                None
        """
        global done
        done = True
        # close the current figure
        plt.close(plt.gcf())


class data_loader(object):

    def __init__(self):        
        '''
			Initialize dataloader
		'''

    def draw_box(self,img):
        '''
            Let user draw ROI of image
            Inputs:
                img - cv2 read image
            Outputs:
                callback.ROI - ROIs from roipoly
        '''
        callback = Callback(img)
        fig = plt.figure()
        plt.imshow(img, interpolation='nearest')
        plt.title("left click: line segment         right click: close region")
        plt.subplots_adjust(bottom=0.2)
        ax = plt.gca()
        axblue = plt.axes([0.1, 0.05, 0.2, 0.075])
        axred = plt.axes([0.4, 0.05, 0.2, 0.075])
        axprev = plt.axes([0.75, 0.05, 0.1, 0.075])
        bblue = Button(axblue, 'Add Barrel Blue')
        bblue.on_clicked(callback.addBarrelBlue)
        bred = Button(axred, 'Add Non-Barrel Blue')
        bred.on_clicked(callback.addNonBarrelBlue)
        bprev = Button(axprev, 'Finish')
        bprev.on_clicked(callback.finish)
        plt.show()

        return callback.ROI

    def save_masks(self,folder):
        '''
            Save masks by drawing boxes on images
            Inputs:
                folder - name of folder containing images
            Outputs:
                None
        '''
        filenames = []
        masks = []
        for filename in os.listdir(folder):
            # read one training image
            if filename[-4:] == '.png':
                img = cv2.imread(os.path.join(folder,filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #let user draw box, store ROI
                mask = self.draw_box(img)
                mask = np.clip(mask,0,2)
                mask = mask.astype(int)
                masks.append(mask)
                filenames.append(os.path.join(folder,filename))
        #save masks                                
        with open('masks.pickle', 'wb') as f:
            pickle.dump((masks,filenames), f)


    def generate(self,wsize2,stride):
        '''
            Create training set
            Inputs:
                wsize2 - window size // 2
                stride - stride length
            Outputs:
                feature, label - feature vector (pixel value from training image) and associated binary label
        '''
        #load drawn masks
        with open('masks.pickle', 'rb') as f:
            data = pickle.load(f)
        masks = data[0]
        filenames = data[1]
        features = []
        labels = []
        #for each training image, pull out features and labels
        for (index,filename) in enumerate(filenames):
            mask = masks[index]
            img = cv2.imread(filename)
            #use YCRCB color space
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            #to be used for standardizing
            avg_img = np.array([np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])])
            std_img = np.array([np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2])])
            #slide a window across image
            for i in range(wsize2,img.shape[0]-wsize2,stride):
                for j in range(wsize2,img.shape[1]-wsize2,stride):
                    #take standardized window to be feature vector
                    R = (img[i-wsize2:i+wsize2+1,j-wsize2:j+wsize2+1,0].reshape(1,-1) - avg_img[0])//std_img[0]
                    G = (img[i-wsize2:i+wsize2+1,j-wsize2:j+wsize2+1,1].reshape(1,-1) - avg_img[1])//std_img[1]
                    B = (img[i-wsize2:i+wsize2+1,j-wsize2:j+wsize2+1,2].reshape(1,-1) - avg_img[2])//std_img[2]
                    feature = np.append(R,[G,B]).reshape(1,-1)
                    #standardize
                    label = int(mask[i,j])
                    one_hot_label = np.zeros([3])
                    one_hot_label[label] = 1
                    features.append(feature)
                    labels.append(one_hot_label)
            print(filename + ' done!')
        features = np.array(features)
        labels = np.array(labels)
        return features, labels


