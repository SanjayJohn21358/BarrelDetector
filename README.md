## Barrel Detector Module

Detects blue barrels in an image and draws bounding boxes around them. Uses Logistic Regression.


## Files:

data_loader.py: Contains Callback and DataLoader class. Callback allows you to choose what bounding boxes to draw on roipoly (i.e. assign a bounding box to a class in GUI) and DataLoader creates a (mask,filename) tuple that can be accessed in the training module to generate feature vectors based on criteria in generate function.

train.py: Contains LR_Model class, implementing logistic regression. Final version implements binary classification, Multi folder version contains multi-label classification. 

barrel_detector.py: Contains BarrelDetector class, uses output weights from train.py (hardcoded for autograder) to segment image and draw bounding boxes over barrels. Uses closing morphological operation to assist with joining poorly segmented areas.

barrel_detector_test.py: Same as above, but configured for user's system version and adds an accuracy determining function for IoU and accuracy metrics, as wellas for plotting images.

weights.pickle: Pickle file containing weights.
