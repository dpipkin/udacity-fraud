#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


clr = SVC(kernel='rbf', C = 10000.0)
print("Training...")
t0 = time()
clr.fit(features_train, labels_train)
print("Training Time: {:.2f}".format(time() - t0))

indexes = [10, 26, 50]
print("Predicting at indexes {}".format(indexes))
print(clr.predict([features_test[i] for i in indexes]))
print("Num. Chris: {}".format(sum(clr.predict(features_test))))

print("Accuracy: {}".format(clr.score(features_test, labels_test)))
