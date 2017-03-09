#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.tree import DecisionTreeClassifier

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = DecisionTreeClassifier(min_samples_split=40)
print("Fitting the decision tree with min sample of 40...")
t0 = time()
clf.fit(features_train, labels_train)
print("Training time: {:.2f}".format(time() - t0))

acc = clf.score(features_test, labels_test)
print("Accuracy: {}".format(acc))


