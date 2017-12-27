#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options','other', 'percent_to_poi',
                 'expenses', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
outliers = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL']
for outlier in outliers:
    data_dict.pop(outlier)

### Task 3: Create new feature(s)

# fill NaNs
to_fill = [feat for feat in features_list if feat not in ['poi', 'percent_to_poi', 'log_total_payments']]
to_fill.extend(['from_this_person_to_poi','from_poi_to_this_person',
                'from_messages','to_messages'])
for attr in to_fill:
    for person in data_dict.itervalues():
        if person[attr] == 'NaN':
            person[attr] = 0

# Create new feature
for _, attrs in data_dict.iteritems():
    # wrapping in a `try` because I filled the NaNs with 0s
    try:
        attrs['percent_to_poi'] = attrs['from_this_person_to_poi'] / attrs['from_messages']
    except ZeroDivisionError:
        attrs['percent_to_poi'] = 0
    try:
        attrs['percent_from_poi'] = attrs['from_poi_to_this_person'] / attrs['to_messages']
    except ZeroDivisionError:
        attrs['percent_from_poi'] = 0
    try:
        attrs['log_total_payments'] = np.log(attrs['total_payments'])
    except TypeError:
        attrs['log_total_payments'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=70, max_depth=9)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
