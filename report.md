# Fraudsters of Enron

## Project and Dataset Intro

The goal of this project is to identify persons of interest in 2002 Enron fraud debacle. Persons of interest are individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity. A detailed list can be found [here](http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm). To help us with the prediction, we have email and financial information from public datasets for many individuals that worked at Enron. This includes things like their salaries, stock values, who they sent emails to, etc.

### No peeking

![Using a cheatsheet](pictures/cheating.jpg)

The point is not just to identify persons of interest, but to **teach a computer** how to identify persons of interest using financial and email data. I make this distinction because we could easily just make a model that uses email addresses to identify people, and lookup that person in a list of persons of interest. But instead, we are going to have a model that can look at the underlying data and make a guess based on that. We'll use a couple tricks like [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) to make sure that our computer isn't just memorizing answers for the test.

### Outlying data points

The course mentioned that the list of people was compiled using the data from the "insiderpay" PDF. When I was looking through that, I found a value called "The Travel Agency in the Park". Because that is not a possible "person of interest" I decided to drop it. Another result of pulling the data from that file is that there was a TOTAL row. That is also not a potential person of interest so I dropped it as well.

## Creating and Selecting Features

There are two ways I looked at creating new features. When I looked at `total_payments` it was very skewed right so I created a `log_total_payments` to try to get a better picture of things. I also decided that the `to_` and `from_poi` counts could better be represented as a percentage of total emails that person sent or received. I created a `percent_to_poi` feature using `from_this_person_to_poi / from_messages` and a `percent_from_poi` feature using `from_poi_to_this_person / to_messages`.

I took two different approaches to selecting the best features. First I looked at the ANOVA f-values between each feature and then I looked at feature importances for a Gradient Boosting Classifier. Here are the top six for both.

### SelectKBest

| feature                 | f-value   |
| ----------------------- | --------- |
| exercised_stock_options | 25.097542 |
| total_stock_value       | 24.467654 |
| bonus                   | 21.060002 |
| salary                  | 18.575703 |
| percent_to_poi          | 16.641707 |
| deferred_income         | 11.595548 |

### Boosted Tree Feature Importances

| feature                 | importance     |
| ----------------------- | -------------- |
| exercised_stock_options | 0.244999727018 |
| other                   | 0.162968411753 |
| shared_receipt_with_poi | 0.148073343323 |
| percent_to_poi          | 0.13288702193  |
| expenses                | 0.131139274733 |
| restricted_stock        | 0.037811224554 |

*Note: NaN values were set to 0 when computing these numbers.

`exercised_stock_options` and `percent_to_poi` show up on both of those lists. `shared_receipt_with_poi` also shows up as #9 on the KBest list, so that seemed like it was worth looking at.

I ended up choosing the five most important features for my optimized Gradient Boosted model.

## Algorithm Choice

I ended up using a Gradient Boosting classifier. I tested Naive Bayes, Random Forests, Gradient Boosting, SVM, and a dummy model. The dummy classifier generates random predictions, respecting the training set class distribution. To test the performance of the models, I looked at precision and recall, which are ways to measure how often things are correctly categorized vs miscategorizations like false negatives and false positives. I took the average recall and precision of the 5 models over 1000 stratified shuffle splits and got the following results.

| Model             | Avg. Precision | Avg. Recall |
| ----------------- | -------------- | ----------- |
| Naive Bayes       | 0.288          | 0.4         |
| Random Forest     | 0.184          | 0.120       |
| Gradient Boosting | 0.362          | 0.278       |
| SVM               | 0.225          | 0.313       |
| Dummy             | 0.117          | 0.125       |

A problem I had early on was using too few splits. I was just using k-fold cross validation over four or five folds and wasn't getting consistent results when testing. The Udacity live help got me pointed in the right direction with the StratifiedShuffleSplit functionality.

## Hyper parameter tuning

Tuning parameters is important to improve your accuracy, precision, and recall. Tuning is done by modifying the parameters that the algorithm uses to train your model. Tuning parameters involves a little bit of guessing and a lot of looking at others' research to find reasonable ranges that make sense for your data set. If you don't tune your parameters well, your results are not likely to be precise.

For my model, I followed some suggestions from [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/). I started off by tuning `n_estimators` and then found a good `max_depth` from there. These values were discovered using the `GridSearchCV` function to look at `n_estimators` from 30 to 80 and `max_depth` from 5 to 15.The site recommended `min_samples_split` of around 0.5-1% of the total values. Our dataset is only 144 samples though, so I figured the default of 2 was probably good right where it was.

## Validation

Validation involves checking your model to make sure it generalizes well. If you don't validate correctly, you might over fit. Over fitting is when your model gets too specific for the data set that you have trained it on. For example, if you have a model that is trying to recognize cats-- an overfit model would be one that can only recognize cats in boxes, because all of your training sets are just cats in boxes.

I validated my model by using a stratified shuffle split. Like mentioned above, this worked so much better than my previous k-fold cross validation strategy. It splits the dataset into a bunch of smaller test sets, preserving the percentage of positive/negative samples. I split it into 1000 test sets of ~14 samples each.

## Evaluation Metrics

Evaluation metrics that I chose include average precision and average recall, I got ~40% on each of these. In this data set, we are classifying persons of interest. Of the people we guessed that were persons of interest, 40% of them were actually persons of interest. The recall rate indicates that out of all of the persons of interest, we were able to accurately identify 40% of them. I got an average accuracy of 84%.
