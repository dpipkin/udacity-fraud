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

I took two different approaches to selecting the best features. First I looked at the ANOVA f-values between each feature and then I looked at feature importances for a Decision Tree Classifier. Here are the top six for both.

### SelectKBest

| feature                 | f-value   |
| ----------------------- | --------- |
| exercised_stock_options | 25.097542 |
| total_stock_value       | 24.467654 |
| bonus                   | 21.060002 |
| salary                  | 18.575703 |
| percent_to_poi          | 16.641707 |
| deferred_income         | 11.595548 |

### Feature Importances

| feature                 | importancce |
| ----------------------- | ----------- |
| exercised_stock_options | 0.255578    |
| other                   | 0.190136    |
| percent_to_poi          | 0.136054    |
| expenses                | 0.119346    |
| shared_receipt_with_poi | 0.118636    |
| log_total_payments      | 0.093122    |

*Note: NaN values were set to 0 when computing these numbers.

`exercised_stock_options` and `percent_to_poi` show up on both of those lists. `shared_receipt_with_poi` also shows up as #9 on the KBest list, so that's probably worth looking at.

## Algorithm Choice

I ended up using Naive Bayes. I tested Naive Bayes, Random Forests, Gradient Boosting, SVM, and a dummy model. The dummy classifier generates random predictions, respecting the training set class distribution. To test the performance of the models, I looked at precision and recall, which are ways to measure how often things are correctly categorized vs miscategorizations like false negatives and false positives. I took the average recall and precision of the 5 models over 4 cross validation steps and got the following results

| Model             | Avg. Precision | Avg. Recall |
| ----------------- | -------------- | ----------- |
| Naive Bayes       | 0.6            | 0.6         |
| Random Forest     | 0.542          | 0.246       |
| Gradient Boosting | 0.146          | 0.133       |
| SVM               | 0.159          | 0.338       |
| Dummy             | 0.133          | 0.113       |

## Hyper parameter tuning

Tuning parameters is important to improve your accuracy, precision, and recall. Tuning is done by modifying the parameters that the algorithm uses to train your model. Tuning parameters involves a little bit of guessing and a lot of looking at others' research to find reasonable ranges that make sense for your data set. If you don't tune your parameters well, your results are not likely to be precise. For my model, I did not do any tuning, because I chose Naive Bayes which does not have hyper parameters. If I had used Random Forest, I would have used parameters and I could have tuned it by doing a grid search over the hyper parameters. A grid search is a library that tests out every combination of values within a range that you give it.



## Validation

Validation involves checking your model to make sure it generalizes well. If you don't validate correctly, you might over fit. Over fitting is when your model gets too specific for the data set that you have trained it on. For example, if you have a model that is trying to recognize cats-- an overfit model would be one that can only recognize cats in boxes, because all of your training sets are just cats in boxes. I validated my analysis by using k-fold cross validation. This validation splits my data set into four smaller, equally sized sets. It then takes one set that it sets apart as a test data set and then it trains on the other three sets of data. It then repeats this process four times until it has reviewed all the data.

## Evaluation Metrics

Evaluation metrics that I chose include average precision and average recall, I got 60% on each of these. In this data set, we are classifying persons of interest. Of the people we guessed that were persons of interest, 60% of them were actually persons of interest. The recall rate indicates that out of all of the persons of interest, we were able to accurately identify 60% of them. I got an average accuracy of 86%.

