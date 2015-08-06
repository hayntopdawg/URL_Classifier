import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import dataset

from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from vocabulary import read_csv


__author__ = 'Jamie Fujimoto'


def get_features(path, amount=None):
    features = []
    for idx, row in enumerate(read_csv(path, headers=False)):
        if idx >= amount and amount != None: break
        features.append(row[0])
    return features
# print get_features('../Dataset/Vocab/ranked_words.csv', 1)


# def get_samples(path, amount=None):
#     max_amt = get_maxlines(path.split('/')[-1])
#     if amount == None or amount > max_amt: amount = max_amt
#     samples_df = pd.read_csv(path)
#     return list(samples_df['url'].sample(amount))
# samples = get_samples('../Data/URL Files/mal_urls.csv', 100)
# print len(samples)


def create_testing_dataset(samples, features, label):
    X = np.zeros((len(samples), len(features)))
    y = np.zeros(len(samples)) if label == 0 else np.ones(len(samples))

    for row, sample in enumerate(samples):
        for col, feature in enumerate(features):
            X[row][col] = 1 if feature in sample else 0
    return X, y


# http://sujitpal.blogspot.com/2013/05/feature-selection-with-scikit-learn.html
def get_accuracy(X, y, nfolds, clf):
    kfold = KFold(X.shape[0], n_folds=nfolds, shuffle=True)
    acc = 0
    i = 0
    # print("%s (#-features=%d)..." % (clfname, nfeats))
    for train, test in kfold:
        i += 1
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        clf.fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        score = accuracy_score(ytest, ypred)
        # print "  Fold #%d, accuracy=%f" % (i, score)
        acc += score
    acc /= nfolds
    # print "(#-features=%d) accuracy=%f" % (X.shape[1], acc)
    return acc


def plot(accuracies, nFeatures, classifiers):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax = plt.subplot(111)
    c = ['green', 'blue', 'yellow', 'brown', 'red', 'orange']
    for i in xrange(accuracies.shape[0]):
        ax.plot(nFeatures, accuracies[i, :], color=c[i], marker='o', label=type(classifiers[i]).__name__)
    plt.xlabel("#-Features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs #-Features for different classifiers")
    ax.set_xscale("log")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.7])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
    plt.grid(True)
    plt.show()


def controller():
    nFeatures = [10, 25, 50, 70, 80, 90, 100, 110, 120, 130, 140, 150, 175, 200, 500, 1000, 5000]
    nSamples = 1500
    classifiers = [GaussianNB(), BernoulliNB(), LinearSVC(), DecisionTreeClassifier(), LogisticRegression(), SGDClassifier()]
    # scoreFunc = [None, chi2, f_classif]

    path = os.path.join(os.path.expanduser('~'), 'OneDrive\\RPI\\Summer Project\\URL Classifier\\Data\\URL Files')
    mal_samples = dataset.get_samples(os.path.join(path, 'mal_urls.csv'), amount=nSamples)
    ben_samples = dataset.get_samples(os.path.join(path, 'ben_urls.csv'), amount=nSamples)
    features = get_features('../Dataset/Vocab/ranked_words.csv', amount=None)
    nFeatures.append(len(features))
    accuracies = np.zeros((len(classifiers), len(nFeatures)))

    for col, nf in enumerate(nFeatures):
        X_mal, y_mal = create_testing_dataset(mal_samples, features[0:nf], 1)
        X_ben, y_ben = create_testing_dataset(ben_samples, features[0:nf], 0)
        X = np.concatenate((X_mal, X_ben), axis=0)
        y = np.concatenate((y_mal, y_ben))

        for row, clf in enumerate(classifiers):
            accuracy = get_accuracy(X, y, 10, clf) * 100
            accuracies[row, col] = accuracy

    print accuracies
    plot(accuracies, nFeatures, classifiers)


if __name__ == '__main__':
    controller()