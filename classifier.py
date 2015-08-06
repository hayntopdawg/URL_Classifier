__author__ = 'Jamie Fujimoto'

import dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import processor
import time

from operator import itemgetter
from processor import process_url

# Machine Learning Algorithms
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

# For evaluating ML results
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.externals import joblib


def get_accuracy(X, y, nfolds, clf):
    # kfold = KFold(X.shape[0], n_folds=nfolds, shuffle=True)
    skfold = StratifiedKFold(y, n_folds=nfolds, shuffle=True)
    acc = 0
    i = 0
    # print("%s (#-features=%d)..." % (clfname, nfeats))
    # for train, test in kfold:
    for train, test in skfold:
        i += 1
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        clf.fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        score = accuracy_score(ytest, ypred)
        # cm = confusion_matrix(ytest, ypred)
        # print cm
        # print "  Fold #%d, accuracy=%f" % (i, score)
        acc += score
    acc /= nfolds
    # print "(#-features=%d) accuracy=%f" % (X.shape[1], acc)
    return acc


def save_classifier(clf, filename):
    joblib.dump(clf, filename, compress=9)


def load_classifier(filename):
    return joblib.load(filename)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    np.set_printoptions(precision=2)
    print('Confusion matrix')
    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign", "Malicious"])
    plt.yticks(tick_marks, ["Benign", "Malicious"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_classifier(path, model, **kwargs):
    train = dataset.from_npy(path, "train.npy")
    test = dataset.from_npy(path, "test.npy")

    ###  Logistic Regression  ###
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    clf = model(**kwargs)

    start = time.time()
    clf.fit(train_X, train_y)
    finit = time.time()
    delta = finit - start

    print "Training took %0.3f seconds" % delta
    print clf

    clf_pred = clf.predict(test_X)

    print "Accuracy: " + str(accuracy_score(test_y, clf_pred))
    print "F1 Score: " + str(f1_score(test_y, clf_pred))

    cm = confusion_matrix(test_y, clf_pred)
    plot_confusion_matrix(cm)
    plt.draw()

    return clf


# http://pythonprogramming.net/combine-classifier-algorithms-nltk-tutorial/
class VoteClassifier():
    def __init__(self, classifiers):
        self._classifiers = classifiers
        self._cm = {}


    def fit(self, train_X, train_y):
        for clf in self._classifiers:
            clf.fit(train_X, train_y)


    def confusion_matrix(self, test_X, test_y):
        for clf in self._classifiers:
            predict = clf.predict(test_X)
            self._cm[type(clf).__name__] = confusion_matrix(test_y, predict)


    def vote(self, X):
        votes = []
        for clf in self._classifiers:
            vote = clf.predict(X)
            votes.append(self.confidence(type(clf).__name__, vote))

        # print votes
        winner = self.find_winner(votes)
        return winner

        # # http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
        # winner = max(votes, key=votes.count)
        # conf = self.confidence(winner, votes)
        # return int(winner), conf


    def confidence(self, clf_name, vote):
        cm = self._cm[clf_name]
        if vote == 0:
            correct = cm[0][0] / float(sum(cm[0]))
            incorrect = cm[1][0] / float(sum(cm[1]))
            confidence = correct / sum([correct, incorrect])
        else:
            correct = cm[1][1] / float(sum(cm[1]))
            incorrect = cm[0][1] / float(sum(cm[0]))
            confidence = correct / sum([correct, incorrect])
        return (vote, confidence)


    # def confidence(self, winner, votes):
    #     win_count = votes.count(winner)
    #     return float(win_count) / len(votes)


    def find_winner(self, votes):
        """
        Averages the confidence of each classifier's vote.
        Returns the vote with the higher average along with its correlated confidence.
        """
        tally = {0:[], 1:[]}
        for vote in votes:
            if vote[0] == 1:
                tally[0].append(vote[1])
                tally[1].append(1 - vote[1])  # Appends the probability that the vote is the opposite classification
            else:
                tally[1].append(vote[1])
                tally[0].append(1 - vote[1])
        conf_0 = sum(tally[0]) / len(tally[0])
        conf_1 = sum(tally[1]) / len(tally[1])

        return (0, conf_0) if conf_0 > conf_1 else (1, conf_1)
        # http://stackoverflow.com/questions/13145368/find-the-maximum-value-in-a-list-of-tuples-in-python
        # winner = max(votes, key=itemgetter(1))
        # return winner


def create_voter():
    classifiers = [GaussianNB(), BernoulliNB(), LinearSVC(), DecisionTreeClassifier(), RandomForestClassifier(),
                   ExtraTreesClassifier(), LogisticRegression(), SGDClassifier()]
    voter = VoteClassifier(classifiers)
    return voter


def save_voter(voter, path):
    for clf in voter._classifiers:
        save_classifier(clf, os.path.join(path, type(clf).__name__ + '.pkl'))


def load_voter(path):
    voter = create_voter()
    for i, clf in enumerate(voter._classifiers):
        voter._classifiers[i] = load_classifier(os.path.join(path, type(clf).__name__ + '.pkl'))
    return voter


def controller(folder):
    path = os.path.join(os.path.expanduser('~'), 'OneDrive\\RPI\\Summer Project\\URL Classifier\\Dataset', folder)
    X = dataset.from_npy(path, 'X.npy')
    y = dataset.from_npy(path, 'y.npy')
    classifiers = [GaussianNB(), BernoulliNB(), LinearSVC(), DecisionTreeClassifier(), RandomForestClassifier(),
                   ExtraTreesClassifier(), LogisticRegression(), SGDClassifier()]
    for clf in classifiers:
        accuracy = get_accuracy(X, y, 10, clf) * 100
        print type(clf).__name__, ':', accuracy


if __name__ == '__main__':
    # controller('Lexical')
    # controller('Trial_03')
    trial = 'Trial_03'
    # trial = 'Lexical'
    path = os.path.join(os.path.expanduser('~'), 'OneDrive\\RPI\\Summer Project\\URL Classifier\\Dataset', trial)

    # voter = create_voter()
    #
    # train = dataset.from_npy(path, 'train.npy')
    # train_X, train_y = train[:, :-1], train[:, -1]
    # voter.fit(train_X, train_y)

    # save_voter(voter, path)

    voter = load_voter(path)
    test = dataset.from_npy(path, 'test.npy')
    test_X, test_y = test[:, :-1], test[:, -1]
    voter.confusion_matrix(test_X, test_y)

    # for clf in voter._classifiers:
    #     print type(clf).__name__
    #     cm = confusion_matrix(test_y, clf.predict(test_X))
    #     plot_confusion_matrix(cm)
    #     plt.draw()
    #
    test_urls = [
        'https://www.youtube.com/watch?v=4WM6hB7l4Lc&list=PLQVvvaa0QuDd0flgGphKCej-9jp-QdzZ3&index=12&feature=iv&src_vid=81ZGOib7DTk&annotation_id=annotation_1856532697',
        'http://ld.mediaget.com/index.php?l=ru&amp;fu=http:/www.playground.ru/download/?cheat=grand_theft_auto_4_gta_iv_episodes_from_liberty_city_eflc_sohranenie_100-41709&amp;r=playground.ru&amp;f=grand_theft_auto_4_gta_iv_episodes_from_liberty_city_eflc__&%23x421;&%23x43e;&%23x445;&%23x440;&%23x430;&%23x43d;&%23x435;&%23x43d;&%23x438;&%23x435;_100%25',
        'https://raw.github.com/inquisb/shellcodeexec/master/windows/shellcodeexec.x32.exe',
        'http://www.ezthemes.com/site_advertisers/extrafindWD.exe']
    for test_url in test_urls:
        processed_url = process_url(test_url).values()
        print voter.vote(processed_url)
        # break
        # winner, conf = voter.vote(processed_url)
        # print 'Classification:', 'Malicious' if winner == 1 else 'Benign', 'Confidence:', conf * 100
    #
    # plt.show()