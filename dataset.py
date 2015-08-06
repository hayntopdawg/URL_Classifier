import numpy as np
import os
import pandas as pd
import random
import time

from processor import process_url
from Scripts.url import get_maxlines

from sklearn.linear_model import LogisticRegression

__author__ = 'Jamie Fujimoto'

#  Build training and testing data of equal class distribution
#  Save datasets to user specified folder to replicate results (save as npy and/or csv)


CODE_FOLDER = 'OneDrive\\RPI\\Summer Project\\URL Classifier'


def get_samples(path, amount=None):
    max_amt = get_maxlines(path.split('\\')[-1])
    if amount == None or amount > max_amt: amount = max_amt
    df = pd.read_csv(path)
    samples = list(df['url'].sample(amount))
    return samples
# path = os.path.join(os.path.expanduser('~'), 'OneDrive\RPI\Summer Project\Code\Data\URL Files')
# samples = get_samples(os.path.join(path, 'mal_urls.csv'), 5)
# print samples


def create_dataset(urls, label):
    X, y = [], []
    for url in urls:
        url_dict = process_url(url)
        values = []
        for value in url_dict.itervalues():
            values.append(value)
            # print value
            # print item.value
        X.append(values)
        y.append(label)
    return np.array(X), np.array(y)
# samples = get_samples('Data/URL Files/mal_urls.csv', 10)
# X, y = create_dataset(samples, 1)
# print X, type(X), X.shape
# print y, type(y), y.shape


def create_train_test_sets(X, y, test_size):
    y = y.reshape(y.shape[0], 1)
    data = np.concatenate((X, y), axis=1)

    # if test_size is int
    test_rows = np.random.randint(data.shape[0], size=test_size)
    test_rows.sort()
    train = np.delete(data, obj=test_rows, axis=0)
    test = data[test_rows, :]

    # if test_size is float

    return train, test
# X = np.arange(10).reshape(5,2)
# y = np.ones(5)
# print create_train_test_sets(X, y, 2)


def to_csv(path, filename, data):
    if not os.path.exists(path):
        os.mkdir(path)
    pd.DataFrame(data).to_csv(os.path.join(path, filename), header=False, index=False)


def to_npy(path, filename, data):
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(file=os.path.join(path, filename), arr=data)


def from_csv(path, filename):
    data = pd.read_csv(os.path.join(path, filename))
    return data.as_matrix()


def from_npy(path, filename):
    return np.load(os.path.join(path, filename))


def controller(folder, dataset_size, test_size):
    mal_path = os.path.join(os.path.expanduser('~'), CODE_FOLDER, 'Data\\URL Files\\mal_urls.csv')
    ben_path = os.path.join(os.path.expanduser('~'), CODE_FOLDER, 'Data\\URL Files\\ben_urls.csv')
    path = os.path.join(os.path.expanduser('~'), CODE_FOLDER, 'Dataset', folder)

    if dataset_size % 2 != 0: dataset_size += 1
    amount = dataset_size / 2
    if test_size % 2 != 0: test_size += 1

    start = time.time()

    mal_urls = get_samples(mal_path, amount=amount)
    to_csv(path, 'mal_url_sample.csv', mal_urls)
    ben_urls = get_samples(ben_path, amount=amount)
    to_csv(path, 'ben_url_sample.csv', ben_urls)

    X_mal, y_mal = create_dataset(mal_urls, 1)
    X_ben, y_ben = create_dataset(ben_urls, 0)
    X = np.concatenate((X_mal, X_ben), axis=0)
    y = np.concatenate((y_mal, y_ben))
    to_npy(path, 'X.npy', X)
    to_npy(path, 'y.npy', y)

    train_mal, test_mal = create_train_test_sets(X_mal, y_mal, test_size / 2)
    train_ben, test_ben = create_train_test_sets(X_ben, y_ben, test_size / 2)
    train = np.concatenate((train_mal, train_ben), axis=0)
    test =  np.concatenate((test_mal, test_ben), axis=0)
    to_npy(path, 'train.npy', train)
    to_npy(path, 'test.npy', test)

    finit = time.time()
    delta = finit - start
    print "Total time: %0.3f seconds" % delta


# if __name__ == '__main__':
    # controller('Lexical', 60000, 10000)