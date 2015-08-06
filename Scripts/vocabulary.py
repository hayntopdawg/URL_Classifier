import csv
import enchant
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import re

from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier #, ExtraTreesClassifier
from sklearn.externals import joblib

__author__ = 'Jamie Fujimoto'


# Initialize global variables
MAL_URL_LOC = '../Data/URL Files/mal_urls.csv'
BEN_URL_LOC = '../Data/URL Files/ben_urls.csv'
punish = 500
common = 'http|https|ftp|www|com|html|htm|php'
porter = nltk.PorterStemmer()
d = enchant.Dict("en_US")


def read_csv(filepath, headers=True):
    with open(filepath, 'rU') as data:
        if headers:
            reader = csv.DictReader(data)
        else:
            reader = csv.reader(data)
        for row in reader:
            yield row


def get_urls(num, filepath):
    url_df = pd.read_csv('{}'.format(filepath))
    return list(url_df['url'].sample(num))


def extract_vocab(urls):
    v = []
    for words in urls:
        # Split URL words on non-alphanumeric chars, underscores and common URL terms
        m = re.split('[\W_]|'+common, words.lower())
        # Stem words
        m = [porter.stem(t) for t in m]
        # Filter strings less than 3 chars, strings with digits, and strings not in English (US) language
        m = filter(lambda x: (len(x) > 2 and not any(c.isdigit() for c in x) and d.check(x)), m)
        v.extend(m)

    # Eliminate duplicates
    return list(set(v))


def add_word_features(url):
    feat = {}
    for w in vocab:
        feat[w] = 1 if w.encode('utf-8') in url else 0  # for every word in vocab, if word is in url assign 1 else 0
    return feat


def create_vocab_dataset(size):
    global vocab

    # Get list of urls
    mal_urls = get_urls(size / 2, MAL_URL_LOC)
    ben_urls = get_urls(size / 2, BEN_URL_LOC)
    all_urls = mal_urls + ben_urls

    # Create vocab list from list of urls
    vocab = extract_vocab(all_urls)

    # Create list of tuples ({url: [features]}, label)
    url_dataset = [(add_word_features(mal_url), 1) for mal_url in mal_urls] + \
                  [(add_word_features(ben_url), 0) for ben_url in ben_urls]
    return url_dataset


def save_words_to_file(clf, folder, iteration):
    # Check if folder exists
    path = '{}'.format(folder)
    if not os.path.isdir(path):
        os.mkdir(path)

    # Write every word to a csv file
    with open('{}/NLTK_NB_informative_features_{}.csv'.format(folder, iteration), 'wb') as f:
        writer = csv.writer(f)
        for feat in clf.most_informative_features(len(vocab)):
            writer.writerow((feat[0], feat[1]))


def save_important_features(forest, features, folder, iteration):
    importances = forest.feature_importances_

    # Check if folder exists
    path = '{}'.format(folder)
    if not os.path.isdir(path):
        os.mkdir(path)

    indices = np.argsort(importances)[::-1]
    with open('{}/SKLearn_RF_important_features_{}.csv'.format(folder, iteration), 'wb') as f:
        writer = csv.writer(f)

        for i in xrange(len(indices)):
            writer.writerow([features[indices[i]], importances[indices[i]]])


def feature_ranker(folder, iteration, sample_size=9000):
    dataset = create_vocab_dataset(sample_size)

    classifier = nltk.NaiveBayesClassifier.train(dataset)
    save_words_to_file(classifier, '{}/NB'.format(folder), iteration)

    X, y = [], []
    for i in dataset:
        X.append(i[0].values())
        y.append(int(i[1]))

    words = dataset[0][0].keys()

    forest = RandomForestClassifier(n_estimators=250, n_jobs=3, random_state=0)
    forest.fit(X, y)
    save_important_features(forest, words, '{}/RF'.format(folder), iteration)


def gather_filenames(path):
    filenames = []
    for file_ in os.listdir(path):
        if file_.endswith('.csv'): filenames.append(path + '/' + file_)
    return filenames


def collect_word_rankings(filename):
    """
    Creates a dictionary of words with their respective ranking.
    i.e.
    {word:rank, word:rank, ...}
    """
    return {row[0]: idx for idx, row in enumerate(read_csv(filename, headers=False))}


def consolidate_word_rankings(folder):
    """
    Creates a dictionary of words with a list of their respective rankings.
    i.e.
    {word:[ranks], word:[ranks], ...}
    """
    big_dict = {}
    for f in folder:
        word_ranks = collect_word_rankings(f)
        for k in word_ranks:
            if k not in big_dict:
                big_dict[k] = [word_ranks[k]]
            else:
                big_dict[k].append(word_ranks[k])
    return big_dict


def combine_dicts(NB_folder, RF_folder):
    """
    Creates a dictionary of words with a tuple of their respective rankings.
    i.e.
    {word:([ranks], [ranks]), word:([ranks], [ranks]), ...}
    """
    NB_dict = consolidate_word_rankings(NB_folder)
    RF_dict = consolidate_word_rankings(RF_folder)
    big_dict = {}
    key_list = NB_dict.keys() + RF_dict.keys()
    for k in key_list:
        if k not in NB_dict:
            big_dict[k] = ([], RF_dict[k])
        elif k not in RF_dict:
            big_dict[k] = (NB_dict[k], [])
        else:
            big_dict[k] = (NB_dict[k], RF_dict[k])
    return big_dict


def ave(num_list):
    # Need to account for the empty list (divide by zero error).
    # Chose 500 as to not punish a word too much for only being in one list.
    return sum(num_list) / float(len(num_list)) if len(num_list) > 0 else punish


def ave_ranks(words):
    """
    Determines average place for each word
    """
    return {item[0]: (ave(item[1][0]), ave(item[1][1])) for item in words.iteritems()}


def order_words(words):
    """
    Orders list based on word's average place between both lists.
    """
    return sorted(words.iteritems(), key=lambda x: ave(x[1]))


def build_word_lists(folder, n, sample_size):
    """"
    Runs NLTK NB classifier to get an ordered list of the most informative features.
    Runs SKLearn Random Forest classifier to get an ordered list of the most important features.
    Saves both lists as NB_feat_##.csv and RF_feat_##.csv respectively.
    """
    # Check if folder exists
    path = os.path.join(os.path.expanduser('~'), 'OneDrive\\RPI\\Summer Project\\URL Classifier\\Dataset', folder)
    if not os.path.isdir(path):
        os.mkdir(path)

    for i in xrange(n):
        feature_ranker(path, '{0:0{1}d}'.format(i, len(str(n-1))), sample_size)
        print "Completed {} iteration(s).".format(i + 1)


def get_ranked_word_list(folder):
    path = os.path.join(os.path.expanduser('~'), 'OneDrive\\RPI\\Summer Project\\URL Classifier\\Dataset', folder)

    # Gather list of filenames
    NB_Folder = gather_filenames(os.path.join(path, folder, 'NB'))
    RF_Folder = gather_filenames(os.path.join(path, folder, 'RF'))

    # Loop through all files:
    # Read files to create dictionary, format: {word:([#,#,...], RF:[#,#,...]), word:(...), ...}
    words = combine_dicts(NB_Folder, RF_Folder)

    # Calculate average place for each word, format: {word:(NB_place, RF_place), word:(#,#), ...}
    ap = ave_ranks(words)

    # Order list based on average of places
    ow = order_words(ap)
    print ow

    # Save ordered word rankings
    with open(os.path.join(path, folder, 'ranked_words.csv'), 'wb') as f:
        writer = csv.writer(f)
        for word in ow:
            writer.writerow([word[0], word[1][0], word[1][1]])


if __name__ == '__main__':
    # build_word_lists('Vocab', 100, 9000)
    # get_ranked_word_list('Vocab')
    pass