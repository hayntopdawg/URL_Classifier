import collections
import csv
import os
import re
import socket
import time
import tldextract

from urlparse import urlparse

# other features to consider: # of special characters (in domain, path, query, url), legitimate TLD

# Labels:
# benign: 0
# malicious: 1


def read_csv(filepath):
    with open(filepath, 'rU') as data:
        reader = csv.reader(data)
        for row in reader:
            yield row


def process_url(url):
    parsed_url = parse_url(url)
    hostname = parsed_url.hostname
    path = parsed_url.path
    query = parsed_url.query

    hostname_tokens = get_tokens(hostname)
    path_tokens = get_tokens(path)
    query_tokens = get_tokens(query)

    processed_url = collections.OrderedDict()

    # Scalar based features
    processed_url['host_len'] = len(hostname)
    processed_url['path_len'] = len(path)
    processed_url['query_len'] = len(query)
    processed_url['url_len'] = len(url)

    processed_url['num_host_dots'] = hostname.count('.')
    processed_url['num_path_dots'] = path.count('.')
    processed_url['num_query_dots'] = query.count('.')
    processed_url['num_url_dots'] = url.count('.')

    processed_url['host_tok_count'] = len(hostname_tokens)
    processed_url['path_tok_count'] = len(path_tokens)
    processed_url['query_tok_count'] = len(query_tokens)

    processed_url['ave_host_tok_len'] = ave_len(hostname_tokens)
    processed_url['ave_path_tok_len'] = ave_len(path_tokens)
    processed_url['ave_query_tok_len'] = ave_len(query_tokens)

    processed_url['max_host_tok_len'] = len(get_max_token_len(hostname_tokens))
    processed_url['max_path_tok_len'] = len(get_max_token_len(path_tokens))
    processed_url['max_query_tok_len'] = len(get_max_token_len(query_tokens))

    # Binary based features
    processed_url['host_digit'] = 1 if has_digit(hostname) else 0
    processed_url['path_digit'] = 1 if has_digit(path) else 0
    processed_url['query_digit'] = 1 if has_digit(query) else 0

    processed_url['host_-'] = 1 if '-' in hostname else 0
    processed_url['path_-'] = 1 if '-' in path else 0
    processed_url['query_-'] = 1 if '-' in query else 0

    processed_url['host_='] = 1 if '=' in hostname else 0
    processed_url['path_='] = 1 if '=' in path else 0
    processed_url['query_='] = 1 if '=' in query else 0

    processed_url['is_ip'] = is_ip(hostname)
    processed_url['sec_sen_word'] = has_sen_sec_tok(path_tokens)
    # processed_url['tld'] = hash(get_tld(url)  # consider making it binary (valid or not valid))

    fw = get_feature_words('Vocab', 'ranked_words')
    for word in fw:
        processed_url[word] = 1 if word in url else 0

    return processed_url


def parse_url(url):
    return urlparse(url)


# def get_tld(url):
#     return tldextract.extract(url).suffix


def get_tokens(string):
    try:
        # filter out the empty string
        return filter(None, re.split('\W', string))
    except TypeError:
        return []


# http://stackoverflow.com/questions/15772371/finding-average-length-of-items-in-a-list-python
def ave_len(tokens):
    lens = [len(i) for i in tokens]
    return 0 if len(lens) == 0 else (float(sum(lens)) / len(lens))


def get_max_token_len(tokens):
    if tokens:
        return max(tokens, key=len)
    else:
        return []


#http://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
def has_digit(string):
    return any(char.isdigit() for char in string)


def is_ip(address):
    try:
        socket.inet_aton(address)
        return 1
    except socket.error:
        return 0


def has_sen_sec_tok(path_tokens):
    suspicious_words = ['confirm', 'account', 'banking', 'secure', 'ebayisapi', 'webscr', 'login', 'signin']
    for token in path_tokens:
        if token.lower() in suspicious_words:
            return 1
    return 0


def get_feature_words(folder, file):
    words = []
    for i, row in enumerate(read_csv('Dataset/{}/{}.csv'.format(folder, file))):
        # if i > 0: break
        if i > 70: break  # get the first 70 words
        words.append(row[0])
    return words