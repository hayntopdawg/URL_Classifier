import csv
import re

__author__ = 'Jamie Fujimoto'


def txt_to_csv(in_file, out_file):
    with open(out_file, 'wb') as fout:
        writer = csv.DictWriter(fout, fieldnames=['url'])
        writer.writeheader()

        with open(in_file, 'rb') as fin:
            for line in fin:
                writer.writerow({'url':line.strip()})


def html_to_csv(in_file, out_file):
    urls = []
    with open(in_file, 'rb') as fin:
        for line in fin:
            m = re.findall('>(.*?)<', line)
            urls = filter(None, m)  # filters out the empty string

    with open(out_file, 'wb') as fout:
        writer = csv.DictWriter(fout, fieldnames=['url'])
        writer.writeheader()

        for url in urls:
            writer.writerow({'url': url})


if __name__ == "__main__":
    txt_to_csv('../Data/openphish.txt', '../Data/openphish.csv')
    html_to_csv('../Data/cybercrimetracker.txt', '../Data/cybercrimetracker.csv')