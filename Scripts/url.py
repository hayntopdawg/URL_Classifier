import csv
import os
import time

__author__ = 'Jamie Fujimoto'


def read_csv(filepath, headers=True):
    with open(filepath, 'rU') as data:
        if headers:
            reader = csv.DictReader(data)
        else:
            reader = csv.reader(data)
        for row in reader:
            yield row


def create_urlfile(in_file, out_folder, out_file, append=False):
    mode = 'wb'
    out = '{}/{}.csv'.format(out_folder, out_file)
    count = 0
    seen = set()
    # if outfile exists and append == True: mode = append
    if os.path.isfile(out) and append:
        mode = 'ab'

    start = time.time()
    # open outfile
    with open(out, mode) as file:
        writer = csv.DictWriter(file, fieldnames=['url'])
        if mode == 'wb': writer.writeheader()

        # read row of infile
        for idx, row in enumerate(read_csv(in_file, True)):
            # if idx > 0: break
            if idx % 1000 == 0:
                print 'Processed %d items' % idx

            url = row['url']

            # skip url if it has a comma
            if "," in url: continue
            # skip url if it has a newline char
            if "\n" in url: continue
            # skip url if already seen
            if url in seen: continue
            seen.add(url)

            # write url to outfile
            writer.writerow({'url': url})
            count += 1

    finit = time.time()
    delta = finit - start

    print "%s took %0.3f seconds" % (out_file, delta)
    print "%s created successfully." % out_file
    return count


def log_info(filename, lines, append=False):
    mode = 'wb'
    path = os.path.join(os.path.expanduser('~'), 'OneDrive\\RPI\\Summer Project\\URL Classifier\\Data\\URL Files\\info.csv')
    if os.path.isfile(path) and append:
        mode = 'ab'
    with open(path, mode) as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'num_lines'])
        if mode == 'wb': writer.writeheader()
        writer.writerow({'file': '{}.csv'.format(filename),
                         'num_lines': lines})


def get_maxlines(filename):
    path = os.path.join(os.path.expanduser('~'), 'OneDrive\\RPI\\Summer Project\\URL Classifier\\Data\\URL Files\\info.csv')
    for row in read_csv(path, True):
        if row['file'] == filename:
            return int(row['num_lines'])
# print get_maxlines('mal_urls.csv')


if __name__ == "__main__":
    # # Phishing website data filename to be imported
    # # Data from PhishTank
    # count = create_urlfile('../Data/phishtank.csv', '../Data/URL Files', 'mal_urls', append=False)

    # # CyberCrime website data filename to be imported
    # # Data from CyberCrime
    # count += create_urlfile('../Data/cybercrimetracker.csv', '../Data/URL Files', 'mal_urls', append=True)

    # # OpenPhish website data filename to be imported
    # # Data from OpenPhish
    # count += create_urlfile('../Data/openphish.csv', '../Data/URL Files', 'mal_urls', append=True)
    # log_info('mal_urls', count, append=False)

    # Benign website data filename to be imported
    # Data from DMOZ.org
    count = create_urlfile('../Data/DMOZ.csv', '../Data/URL Files', 'ben_urls', append=False)
    log_info('ben_urls', count, append=True)
