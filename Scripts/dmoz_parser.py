import csv
import gzip
from lxml import etree as ET


__author__ = 'Jamie Fujimoto'


def to_csv(folder, filename):
    '''
    Extracts data from DMOZ gzip file (content.rdf.u8.gz) and writes to csv
    File name:  DMOZ_data.csv
    '''
    
    nsmap = {'d': 'http://purl.org/dc/elements/1.0/'}
    start = False
    count = 0

    with open('{}/{}.csv'.format(folder, filename), 'wb') as csvfile:
        fieldnames = ['url', 'title', 'description', 'priority', 'topic']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        
        with gzip.open('{}/content.rdf.u8.gz'.format(folder), 'rb') as content:
            for event, element in ET.iterparse(content, events=('start','end'), 
                                               tag=['{http://dmoz.org/rdf/}ExternalPage', 
                                                    '{http://dmoz.org/rdf/}topic', 
                                                    '{http://dmoz.org/rdf/}priority']):
                if event == 'start' and element.tag == '{http://dmoz.org/rdf/}ExternalPage':
                    start = True
                elif event == 'end' and element.tag == '{http://dmoz.org/rdf/}ExternalPage':
                    start = False

                title = element.xpath('d:Title/text()', namespaces=nsmap)
                description = element.xpath('d:Description/text()', namespaces=nsmap)
                title, description = title and title[0] or '', description and description[0] or ''

                if start == True:
                    if element.tag == '{http://dmoz.org/rdf/}ExternalPage':
                        url = element.get('about')
                    if element.tag == '{http://dmoz.org/rdf/}topic':
                        topic = element.text
                    if element.tag == '{http://dmoz.org/rdf/}priority':
                        priority = element.text

                elif start == False:
                    writer.writerow({'url':unicode(url).encode("utf-8"), 
                                     'title':unicode(title).encode("utf-8"), 
                                     'description':unicode(description).encode("utf-8"), 
                                     'priority':unicode(priority).encode("utf-8"), 
                                     'topic':unicode(topic).encode("utf-8")})
                    count += 1
                    if count % 1000 == 0:
                        print 'Processed {} items'.format(count)


if __name__ == '__main__':
    to_csv('../Data', 'DMOZ')