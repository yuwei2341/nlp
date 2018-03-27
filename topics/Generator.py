import os
import sys
import glob
import random
import collections
import re

class TextGenerator(object):
    '''
    '''

    EMPTY_STRING = ''
    CUTOFF = 100
    PATTERN = re.compile('([^\s\w]|_)+')

    def __init__(self):
        '''
        '''
        
        self.documents_raw = []
        self.processed = []
        self.mc = collections.defaultdict(list)
        self.paragraphs = []
        
    def _get_params(self):
        param = sys.argv
        self.input_directory_name = param[1]
        self.P = int(param[2])
        self.N = int(param[3])

    def _read_file(self):
        files = glob.glob(os.path.join(self.input_directory_name, '*'))
        for i in files:
            with open(i, 'r') as f:
                self.documents_raw.append(f.read())
    
    @staticmethod
    def _parse_doc(document):
        '''Parse a single paragraph
        Remove whitespaces, punctuations    
        '''

        tokenizer = RegexpTokenizer(r'\w+')
        intermediate = tokenizer.tokenize(document)    
        parsed = [i.lower() for i in intermediate]

        return parsed

    def _parse_collection(self):
        # parse all docs  
        
        self.documents_raw
        for raw_text in self.documents_raw:
            doc = self.PATTERN.sub('', unicode(raw_text, 'ascii', 'ignore').lower())
            paragraphs = doc.split('\r\n\r\n')        
            for paragraph in paragraphs:    
                parsed = paragraph.split()
                if parsed:
                    self.processed.append(parsed)
        
    def _process_input(self):

        key_start = [self.EMPTY_STRING] * self.P

        for line in self.processed:
            key = key_start
            for word in line:
                self.mc[tuple(key)].append(word)
                key = key[1:] + [word]
            self.mc[tuple(key)].append(self.EMPTY_STRING)

    def _generate_output(self):
        
        key_start = [self.EMPTY_STRING] * self.P
        self.paragraphs = []

        # Only start with paragraph starting prifixes in the original text
        j = 0
        while j < self.N:
            key = key_start
            paragraph = []
            for i in range(self.CUTOFF):
                suf_start = self.mc[tuple(key)]
                next_word = random.choice(suf_start)
                if next_word is self.EMPTY_STRING:
                    break
                paragraph.append(next_word)
                key = key[1:] + [next_word]
            joined = ' '.join(paragraph)
            if joined:
                self.paragraphs.append(joined)
                j += 1
    
    def _print_result(self):
        for i in self.paragraphs:
            print i + '\n'
        print 'Number of prefixes: {}'.format(len(self.mc))
        
    def run(self):
        
        self._get_params()
        self._read_file()
        self._parse_collection()
        self._process_input()
        self._generate_output()
        self._print_result()

if __name__ == '__main__':

    g = TextGenerator()
    g.run()