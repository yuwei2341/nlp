#!/usr/bin/env python3

import glob
from collections import defaultdict

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy


class QuranCorpus:
    '''Class for reading Quran corpus and preprocessing
    Preprocessing includes: tokenization, removing stop words, lemmatization
    '''
    def __init__(self, is_remove_basamal=True, is_stop=True, additional_stops=None, is_stemming=False):

        self.is_remove_basamal = is_remove_basamal
        self.is_stop = is_stop
        self.additional_stops = additional_stops
        self.is_stemming = is_stemming

        self.sura_medina = [2, 3, 4, 5, 8, 9, 22, 24, 33, 47, 48, 49, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 98, 110]
        self.sura_mecca_first = [96, 74, 111, 106, 108, 104, 107, 102, 105, 92, 90, 94, 93, 97, 86, 91, 80, 68, 87, 95, 103, 
                                 85, 73, 101, 99, 82, 81, 53, 84, 100, 79, 77, 78, 88, 89, 75, 83, 69, 51, 52, 56, 70, 55, 112, 109, 113, 114, 1]
        self.sura_mecca_second = [54, 37, 71, 76, 44, 50, 20, 26, 15, 19, 38, 36, 43, 72, 67, 23, 21, 25, 17, 27, 18]
        self.sura_mecca_third = [32, 41, 45, 16, 30, 11, 14, 12, 40, 28, 39, 29, 31, 42, 10, 34, 35, 7, 46, 6, 13]
        self.ALL_SURAH_INDICES = list(range(1, 115))
        self.BASMALA = 'in the name of allah, the gracious, the merciful.'

        self.documents = [] 
        self.i_surah = []
        self.i_verse = []

    def _remove_basmala(self):
        """Remove BASMALA from documents
        If documents is a list of lists (book of surahs), BASMALA is an element in the
        inner list. If documents is a list of verses, then the index of the verse must
        be provided
        """
        for doc in self.documents_by_surah_as_list:
            if self.BASMALA in doc:
                doc.remove(self.BASMALA)

        for i, doc in enumerate(self.documents_by_verse):
            if doc == self.BASMALA:
                del self.i_verse[i]
                del self.i_surah[i]
                del self.documents_by_verse[i]

    def read_in_quran(self, suras_indices=None, data_folder='../data/quran-verse-by-verse-text/'):
        """Read in quran from files each of which is a verse and is named by surah-verse
        Returns:
            documents: list of lists of verses (a book of surahs of verses)
            i_surah: list of surah number corresponding to each verse in documents
            i_verse: list of verse number under each surah corresponding to each verse
                in documents
        """
        suras_indices = suras_indices or self.ALL_SURAH_INDICES
        documents = [] 
        i_surah = []
        i_verse = []
        for chapter in suras_indices:
            # files contain file names of the same surah
            files = sorted(glob.glob(data_folder + str(chapter).zfill(3) + '*'))
            # Remove non ascii and change to lower case; not necessary with RegexpTokenizer(r'\w+')
            surah = []
            for ind_verse, f in enumerate(files):
                with open(f, 'r', encoding='utf-8') as hf: # In windows the default is not utf-8
                    text = hf.read()
                    verse = text.encode('ascii', errors='ignore').lower().decode('utf-8')
                    surah.append(verse)
                    i_surah.append(chapter)
                    i_verse.append(ind_verse + 1)
            documents.append(surah)  

        self.documents_by_surah_as_list = documents 
        self.documents_by_verse = [verse for surah in documents for verse in surah]
        self.i_surah = i_surah 
        self.i_verse = i_verse

        if self.is_remove_basamal:
            self._remove_basmala()
        
        self.documents_by_surah = [' '.join(surah) for surah in self.documents_by_surah_as_list]


    @staticmethod
    def tokenize_docs(docs):
        '''removing whitespaces and punctuations, and tokenize docs '''
        tokenizer = RegexpTokenizer(r'\w+')
        for document in docs:
            yield(tokenizer.tokenize(document)) 

    @staticmethod
    def remove_stopwords(docs, additional_stops=None):
        """Remove stop words """
        stop = stopwords.words('english')
        if additional_stops is not None:
            stop = stop + additional_stops 
        return [[word for word in doc if word not in stop] for doc in docs]

    @staticmethod
    def lemmatization(docs, is_stemming=False):
        """Reduce a word to its base or root form (lemma)
        """        
        stem_map = defaultdict(lambda : defaultdict(int))
        parsed_docs = []
        if is_stemming:  # do stemming
            stemmer = PorterStemmer()
            for doc in docs:
                parsed_doc = []
                for word in doc:
                    stemmed = stemmer.stem(word)
                    parsed_doc.append(stemmed)
                    stem_map[stemmed][word] += 1
                parsed_docs.append(parsed_doc)
        else:  # do lemmatization with stacy
            # Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  # en_core_web_sm.load(disable=['parser', 'ner'])
            # Extract the lemma for each token and join
            for doc in docs:
                nlp_parsed = nlp(' '.join(doc))
                parsed_doc = []
                for word in nlp_parsed: 
                    lemma = word.lemma_
                    parsed_doc.append(lemma)
                    stem_map[lemma][word.text] += 1
                parsed_docs.append(parsed_doc)
        return parsed_docs, stem_map
    
    def parse_docs(self, is_by_verse=True):
        """Tokenize docs, remove stopwords (or not), and do lemmatization
        Return:
            intermediates: tokens (stopwords removed by default)
            processed: tokens after lemmatization
            stem_map: mapping stem-word and number of times a word appeared
        """
        docs = self.documents_by_verse if is_by_verse else self.documents_by_surah
        intermediates = list(QuranCorpus.tokenize_docs(docs))
        if self.is_stop:
            intermediates = QuranCorpus.remove_stopwords(intermediates, self.additional_stops)
        processed, stem_map = QuranCorpus.lemmatization(intermediates, self.is_stemming)
        return intermediates, processed, stem_map