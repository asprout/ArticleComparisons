import string # remove punctuation
import numpy as np 
from collections import Counter # contains word counter function
import nltk # tokenize words and sentences based on language
# Following line may be necessary to run nltk calls:
# nltk.download('punkt')

class Document:
    def __init__(self, text, text_encoding = "utf8", text_language = "english"):
        self.text = text 
        self.text_encoding = text_encoding
        self.text_language = text_language
        self.paragraphs = self.parse_paragraphs()
        self.sentences = self.parse_sentences()
        self.bow_sentences = self.parse_bow_sentences()

    def __str__(self):
        return self.text

    def parse_paragraphs(self, para_sep = "\n"):
        return self.text.split(para_sep)

    def parse_sentences(self):
        sentences = []
        for para in self.paragraphs:
            sentences.extend(nltk.sent_tokenize(para, self.text_language))
        return sentences

    def parse_bow_sentences(self):
        ''' Returns a list of dictionaries for each sentence, 
        where the keys are words and values are counts of the words in 
        the sentence 
        '''
        bow_sentences = []
        for sent in self.sentences:
            tokens = [token for token in nltk.word_tokenize(sent.lower()) 
                                                        if token.isalnum()]
            bow_sentences.append(tokens)
        return [dict(Counter(tokens)) for tokens in bow_sentences]

    def get_bow_sentences(self):
        return self.bow_sentences


class DocumentComparisons:
    def __init__(self, jaccard_threshold = .9, min_sentence = 4):
        self.jaccard_threshold = .9
        self.min_sentence = 4

    def jaccard_index(self, bow_a, bow_b):
        ''' Takes two BOW dictionaries and returns the jaccard index, defined as
            Jaccard(A, B) = |A and B|/|A or B|
        '''
        intsec_words = set(bow_a) & set(bow_b)
        intsec_vals = [min(bow_a.get(word), bow_b.get(word)) for 
                                                        word in intsec_words]
        intsec = sum(intsec_vals)
        union = sum(bow_a.values()) + sum(bow_b.values()) - intsec
        return float(intsec / union)


class ArticleComparisons:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.comparer = DocumentComparisons()

    def jaccard_matrix(self):
        ''' With an m-sentence source document and n-sentence target, 
        returns a m x n symmetric matrix with jaccard indices between 
        the sentences 
        '''
        source_bow = self.source.get_bow_sentences()
        target_bow = self.target.get_bow_sentences()
        jac_mat = np.zeros((len(source_bow), len(target_bow)))
        for i in range(len(source_bow)):
            for j in range(len(target_bow)):
                jac_mat[i, j] = self.comparer.jaccard_index(source_bow[i], 
                                                            target_bow[j])
        return jac_mat



