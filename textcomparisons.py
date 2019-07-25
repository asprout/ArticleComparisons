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
        self.bow_indices = []
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
        self.bow_indices = []
        for i, sent in enumerate(self.sentences):
            tokens = [token for token in nltk.word_tokenize(sent.lower()) 
                                                        if token.isalnum()]
            if (len(tokens) > 0):
                bow_sentences.append(tokens)
                self.bow_indices.append(i)
        return [dict(Counter(tokens)) for tokens in bow_sentences]

    def get_sentences(self):
        if len(self.bow_indices) > 0:
            return [self.sentences[i] for i in self.bow_indices]
        return None

    def get_bow_sentences(self):
        return self.bow_sentences


class DocumentComparisons:
    def __init__(self, jaccard_threshold = .5, min_sentence = 4):
        self.last_source = None
        self.last_target = None
        self.last_jaccard_matrix = None
        self.last_jaccard_weights = None 
        self.last_jaccard_matches = None
        self.jaccard_threshold = .5
        self.min_sentence = 4

    def jaccard_matrix(self, source, target):
        ''' With an m-sentence source document and n-sentence target, 
        returns a m x n symmetric matrix with jaccard indices between 
        the sentences. If weighted, then the indices within the 
        matrix will be weighted by sentence length. 
        '''
        source_bow = source.get_bow_sentences()
        target_bow = target.get_bow_sentences()
        jac_mat = np.zeros((len(source_bow), len(target_bow)))
        weight_mat = np.zeros(jac_mat.shape)
        for i in range(len(source_bow)):
            for j in range(len(target_bow)):
                jac_mat[i, j] = self.jaccard_index(source_bow[i], 
                                                    target_bow[j])
                weight_mat[i, j] = np.mean([len(source_bow[i]), 
                                            len(target_bow[j])])
        # Stores the last matrix computed to avoid the need for
        # redundant function calls
        self.last_source = source
        self.last_target = target
        self.last_jaccard_matrix = jac_mat
        self.last_jaccard_weights = weight_mat
        return jac_mat

    def get_jaccard_matrix(self, source = None, target = None, 
                                            weighted = False):
        if source is None or target is None:
            return None
        self.jaccard_matrix(source, target)
        if weighted:
            return self.last_jaccard_matrix * self.last_jaccard_weights
        return self.last_jaccard_matrix

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

    def get_match_matrix(self, source = None, target = None):
        ''' Match the sentences between source and target, 
        where multiple sentences can map onto one
        Note: uses jaccard_matrix if available
        '''
        if all([self.last_jaccard_matrix is None, source, target]):
            jac_mat = self.jaccard_matrix(source, target)
        else:
            jac_mat = self.last_jaccard_matrix
        matches = np.zeros(jac_mat.shape)
        # assign sentences to match maximum jaccard indices
        for i in range(matches.shape[0]):
            maxmatch = np.argmax(jac_mat[i, :])
            if jac_mat[i, maxmatch] > self.jaccard_threshold:
                matches[i, maxmatch] = 1
        for j in range(matches.shape[1]):
            maxmatch = np.argmax(jac_mat[:, j])
            if jac_mat[maxmatch, j] > self.jaccard_threshold:
                matches[maxmatch, j] = 1
        # a single sentence can match two IFF the two are adjacent
        for i in range(matches.shape[0]):
            if np.sum(matches[i, :]) > 1:
                matches[i, :] = self.del_nonconsecutives(matches[i, :])
        for j in range(matches.shape[1]):
            if np.sum(matches[:, j]) > 1:
                matches[:, j] = self.del_nonconsecutives(matches[:, j])
        self.last_jaccard_matches = matches
        return matches

    def del_nonconsecutives(self, arr):
        ''' Takes an array and leaves only the maximum consecutive
        non-zero 
        '''
        maxind = 0
        maxcount = 0 
        curind = -1
        curcount = 0
        for i in range(len(arr)):
            if arr[i] == 0:
                curind = -1
                curcount = 0
            else:
                if curind == -1:
                    curind = i
                curcount += 1 
            if curcount > maxcount:
                maxind = curind
                maxcount = curcount 
        return [0] * maxind + [1] * maxcount + [0] * (len(arr) - (maxind + maxcount))

    def print_sentence_matches(self):
        if self.last_jaccard_matches is None:
            return
        matches = self.last_jaccard_matches
        source_sents = self.last_source.get_sentences()
        target_sents = self.last_target.get_sentences()
        for i in range(matches.shape[0]):
            if np.sum(matches[i, :] > 0):
                print(source_sents[i], "\n")
                for j in [j for j in range(matches.shape[1]) if matches[i, j] > 0]:
                    print(target_sents[j], "\n")


class ArticleComparisons:
    def __init__(self):
        # Holds documents in a dictionary by id
        self.docs = {}
        self.comparer = DocumentComparisons()

    def add_doc(self, name, doc):
        self.docs["name"] = doc 

    def add_docs(self, docdict):
        self.docs.update(docdict)

    def jac_score_mat(self, docdict = None, weighted = False):
        ''' Returns the matrix of match scores based on pairwise
        document Jaccard matrices in docdict
        '''
        if docdict is None:
            docdict = self.docs
        score_mat = np.zeros((len(docdict), len(docdict)))
        for i, doc1id in enumerate(docdict):
            for j, doc2id in enumerate(docdict):
                doc1 = docdict[doc1id]
                doc2 = docdict[doc2id]
                if (i > j):
                    score_mat[i, j] = score_mat[j, i]
                else:
                    jac_mat = self.comparer.get_jaccard_matrix(doc1, doc2, 
                                                                weighted)
                    match_mat = self.comparer.get_match_matrix()
                    score_mat[i, j] = np.sum(jac_mat * match_mat) / np.mean(jac_mat.shape)
        return score_mat



