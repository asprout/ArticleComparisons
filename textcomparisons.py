import string # remove punctuation
import numpy as np 
from collections import Counter # contains word counter function
import nltk # tokenize words and sentences based on language
# Following line may be necessary to run nltk calls:
# nltk.download('punkt')

class Document:
    def __init__(self, text, clean = False, stem = True,
                text_encoding = "utf8", text_language = "english"):
        self.text = text 
        # Boolean representing if the article is 'clean'
        # To change how parse_paragraphs is called
        self.clean = clean 
        self.stem = stem
        self.text_encoding = text_encoding
        self.text_language = text_language
        self.paragraphs = self.parse_paragraphs()
        self.sentences = self.parse_sentences()
        self.bow_indices = []
        self.bow_sentences = self.parse_bow_sentences()

    def __str__(self):
        if self.clean:
            return self.text
        return ("\n").join(self.paragraphs)

    def parse_paragraphs(self, para_sep = "\n"):
        if self.clean:
            self.paragraphs = self.text.split(para_sep)
        else: # Clean up bad text from scraped article
            paras = self.text.split("###")
            paras = [p for p in paras if len(p.split()) > 3 and not p.isupper()]
            self.paragraphs = paras 
        return self.paragraphs

    def parse_sentences(self):
        sentences = []
        for para in self.paragraphs:
            sentences.extend(nltk.sent_tokenize(para, self.text_language))
        self.sentences = sentences if self.clean else list(Counter(sentences))
        return self.sentences

    def parse_bow_sentences(self):
        ''' Returns a list of dictionaries for each sentence, 
        where the keys are words and values are counts within the sentences
        '''
        bow_sentences = []
        bow_indices = []
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        ps = nltk.PorterStemmer()
        for i, sent in enumerate(self.sentences):
            tokens = [token for token in tokenizer.tokenize(sent)]
            if self.stem:
                tokens = [ps.stem(token) for token in tokens]
            if len(tokens) > 0 and tokens not in bow_sentences:
                bow_sentences.append(tokens)
                bow_indices.append(i)
        self.bow_sentences = [dict(Counter(tokens)) for tokens in bow_sentences]
        self.bow_indices = bow_indices 
        return self.bow_sentences

    def get_sentences(self):
        ''' Returns the sentences from the article that correspond to actual
        text (i.e. len(tokens) > 0)
        '''
        if len(self.bow_indices) > 0:
            return [self.sentences[i] for i in self.bow_indices]
        return None

    def get_bow_sentences(self):
        return self.bow_sentences


class DocumentComparisons:
    def __init__(self, jaccard_threshold = .5, same_threshold = .9):
        self.last_source = None
        self.last_target = None
        self.last_jaccard_matrix = None
        self.last_jaccard_weights = None 
        self.last_jaccard_matches = None
        self.jaccard_threshold = jaccard_threshold
        self.same_threshold = same_threshold # When two sentences are identical

    def jaccard_matrix(self, source = None, target = None, weighted = False):
        ''' With an m-sentence source document and n-sentence target, 
        returns a m x n symmetric matrix with jaccard indices between 
        the sentences. If weighted, then the indices within the 
        matrix will be weighted by average pairwise sentence length. 
        '''
        if source is None or target is None: # Just return stored matrices
            if weighted:
                return self.last_jaccard_weights * self.last_jaccard_matrix
            return self.last_jaccard_matrix
        # Otherwise compute matrices given the source and target 
        source_bow = source.get_bow_sentences()
        target_bow = target.get_bow_sentences()
        jac_mat = np.zeros((len(source_bow), len(target_bow)))
        weight_mat = np.zeros(jac_mat.shape)
        for i in range(len(source_bow)):
            for j in range(len(target_bow)):
                jac_mat[i, j] = self.jaccard_index(source_bow[i], target_bow[j])
                weight_mat[i, j] = np.mean([len(source_bow[i]), 
                                            len(target_bow[j])])
        # Stores the last matrix computed to avoid redundant work
        self.last_source = source
        self.last_target = target
        self.last_jaccard_matrix = jac_mat
        self.last_jaccard_weights = weight_mat
        if weighted:
            return self.last_jaccard_weights * self.last_jaccard_matrix
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

    def match_matrix(self, source = None, target = None, threshold = None):
        ''' Match the sentences between source and target, 
        where multiple sentences can map onto one
        Note: uses jaccard_matrix if available
        '''
        if threshold is None:
            threshold = self.jaccard_threshold
        jac_mat = self.jaccard_matrix(source, target)
        matches = np.zeros(jac_mat.shape)
        # assign sentences to match maximum jaccard indices
        for i in range(matches.shape[0]):
            maxmatch = np.argmax(jac_mat[i, :])
            if jac_mat[i, maxmatch] > threshold:
                matches[i, maxmatch] = 1
        for j in range(matches.shape[1]):
            maxmatch = np.argmax(jac_mat[:, j])
            if jac_mat[maxmatch, j] > threshold:
                matches[maxmatch, j] = 1

        for i in range(matches.shape[0]):
            matches[i, :] = self.weigh_matches(matches[i, :], jac_mat[i, :])
            #matches[i, :] = self.del_nonconsecutives(matches[i, :])
        for j in range(matches.shape[1]):
            matches[:, j] = self.weigh_matches(matches[:, j], jac_mat[:, j])
            #matches[:, j] = self.del_nonconsecutives(matches[:, j])

        self.last_jaccard_matches = matches
        self.jaccard_threshold = threshold
        return matches

    def get_match_matrix(self):
        return self.last_jaccard_matches

    def weigh_matches(self, matches, jaccards):
        ''' if the sum is > 1, normalizes arr such that it sums to one:
        if the largest value is above a threshold, sets the largest to one, 
        otherwise divides by the sum
        '''
        total = np.sum(matches)
        if total <= 1:
            return matches
        maxmatch = np.argmax(jaccards)
        if jaccards[maxmatch] > self.same_threshold:
            return [0] * maxmatch + [1] + [0] * (len(jaccards) - maxmatch - 1)
        return matches/total

    def del_nonconsecutives(self, arr):
        ''' Takes an array and leaves only the maximum consecutive sequence
        Note: assumes positive numbers only
        '''
        if np.sum(arr) <= 0: # No sequence
            return arr

        maxind = 0 # Index of start of sequence
        maxcount = 0 
        curcount = 0
        for i in range(len(arr)):
            if arr[i] == 0:
                curcount = 0
            else:
                curcount += 1 
            if curcount > maxcount:
                maxind = i - curcount + 1
                maxcount = curcount 
        return [0] * maxind + [1] * maxcount + [0] * (len(arr) - (maxind + maxcount))

    def print_sentence_matches(self, threshold = None):
        if self.last_jaccard_matches is None:
            return
        if threshold is None:
            threshold = self.jaccard_threshold
        matches = self.match_matrix(threshold = threshold)
        jac_mat = self.last_jaccard_matrix
        source_sents = self.last_source.get_sentences()
        target_sents = self.last_target.get_sentences()
        for i in range(matches.shape[0]):
            if np.sum(matches[i, :] > 0):
                print("S", i, ":", source_sents[i], "\n")
                for j in np.nonzero(matches[i, :])[0]:
                    print("\tT", j, np.round(jac_mat[i, j], 2), ":", target_sents[j], "\n")


class ArticleComparisons:
    def __init__(self, docs = {}):
        self.docs = docs # Holds documents in a dictionary by id
        self.comparer = DocumentComparisons()
        self.score_mat = None 

    def add_doc(self, name, doc):
        self.docs["name"] = doc 

    def add_docs(self, docs):
        self.docs.update(docs)

    def jac_score_mat(self, docs = None, weighted = False):
        ''' Returns the matrix of match scores based on pairwise
        document Jaccard matrices in docs
        '''
        if docs is None: # Return stored information 
            if self.score_mat is None and len(self.docs) > 0:
                return jac_score_mat(self.docs, weighted)
            return self.score_mat 

        score_mat = np.zeros((len(docs), len(docs)))
        for i, doc1id in enumerate(docs):
            for j, doc2id in enumerate(docs):
                doc1 = docs[doc1id]
                doc2 = docs[doc2id]
                if (i > j):
                    score_mat[i, j] = score_mat[j, i]
                else:
                    jac_mat = self.comparer.jaccard_matrix(doc1, doc2, weighted)
                    match_mat = self.comparer.match_matrix()
                    jac_score = np.sum(jac_mat * match_mat)
                    score_mat[i, j] = jac_score / np.mean(jac_mat.shape)
        self.score_mat = score_mat 
        return score_mat



