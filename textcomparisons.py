import numpy as np 
import matplotlib.pyplot as plt
import documents
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import time

# Helper functions

def notNone(lst):
    """ returns True if at least one item is not None """
    return sum([li is not None for li in lst]) > 0

def ceilzero(x):
    return max(x, 0)

def flatten(vec):
    return [val for sublist in vec for val in sublist]

class DocumentComparisons:
    ''' A class to make pairwise document similarity comparisons 
    '''
    def __init__(self, thresh_jaccard = .5, thresh_same_sent = .9):
        self.thresh_jaccard = thresh_jaccard # min Jaccard index to be considered a sentence match
        self.thresh_same_sent = thresh_same_sent # Jaccard index to be considered a definite match
        # Store information about the last comparison done
        self.last_source = None # Note: code currently does not differentiate b/w source and target documents
        self.last_target = None
        self.last_jaccard_matrix = None # Matrix of pairwise sentence Jaccard indices
        self.last_jaccard_weights = None # Matrix of weights by sentence-length
        self.last_jaccard_matches = None # Matrix of pairwise sentence matches (see match_matrix functions)
        self.matches_obsolete = True # If the match matrix needs to be updated for Jaccard matrix

    def jaccard_index(self, bow_a, bow_b, counts = False, visualize = False):
        ''' Takes two BOW dictionaries and returns the jaccard index, defined as
            Jaccard(A, B) = |A and B|/|A or B|
            if counts is true, then uses word count, not just unique words
        '''
        intsec_words = set(bow_a) & set(bow_b)
        if counts:
            intsec_vals = [min(bow_a[word], bow_b[word]) for word in intsec_words]
            intsec = sum(intsec_vals)
            union = sum(bow_a.values()) + sum(bow_b.values()) - intsec
        else:
            intsec = len(intsec_words)
            union = len(bow_a) + len(bow_b) - intsec
        index = float(intsec / max(1.0, union))
        if visualize:
            print("Jaccard Index:", index, "with counts", counts)
            print("I:", intsec_words)
            print("A-B:", set(bow_a) - set(bow_b), "\nB-A:", set(bow_b) - set(bow_a))
        return index

    def jaccard_matrix(self, source, target):
        ''' With an m-sentence source document and n-sentence target, 
        returns a m x n symmetric matrix with pairwise jaccard indices
        '''
        if source is None:
            source = self.last_source
        if target is None:
            target = self.last_target
        source_bow = source.get_bow_sentences()
        target_bow = target.get_bow_sentences()
        if len(source_bow) < 1 or len(target_bow) < 1:
            return None 
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
        self.matches_obsolete = True
        return self.last_jaccard_matrix

    def get_jaccard_matrix(self, source = None, target = None, weighted = False):
        """ Returns the last Jaccard matrix or computes a new one if source
        or target is provided. If weighted, then the indices within the 
        matrix will be weighted by average pairwise sentence length. 
        """
        if self.last_jaccard_matrix is None or notNone([source, target]):
            self.jaccard_matrix(source, target)
        if weighted: 
            return self.last_jaccard_matrix * self.last_jaccard_weights
        return self.last_jaccard_matrix

    def match_matrix(self, source = None, target = None, thresh_jaccard = None):
        ''' Symmetric matrix matching pairwise sentences between two documents:
        1. Creates a match matrix: 1 if pairwise Jaccards >= thres_jaccard, else 0
        2. Weighs match matrix. For each sentence in source:
            2a. If the highest match > thresh_same_sent, sets match to 1 and all 
            other corresponding matches to source/target sentences to 0
            2b. Otherwise, normalizes the row in the match matrix to sum to 1
            2c. Repeat for target
        '''
        jac_mat = self.get_jaccard_matrix(source, target)
        if thresh_jaccard is not None:
            self.thresh_jaccard = thresh_jaccard
        matches = 1.0 * (jac_mat >= self.thresh_jaccard)
        matches = self.weigh_matches(matches, jac_mat) # Weigh rows (source sents)
        matches = (self.weigh_matches(matches.T, jac_mat.T)).T # Weigh columns
        self.last_jaccard_matches = matches
        self.matches_obsolete = False
        return self.last_jaccard_matches

    def get_match_matrix(self, source = None, target = None, thresh_jaccard = None):
        ''' Returns the last computed match matrix, if not obsolete,
        else computes and returns a match matrix. 
        '''
        if any([self.matches_obsolete, notNone([source, target]), 
            thresh_jaccard is not None and thresh_jaccard != self.thresh_jaccard]):
            return self.match_matrix(source, target, thresh_jaccard)
        return self.last_jaccard_matches

    def weigh_matches(self, matches, jaccards):
        ''' Goes through the rows of the match matrix. For each row:
        if the largest value is above the threshold for same sentences,
            sets corresponding match value to one and
            all other indices in the same row and column to 0
        otherwise, divides by the sum such that the normalized sum is 1
        '''
        for i in range(matches.shape[0]):
            argmax = np.argmax(jaccards[i, :]) 
            if jaccards[i, argmax] >= self.thresh_same_sent:
                matches[i, :] = [0] * matches.shape[1]
                matches[:, argmax] = [0] * matches.shape[0]
                matches[i, argmax] = 1
            rowsum = np.sum(matches[i, :])
            if rowsum > 1:
                matches[i, :] = matches[i, :] / rowsum 
        return matches

    def jaccard_score(self, source = None, target = None, weighted = False):
        jac_mat = self.get_jaccard_matrix(source, target, weighted = weighted)
        if jac_mat is None:
            return 0 # No valid sentences in either source or target
        match_mat = self.get_match_matrix(source, target)
        jac_score = np.sum(jac_mat * match_mat)
        #return jac_score / np.mean(jac_mat.shape) 
        return jac_score / np.min(jac_mat.shape) # count snippets as duplicates

    def print_sentence_matches(self, thresh_jaccard = None):
        matches = self.get_match_matrix(thresh_jaccard = thresh_jaccard)
        if matches is None:
            return
        jac_mat = self.get_jaccard_matrix()
        source_sents = self.last_source.get_sentences()
        target_sents = self.last_target.get_sentences()
        for i in range(matches.shape[0]):
            if np.sum(matches[i, :] > 0):
                print("S", i, ":", source_sents[i], "\n")
                for j in np.nonzero(matches[i, :])[0]:
                    print("\tT", j, np.round(jac_mat[i, j], 2), ":", target_sents[j], "\n")

class ArticleComparisons(DocumentComparisons):
    ''' A specific DocumentComparisons instance for dealing with articles
    '''
    def __init__(self, thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .25):
        DocumentComparisons.__init__(self, thresh_jaccard, thresh_same_sent)
        self.docs = {}
        self.score_mat = None # pairwise document similarity score matrix 
        self.thresh_same_doc = thresh_same_doc # similarity score to be considered same article
        self.clusters = None # list of clusters, indexed by document
        self.hclust = None # hierarchical agglo. clustering object

    def update_docdict(self, docs):
        """ If docs is not None, updates the documents stored in class instance, 
        and returns whether or not the class instance was updated
        """
        if docs is not None and docs is not self.docs:
            self.docs = docs 
            return True 
        return False

    def jac_score_mat(self, docs = None, progress = True):
        ''' Returns stored matrix of pairwise document match stores, 
        or computes and returns new matrix if docs is not None
        '''
        start = time.time()
        if self.score_mat is not None and not self.update_docdict(docs):
            return self.score_mat
        if docs is None:
            if self.docs is None: 
                return 
            docs = self.docs 

        score_mat = np.zeros((len(docs), len(docs)))

        for i, doc1id in enumerate(docs):
            if progress and len(docs) > 100 and round(i % (len(docs) / 10)) == 0:
                print(i, "/", len(docs), "done,", round(time.time() - start, 2), "seconds elapsed")
            for j, doc2id in enumerate(docs):
                if (i > j):
                    score_mat[i, j] = score_mat[j, i]
                elif (i == j): # comment this out to make sure self-similarity scores are 1
                    score_mat[i, j] = 1.0 # leave during actual runs to optimize for runtime
                else:
                    score_mat[i, j] = self.jaccard_score(docs[doc1id], docs[doc2id])
        self.score_mat = score_mat 
        return score_mat

    def cluster_articles(self, docs = None, plot = False):
        ''' Runs hierarchical agglomerative clustering on pairwise document
        distances, computed as 1 - document similarity scores (defined
        as the matrix of match-weighted pairwise sentence Jaccard indices).
        linkage = single: A document is assigned to a cluster if any document 
            in the cluster meets the distance threshold.
        linkage = complete: '...' if all documents in cluster meets threshold
        '''
        score_mat = self.jac_score_mat(docs)
        if score_mat is None: return 
        dist_mat = np.vectorize(ceilzero)(1 - score_mat)
        self.hclust = hierarchy.linkage(squareform(dist_mat), method = 'single')
        if plot:
            hierarchy.dendrogram(self.hclust)
            plt.ylabel("Distance")
        return self.hclust 

    def get_article_clustering(self, docs = None, plot = False):
        ''' Returns clustering object, indexed by document
        '''
        if self.update_docdict(docs):
            self.hclust = self.cluster_articles(docs, plot)
        return self.hclust

    def get_article_clusters(self, docs = None, thresh_same_doc = None, plot = False):
        ''' Returns list of clusters, indexed by document
        '''
        self.get_article_clustering(docs, plot)
        if thresh_same_doc is not None and thresh_same_doc != self.thresh_same_doc:
            self.thresh_same_doc = thresh_same_doc
        self.clusters = flatten(hierarchy.cut_tree(self.hclust, height = 1 - self.thresh_same_doc))
        return self.clusters

    def prop_unique_clusters(self, thresh_same_doc = None, inds = None):
        self.get_article_clusters(thresh_same_doc = thresh_same_doc)
        if self.clusters is not None:
            clusters = self.clusters
            if inds is not None and len(inds) > 0:
                clusters = [clusters[i] for i in inds]
            return len(np.unique(clusters))/len(clusters)

    def display_mat(self, mat, normalize = False, xlabs = None, ylabs = None):
        """ Uses matplotlib to display mat; 
        if normalize is True, then coloring is normalized by range of values
        """
        fig = plt.figure(figsize = (20, 20))
        ax = fig.add_subplot(111)
        if normalize:
            ax.matshow(mat, cmap = plt.cm.Blues)
        else:
            ax.matshow(mat, cmap = plt.cm.Blues, vmin = 0, vmax = 1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, round(mat[i, j], 2), va = "center", ha = "center")
        ax.set_xticks([x for x in range(-1, mat.shape[1] + 1)])
        ax.set_yticks([x for x in range(-1, mat.shape[0] + 1)])
        if xlabs is not None: 
            ax.set_xticklabels([''] + xlabs)
            plt.xticks(rotation = 90)
            if ylabs is None and mat.shape[0] == mat.shape[1]:
                ylabs = xlabs
        if ylabs is not None:
            ax.set_yticklabels([''] + ylabs)



