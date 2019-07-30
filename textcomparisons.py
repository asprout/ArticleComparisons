import numpy as np 
import matplotlib.pyplot as plt
import documents
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import time

class DocumentComparisons:
    def __init__(self, thresh_jaccard = .25, thresh_same_sent = .9):
        self.thresh_jaccard = thresh_jaccard
        self.thresh_same_sent = thresh_same_sent 
        # Store information about the last comparison done
        self.last_source = None
        self.last_target = None
        self.last_jaccard_matrix = None
        self.last_jaccard_weights = None 
        self.last_jaccard_matches = None

    def jaccard_index(self, bow_a, bow_b, visualize = False):
        ''' Takes two BOW dictionaries and returns the jaccard index, defined as
            Jaccard(A, B) = |A and B|/|A or B|
        '''
        intsec_words = set(bow_a) & set(bow_b)
        intsec_vals = [min(bow_a[word], bow_b[word]) for word in intsec_words]
        intsec = sum(intsec_vals)
        union = sum(bow_a.values()) + sum(bow_b.values()) - intsec
        index = float(intsec / max(1.0, union))
        if visualize:
            print("Jaccard Index:", index)
            print("U:", intsec_words)
            print("I:", set(bow_a) - set(bow_b), "\n", set(bow_b) - set(bow_a))
        return index

    def jaccard_matrix(self, source, target):
        ''' With an m-sentence source document and n-sentence target, 
        returns a m x n symmetric matrix with pairwise jaccard indices
        '''
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
        return self.last_jaccard_matrix

    def get_jaccard_matrix(self, source = None, target = None, weighted = False):
        """ Returns the last Jaccard matrix or computes a new one if source
        or target is provided. If weighted, then the indices within the 
        matrix will be weighted by average pairwise sentence length. 
        """
        if source is not None or target is not None:
            self.jaccard_matrix(source, target)
        if weighted: 
            return self.last_jaccard_matrix * self.last_jaccard_weights
        return self.last_jaccard_matrix

    def match_matrix(self, source = None, target = None, thresh_jaccard = None):
        ''' Match the sentences between source and target, 
        where multiple sentences can map onto one
        Note: uses jaccard_matrix if available
        '''
        if thresh_jaccard is not None:
            self.thresh_jaccard = thresh_jaccard
        jac_mat = self.get_jaccard_matrix(source, target)
        matches = np.zeros(jac_mat.shape)
        # assign sentences to match maximum jaccard indices
        for i in range(matches.shape[0]):
            maxmatch = np.argmax(jac_mat[i, :])
            if jac_mat[i, maxmatch] > self.thresh_jaccard:
                matches[i, maxmatch] = 1
        for j in range(matches.shape[1]):
            maxmatch = np.argmax(jac_mat[:, j])
            if jac_mat[maxmatch, j] > self.thresh_jaccard:
                matches[maxmatch, j] = 1

        for i in range(matches.shape[0]):
            matches[i, :] = self.weigh_matches(matches[i, :], jac_mat[i, :])
        for j in range(matches.shape[1]):
            matches[:, j] = self.weigh_matches(matches[:, j], jac_mat[:, j])

        self.last_jaccard_matches = matches
        return matches

    def get_match_matrix(self, source = None, target = None, thresh_jaccard = None):
        if any([source is not None, target is not None, thresh_jaccard is not None]):
            self.match_matrix(source, target, thresh_jaccard)
        return self.last_jaccard_matches

    def weigh_matches(self, matches, jaccards):
        ''' if the sum is > 1, normalizes arr such that it sums to one:
        if the largest value is above a threshold, sets the largest to one, 
        otherwise divides by the sum
        '''
        total = np.sum(matches)
        if total <= 1:
            return matches
        argmax = np.argmax(jaccards) # maxmatch = jaccards[argmax] 
        if (self.thresh_same_sent is not None and 
                                    jaccards[argmax] > self.thresh_same_sent):
            return [0] * argmax + [1] + [0] * (len(jaccards) - argmax - 1)
        return matches/total

    def jaccard_score(self, source = None, target = None, weighted = False):
        match_mat = self.get_match_matrix(source, target)
        jac_mat = self.get_jaccard_matrix(weighted = weighted)
        jac_score = np.sum(jac_mat * match_mat)
        #return jac_score / np.mean(jac_mat.shape)
        return jac_score / np.min(jac_mat.shape)

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

    def print_sentence_matches(self, thresh_jaccard = None):
        jac_mat = self.get_jaccard_matrix()
        matches = self.get_match_matrix(thresh_jaccard = thresh_jaccard)
        if matches is not None:
            source_sents = self.last_source.get_sentences()
            target_sents = self.last_target.get_sentences()
        for i in range(matches.shape[0]):
            if np.sum(matches[i, :] > 0):
                print("S", i, ":", source_sents[i], "\n")
                for j in np.nonzero(matches[i, :])[0]:
                    print("\tT", j, np.round(jac_mat[i, j], 2), ":", target_sents[j], "\n")


class ArticleComparisons(DocumentComparisons):
    def __init__(self, thresh_jaccard = .25, thresh_same_sent = .9, thresh_same_doc = .25):
        DocumentComparisons.__init__(self, thresh_jaccard, thresh_same_sent)
        self.thresh_same_doc = thresh_same_doc
        self.weighted = False
        self.docs = {}
        self.score_mat = None 
        self.clusters = None

    def update_docs(self, docs):
        """ If docs is not None, updates the documents stored in class instance, 
        and returns whether or not the class instance was updated
        """
        if docs is not None and docs is not self.docs:
            self.docs = docs 
            return True 
        return False

    def ceilzero(self, x):
        return max(x, 0)

    def flatten(self, vec):
        return [val for sublist in vec for val in sublist]

    def cluster_articles(self, docs = None, thresh_same_doc = None):
        if thresh_same_doc is not None:
            self.thresh_same_doc = thresh_same_doc
        score_mat = self.jac_score_mat(docs)
        if score_mat is None: return 
        dist_mat = np.vectorize(self.ceilzero)(1 - score_mat)
        hclust = hierarchy.linkage(squareform(dist_mat), method = "complete")
        self.clusters = self.flatten(hierarchy.cut_tree(hclust, height = 1 - self.thresh_same_doc))
        return hclust 

    def get_article_clusters(self, docs = None, thresh_same_doc = None):
        self.cluster_articles(docs, thresh_same_doc)
        return self.clusters

    def prop_unique_clusters(self):
        if self.clusters is not None:
            return len(np.unique(self.clusters))/len(self.clusters)

    def jac_score_mat(self, docs = None, weighted = False, progress = True):
        ''' Returns stored matrix of pairwise document match stores, 
        or computes and returns new matrix if docs is not None, or weighted is
        different from self.weighted 
        '''
        start = time.time()
        if all([self.score_mat is not None, 
               not self.update_docs(docs), 
               weighted == self.weighted]):
            return self.score_mat
        if docs is None:
            if self.docs is None: return 
            docs = self.docs 

        score_mat = np.zeros((len(docs), len(docs)))

        for i, doc1id in enumerate(docs):
            if progress and round(i % (len(docs) / 10)) == 0:
                print(i, "/", len(docs), "done,", 
                     round(time.time() - start, 2), "seconds elapsed")
            for j, doc2id in enumerate(docs):
                if (i > j):
                    score_mat[i, j] = score_mat[j, i]
                else:
                    score_mat[i, j] = self.jaccard_score(docs[doc1id], docs[doc2id])
        self.score_mat = score_mat 
        return score_mat

    def display_mat(self, mat, normalize = False, xlabs = None, ylabs = None):
        """ Uses matplotlib to display mat; 
        if normalize is True, then coloring is normalized by value range
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



