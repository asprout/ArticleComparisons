from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import numpy as np 
import time
# Article comparisons modules
import documents
import utils 

class DocumentComparisons:
    # A class to make pairwise document similarity comparisons 
    def __init__(self, thresh_jaccard = .5, thresh_same_sent = .9):
        self.setThresholds(thresh_jaccard, thresh_same_sent)
        # Store information about the last comparison done
        self.last_source = None
        self.last_target = None 
        self.jaccard_matrix = None # Matrix of pairwise sentence Jaccard indices
        self.jaccard_matches = None # Matrix of pairwise sentence matches (see match_matrix functions)

    def setThresholds(self, thresh_jaccard = None, thresh_same_sent = None):
        if thresh_jaccard is not None: # min Jaccard index to be considered a sentence match
            self.thresh_jaccard = max(thresh_jaccard, 0.001)
        if thresh_same_sent is not None: # Jaccard index to be considered a definite match
            self.thresh_same_sent = thresh_same_sent

    def jaccard_index(self, bow_a, bow_b, counts = False, visualize = False):
        ''' Takes two BOW dictionaries and returns the jaccard index, defined as
            Jaccard(A, B) = |A and B|/|A or B|
            if counts is true, then uses word count, not just unique words
        '''
        set_a = set(bow_a)
        set_b = set(bow_b)
        intsec_words = set_a.intersection(set_b)
        if counts:
            intsec_vals = [min(bow_a[word], bow_b[word]) for word in intsec_words]
            intsec = sum(intsec_vals)
            union = sum(bow_a.values()) + sum(bow_b.values()) - intsec
        else:
            intsec = len(intsec_words)
            union = len(set_a) + len(set_b) - intsec
        index = float(intsec / max(1.0, union))
        if visualize:
            print("Jaccard Index:", index, "with counts", counts)
            print("I:", intsec_words)
            print("A-B:", set_a - set_b, "\nB-A:", set_b - set_a)
        return index

    def compute_jaccard_matrix(self, source, target):
        ''' With an m-sentence source document and n-sentence target, 
        returns a m x n symmetric matrix with pairwise jaccard indices
        '''
        if source is None:
            source = self.last_source
        if target is None:
            target = self.last_target 

        source_bow = source.get_bow_sentences()
        source_n = len(source_bow)
        source_lens = source.get_bow_sentence_lens()

        target_bow = target.get_bow_sentences()
        target_n = len(target_bow)
        target_lens = target.get_bow_sentence_lens()

        if source_n < 1 or target_n < 1: 
            return None # No valid sentences

        jac_mat = np.zeros((source_n, target_n))
        # only consider sentences that are within a certain length of each other
        for i in range(source_n):
            candidates = np.where((target_lens >= source_lens[i] * self.thresh_jaccard) & 
                                  (target_lens <= source_lens[i] / self.thresh_jaccard))[0]
            for j in candidates:
                jac_mat[i, j] = self.jaccard_index(source_bow[i], target_bow[j])
        self.last_source = source
        self.last_target = target 
        self.jaccard_matrix = jac_mat
        return self.jaccard_matrix

    def get_jaccard_matrix(self, source = None, target = None):
        """ Returns the last Jaccard matrix or computes a new one if source
        or target is provided.
        """
        if self.jaccard_matrix is None or utils.notNone([source, target]):
            return self.compute_jaccard_matrix(source, target)
        return self.jaccard_matrix

    def compute_match_matrix(self, jac_mat = None):
        ''' Symmetric matrix matching pairwise sentences between two documents:
        1. Creates a match matrix: 1 if pairwise Jaccards >= thres_jaccard, else 0
        2. Weighs match matrix. For each sentence in source:
            2a. If the highest match > thresh_same_sent, sets match to 1 and all 
            other corresponding matches to source/target sentences to 0
            2b. Otherwise, normalizes the row in the match matrix to sum to 1
            2c. Repeat for target
        '''
        if jac_mat is None:
            jac_mat = self.get_jaccard_matrix(self.last_source, self.last_target)
        matches = 1.0 * (jac_mat >= self.thresh_jaccard)
        matches = self.weigh_matches(matches, jac_mat) # Weigh rows (source sents)
        matches = (self.weigh_matches(matches.T, jac_mat.T)).T # Weigh columns
        self.jaccard_matches = matches
        return self.jaccard_matches

    def get_match_matrix(self, jac_mat = None):
        ''' Returns the last computed match matrix if it exists and all parameters are None,
        else computes and returns a match matrix. 
        '''
        if self.jaccard_matches is None or jac_mat is not None:
            return self.compute_match_matrix(jac_mat)
        return self.jaccard_matches

    def weigh_matches(self, matches, jaccards):
        ''' Goes through the rows of the matches matrix. For each row:
        if the largest value is above the threshold for same sentences,
            sets corresponding match value to one and
            all other indices in the same row and column to 0
        otherwise, divides by the sum such that the normalized sum is 1
        '''
        rowsums = np.sum(matches, axis = 1)
        for i in np.where(rowsums > 0)[0]:
            argmax = np.argmax(jaccards[i, :]) 
            if jaccards[i, argmax] >= self.thresh_same_sent:
                matches[i, :] = [0] * matches.shape[1]
                matches[:, argmax] = [0] * matches.shape[0]
                matches[i, argmax] = 1
            elif rowsums[i] > 1:
                matches[i, :] = matches[i, :] / rowsums[i]
        return matches

    def jaccard_score(self, source = None, target = None):
        ''' Prints the similarity score between the source and target docs 
        '''
        jac_mat = self.get_jaccard_matrix(source, target)
        if jac_mat is None:
            return 0 # No valid sentences in either source or target
        match_mat = self.get_match_matrix(jac_mat)
        jac_score = np.sum(jac_mat * match_mat)
        #return jac_score / np.mean(jac_mat.shape) 
        return jac_score / np.min(jac_mat.shape) # count snippets as duplicates

    def print_sentence_matches(self, thresh_jaccard = None):
        jac_mat = self.get_jaccard_matrix()
        if jac_mat is None:
            return 
        if thresh_jaccard is not None:
            self.setThresholds(thresh_jaccard = thresh_jaccard)
            matches = self.get_match_matrix(jac_mat)
        else:
            matches = self.get_match_matrix()
        source_sents = self.last_source.get_sentences()
        target_sents = self.last_target.get_sentences()
        for i in range(matches.shape[0]):
            if np.sum(matches[i, :] > 0):
                print("S", i, ":", source_sents[i], "\n")
                for j in np.nonzero(matches[i, :])[0]:
                    print("\tT", j, np.round(jac_mat[i, j], 2), ":", target_sents[j], "\n")

class DuplicationDetection(DocumentComparisons):
    ''' A specific DocumentComparisons instance for dealing with articles
    '''
    def __init__(self, thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .25):
        DocumentComparisons.__init__(self, thresh_jaccard, thresh_same_sent)
        self.docs = {}
        self.thresh_same_doc = thresh_same_doc # similarity score to be considered same article
        self.sim_mat = None # pairwise document similarity score matrix 
        self.clusters = None # list of clusters, indexed by document
        self.hclust = None # hierarchical agglo. clustering object

    def readArticles(self, path):
        """ Reads df of articles from the given path, and adds a column to store the Doc 
        """
        article_df = pd.read_csv(path)
        article_df["doc"] = None 
        return article_df 

    def dict_by_ids(self, df, ids, para_sep = "###", parser = None):
        """ Given a dataframe of articles and a list of article ids, 
        returns a dictionary with ids as keys and Documents as items, 
        computing and storing the Documents back in the df as needed
        """
        doc_dict = {}
        for doc_id in ids:
            row = df["id"] == doc_id 
            if parser is not None:
                df.loc[row, "doc"] = documents.Document(df.loc[row, "text"].iloc[0], para_sep, parser)
            elif df.loc[row, "doc"].iloc[0] is None:
                df.loc[row, "doc"] = documents.Document(df.loc[row, "text"].iloc[0], para_sep)
            doc_dict[doc_id] = df.loc[row, "doc"].iloc[0]
        return doc_dict

    def similarity_mat(self, docs = None, progress = True):
        ''' Returns stored matrix of pairwise document match stores, 
        or computes and returns new matrix if docs is not None
        '''
        start = time.time()
        if self.sim_mat is not None and (docs is None or docs is self.docs):
            return self.sim_mat
        if docs is None:
            docs = self.docs

        docids = np.sort([i for i in docs.keys()]) # Order by docids 
        ndocs = len(docids)
        sim_mat = np.zeros((ndocs, ndocs))

        for i, doc1id in enumerate(docids):
            for j in range(i + 1, ndocs):
                doc1 = docs[doc1id]
                doc2 = docs[docids[j]]
                if utils.cosinesim(doc1.vec, doc2.vec) >= 0.8:
                    sim_mat[i, j] = self.jaccard_score(doc1, doc2)
            if progress and round(i % 100) == 0:
                print("%d of %d rows completed, %.2fm elapsed" % (i, len(docs), utils.minelapsed(start)))
        sim_mat = sim_mat + sim_mat.transpose()
        np.fill_diagonal(sim_mat, 1.0)
        self.sim_mat = sim_mat 
        self.docs = docs 
        # Reset clusters due to updated score matrix
        self.clusters = None 
        self.hclust = None 
        return sim_mat

    def cluster_articles(self, sim_mat = None, plot = False):
        ''' Runs hierarchical agglomerative clustering on pairwise document
        distances, computed as 1 - document similarity scores (defined
        as the matrix of match-weighted pairwise sentence Jaccard indices).
        linkage = single: A document is assigned to a cluster if any document 
            in the cluster meets the distance threshold.
        linkage = complete: '...' if all documents in cluster meets threshold
        '''
        if sim_mat is None:
            sim_mat = self.sim_mat 
        dist_mat = np.vectorize(utils.ceilzero)(1 - sim_mat)
        self.hclust = hierarchy.linkage(squareform(dist_mat), method = 'single')
        if plot:
            hierarchy.dendrogram(self.hclust)
            plt.ylabel("Distance")
        return self.hclust 

    def get_article_clustering(self, sim_mat = None):
        ''' Returns clustering object, indexed by document
        '''
        if self.hclust is None or sim_mat is not None:
            self.hclust = self.cluster_articles(sim_mat)
        return self.hclust

    def get_cluster_assignments(self, sim_mat = None, thresh_same_doc = None):
        ''' Returns list of document cluster assignments 
        '''
        hclust = self.get_article_clustering(sim_mat)
        if thresh_same_doc is not None:
            self.thresh_same_doc = thresh_same_doc
        self.clusters = utils.flatten(hierarchy.cut_tree(hclust, height = 1 - self.thresh_same_doc))
        return self.clusters

    def prop_unique_clusters(self, thresh_same_doc = None, subset = None):
        clusters = self.get_cluster_assignments(thresh_same_doc = thresh_same_doc)
        return utils.prop_unique(clusters, subset)





