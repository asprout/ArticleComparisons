import multiprocessing as mp # Get number of cores 
import numpy as np
import pandas as pd 
import time
import os
# Article comparisons modules 
import documents
import textcomparisons as tc 
import utils 

class DocumentComparison():
	# Simple class of methods to make comparisons between two documents 
	# Optimized for use with multiprocessing implemented in ParallelComparisons 
	def __init__(self, thresh_jaccard = .5, thresh_same_sent = .9):
		self.thresh_jaccard = thresh_jaccard # min Jaccard index to be considered a sentence match
		self.thresh_same_sent = thresh_same_sent # Jaccard index to be considered a definite match

	def jaccard_index(self, bow_a, bow_b):
		set_a = set(bow_a)
		set_b = set(bow_b)
		intsec_words = set_a.intersection(set_b)
		intsec = len(intsec_words)
		union = len(set_a) + len(set_b) - intsec
		return float(intsec / max(1.0, union))

	def compute_jaccard_matrix(self, source, target):
		source_bow = source.get_bow_sentences()
		source_n = len(source_bow)
		source_lens = source.get_bow_sentence_lens()

		target_bow = target.get_bow_sentences()
		target_n = len(target_bow)
		target_lens = target.get_bow_sentence_lens()

		if source_n < 1 or target_n < 1: 
			return None 

		jac_mat = np.zeros((source_n, target_n))
        # only consider sentences that are within a certain length of each other
		for i in range(source_n):
			candidates = np.where((target_lens >= source_lens[i] * self.thresh_jaccard) * 
                                  (target_lens <= source_lens[i] / self.thresh_jaccard))[0]
			for j in candidates:
				jac_mat[i, j] = self.jaccard_index(source_bow[i], target_bow[j])

		return jac_mat

	def compute_match_matrix(self, source, target):
		jac_mat = self.compute_jaccard_matrix(source, target)
		matches = 1.0 * (jac_mat >= self.thresh_jaccard)
		matches = self.weigh_matches(matches, jac_mat) # Weigh rows (source sents)
		matches = (self.weigh_matches(matches.T, jac_mat.T)).T # Weigh columns
		self.jaccard_matches = matches
		return self.jaccard_matches

	def weigh_matches(self, matches, jaccards):
		rowsums = np.sum(matches, axis = 1)
		rows = np.where(rowsums > 0)[0]
		for i in rows:
			argmax = np.argmax(jaccards[i, :]) 
			if jaccards[i, argmax] >= self.thresh_same_sent:
				matches[i, :] = [0] * matches.shape[1]
				matches[:, argmax] = [0] * matches.shape[0]
				matches[i, argmax] = 1
			rowsum = np.sum(matches[i, :])
			if rowsum > 1:
				matches[i, :] = matches[i, :] / rowsum 
		return matches

	def jaccard_score(self, source = None, target = None):
		jac_mat = self.compute_jaccard_matrix(source, target)
		if jac_mat is None:
			return 0 # No valid sentences in either source or target
		match_mat = self.compute_match_matrix(source, target)
		jac_score = np.sum(jac_mat * match_mat)
		return jac_score / np.min(jac_mat.shape) # count snippets as duplicates

class ParallelComparisons():
	thresh_jaccard = .5
	thresh_same_sent = .9 
	comparer = DocumentComparison(thresh_jaccard, thresh_same_sent)

	def __init__(self, thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .25):
		if (thresh_jaccard != self.thresh_jaccard or thresh_same_sent != self.thresh_same_sent):
			self.comparer = DocumentComparison(thresh_jaccard, thresh_same_sent)
		self.start = 0
		self.print = True

	def worker(self, docs):
		if self.print:
			print(os.getpid(), self.comparer, np.round(time.time() - self.start, 2), "s elapsed")
			#print(os.getpid(), docinds, self.comparer, np.round(time.time() - self.start, 2), "s elapsed")
		self.print = False
		doc1 = docs[0]
		doc2 = docs[1]
		#doc1 = self.docs[self.docids[docinds[0]]]
		#doc2 = self.docs[self.docids[docinds[1]]]
		if utils.cosinesim(doc1.vec, doc2.vec) >= 0.9:
			return self.comparer.jaccard_score(doc1, doc2)
		return 0

	def run(self, docs = None):
		"""
		if docs is not None:
			self.docs = docs 
		self.docids = np.sort([i for i in self.docs.keys()])
		ndocs = len(self.docids)
		score_mat = np.zeros((len(self.docs), len(self.docs)))
		"""
		docids = np.sort([i for i in docs.keys()])
		ndocs = len(docids)
		score_mat = np.zeros((ndocs, ndocs))

		### MULTIPROCESSING 
		tasks = utils.flatten([[[docs[docids[i]], docs[docids[j]]] for j in range(i + 1, ndocs)] for i in range(ndocs)])
		#tasks = utils.flatten([[[i, j] for j in range(i + 1, ndocs)] for i in range(ndocs)])
		print("ntasks: ", len(tasks))
		self.start = time.time()
		pool = mp.Pool(processes = min(mp.cpu_count() - 1, round(ndocs/40)))
		results = pool.map_async(self.worker, tasks)
		jac_scores = results.get()

		score_mat[np.triu_indices(ndocs, 1)] = jac_scores
		score_mat = score_mat + score_mat.transpose()
		np.fill_diagonal(score_mat, 1.0)
		return score_mat 

if __name__=='__main__':
	ac = tc.ArticleComparisons(thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .25)
	article_df = pd.read_pickle(os.path.join("data", "article_df_20180715"))
	sample = [i for i in range(200)]
	article_dict = ac.dict_by_ids(article_df, sample, "###", "nltk")
	print("Created article dict", len(article_dict))

	start = time.time()
	comparer = ParallelComparisons()
	mat = comparer.run(article_dict)
	print("Jaccard sum:", np.sum(mat))
	print(np.round(mat, 2))
	print(time.time() - start)

	start = time.time()
	serialmat = ac.jac_score_mat(article_dict)
	print("Jaccard sum:", np.sum(serialmat))
	print(np.round(serialmat, 2))
	print(time.time() - start)
	# with 200 articles, pool.map takes 74s, and serial processing takes 57