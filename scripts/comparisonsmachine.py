import multiprocessing as mp 
import numpy as np
import pandas as pd 
import time
import os
# Article comparisons modules 
import utils # helper functions 
import documents # document loading classes
import comparisons # serial processing classes

class DocumentComparer:
	"""
	Simple class of methods to make comparisons between two documents 
	Optimized for use with multiprocessing implemented in ParallelComparisons
	For further details or functions with UI, please see comparisons module   
	"""
	def __init__(self, thresh_jaccard = .5, thresh_same_sent = .9):
		self.setThresholds(thresh_jaccard, thresh_same_sent)

	def setThresholds(self, thresh_jaccard = None, thresh_same_sent = None):
		if thresh_jaccard is not None: # min Jaccard index to be considered a sentence match
			self.thresh_jaccard = max(thresh_jaccard, 0.0001)
		if thresh_same_sent is not None: # Jaccard index to be considered a definite match
			self.thresh_same_sent = thresh_same_sent

	def jaccard_index(self, bow_a, bow_b, union_subset = False):
		# Jaccard(A, B) = |A and B|/|A or B|
		if bow_a is None or bow_b is None:
			return -1 if union_subset else 0
		set_a = set([a for a in bow_a if a is not None])
		set_b = set([b for b in bow_b if b is not None])
		if len(set_a) < 1 or len(set_b) < 1:
			return -1 if union_subset else 0		
		intsec_words = set_a.intersection(set_b)
		intsec = len(intsec_words)
		if union_subset:
			union = min(len(set_a), len(set_b))
		else:
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
			return None # No valid sentences

		jac_mat = np.zeros((source_n, target_n))
		entities_matrix = np.zeros((source_n, target_n))
        # only consider sentences that are within a certain length of each other
		for i in range(source_n):
			candidates = np.where((target_lens >= source_lens[i] * self.thresh_jaccard) & 
                                  (target_lens <= source_lens[i] / self.thresh_jaccard))[0]
			for j in candidates:
				jac_mat[i, j] = self.jaccard_index(source_bow[i], target_bow[j])
				if jac_mat[i, j] >= self.thresh_jaccard:
					entities_matrix[i, j] = self.jaccard_index(source.sent_entities[i], target.sent_entities[j], union_subset = True)
		return [jac_mat, entities_matrix]


	def compute_match_matrix(self, jac_mat):
		matches = 1.0 * (jac_mat > self.thresh_jaccard)
		matches = self.weigh_matches(matches, jac_mat) # Weigh rows (source sents)
		matches = (self.weigh_matches(matches.T, jac_mat.T)).T # Weigh columns
		return matches

	def weigh_matches(self, matches, jaccards):
		for i in np.where(np.max(jaccards, axis = 1) > self.thresh_same_sent)[0]:
			argmax = np.argmax(jaccards[i, :])
			matches[i, :] = [0] * matches.shape[1]
			matches[:, argmax] = [0] * matches.shape[0]
			matches[i, argmax] = 1 
		rowsums = np.sum(matches, axis = 1)
		for i in np.where(rowsums > 0)[0]:
			matches[i, :] = matches[i, :] / rowsums[i]
		return matches

	def jaccard_score(self, source, target, weighted = True):
		mats = self.compute_jaccard_matrix(source, target)
		if mats is None:
			return 0 # No valid sentences in either source or target
		[jac_mat, entities_mat] = mats
		match_mat = self.compute_match_matrix(jac_mat)

		if not weighted:
			jac_score = np.sum(jac_mat * match_mat)
			return jac_score / np.min(jac_mat.shape) # count snippets as duplicates
		# Weigh sentences < length 5 lower 
		s_weights = [min(i/5, 1) for i in source.bow_sent_lens]
		t_weights = [min(i/5, 1) for i in target.bow_sent_lens]
		# Weigh sentences without any entities lower (usually unimportant)
		s_weights = [s_weights[i]/2 if len(source.sent_entities[i]) == 0 else s_weights[i] for i in range(len(s_weights))]
		t_weights = [t_weights[i]/2 if len(target.sent_entities[i]) == 0 else t_weights[i] for i in range(len(t_weights))]
		weight_mat = np.reshape([[np.min([s, t]) for t in t_weights] for s in s_weights], [-1, len(t_weights)])
		# match scores * matched sentences * weights 
		score = np.sum(jac_mat * match_mat * weight_mat * abs(entities_mat))
		return 0 if score == 0 else score/np.sum(np.max(weight_mat, axis = 1 - np.argmin(weight_mat.shape)))

class MultiComparisons():

	def __init__(self):
		self.thresh_jaccard = .5
		self.thresh_same_sent = .9
		self.thresh_same_doc = .5
		self.para_sep = "###"
		self.parser = "spacy"
		self.comparer = DocumentComparer(self.thresh_jaccard, self.thresh_same_sent)
		self.pool = None
		self.start = time.time()

	def setThresholds(self, thresh_jaccard = None, thresh_same_sent = None, thresh_same_doc = None):
		if thresh_jaccard is not None:
			self.thresh_jaccard = thresh_jaccard 
		if thresh_same_sent is not None:
			self.thresh_same_sent = thresh_same_sent
		if thresh_same_doc is not None:
			self.thresh_same_doc = thresh_same_doc 
		self.comparer = DocumentComparer(self.thresh_jaccard, self.thresh_same_sent)

	'''
	def badTextChecker(self, doc):
		text = str(doc).lower()
		nchar = len(text)
		penalty = 0.5 * (nchar< 500)
		penalty += 1 * (len([1 for sent in doc.sentences if utils.keywordsin(sent)]) > len(doc.sentences)/2)
		entities = [ent for ent in utils.flatten(doc.sent_entities) if ent is not None]
		penalty += 2 * (len(entities) < 1)
		return penalty

	def filter_articles(self, df, ids = None):
		self.start = time.time()
		if ids is None:
			df["paywall"] = np.nan # reset all 
			ids = [i for i in range(len(df))]
		ndocs = len(ids)
		# MULTIPROCESSING:create and distribute asynchronous tasks to analyze documents
		tasks = [df.loc[df["id"] == i, "doc"].iloc[0] for i in ids]
		if self.pool is None:
			pool = mp.Pool(processes = min(mp.cpu_count() - 2, round(ndocs/50 + 1)))
		else:
			pool = self.pool
			self.pool = None 
		results = pool.imap(self.badTextChecker, tasks)
		for i, res in enumerate(results):
			df.loc[df["id"] == ids[i], "paywall"] = res 
		print("Checked article text validity via multiprocessing, %.2fm elapsed" % (utils.minelapsed(self.start)))
		self.pool = pool
		return docs 
	'''

	def worker(self, docs):
		doc1 = docs[0]
		doc2 = docs[1]
		# Only compare pairs with high vector cosine similarity 
		if utils.cosinesim(doc1.vec, doc2.vec) >= 0.8:
			if 0 < abs(self.comparer.jaccard_index(utils.flatten(doc1.sent_entities), 
						utils.flatten(doc2.sent_entities), union_subset = True)):
				return self.comparer.jaccard_score(doc1, doc2)
		return 0

	def similarity_mat(self, docs, progress = True, ordered = False):
		self.start = time.time()
		docids = [i for i in docs.keys()]
		if ordered:
			docids = np.sort(docids)
		ndocs = len(docids)
		score_mat = np.zeros((ndocs, ndocs)) # Initialize score matrix 
		# MULTIPROCESSING: create and distribute asynchronous document pair comparison tasks
		tasks = [[docs[docids[i]], docs[docids[j]]] for i in range(ndocs) for j in range(i + 1, ndocs)]
		if self.pool is None:
			pool = mp.Pool(processes = min(mp.cpu_count() - 2, round(ndocs/50 + 1)))
		else: # Use the last pool created, and delete it from shared variables to avoid copying
			pool = self.pool
			self.pool = None 
		results = pool.imap(self.worker, tasks)
		mat_inds = np.triu_indices(ndocs, 1)
		for i, score in enumerate(results):
			score_mat[mat_inds[0][i], mat_inds[1][i]] = score 
			if progress and i % 10000 == 0:
				print("%d of %d comparisons made, %.2fm elapsed" % (i, len(tasks), utils.minelapsed(self.start)))
		# Fill in rest of score matrix with upper triangle of results 
		score_mat = score_mat + score_mat.transpose()
		np.fill_diagonal(score_mat, 1.0)
		print("Finished document comparisons via multiprocessing, %.2fm elapsed\n" % (utils.minelapsed(self.start)))
		self.pool = pool
		return score_mat 

	def reader(self, dfitems):
		if dfitems["doc"] is None or self.parser is not None:
			dfitems["doc"] = documents.Document(dfitems["text"], self.para_sep, self.parser)
		return dfitems["doc"]

	def dict_by_ids(self, df, ids, para_sep = None, parser = None):
		self.start = time.time()
		ndocs = len(ids)
		# Update document parsing parameters as necessary 
		if para_sep is not None:
			self.para_sep = para_sep
		if parser is not None:
			self.parser = parser 
		# MULTIPROCESSING: create and distribute asynchronous tasks to read docs
		tasks = [df.loc[df["id"] == i, ["id", "text", "doc"]].iloc[0] for i in ids]
		if self.pool is None:
			pool = mp.Pool(processes = min(mp.cpu_count() - 2, round(ndocs/50 + 1)))
		else:
			pool = self.pool
			self.pool = None
		results = pool.imap(self.reader, tasks)
		docs = {}
		for i, doc in enumerate(results):
			docs[ids[i]] = doc
			df.loc[df["id"] == ids[i], "doc"] = doc 
		print("Loaded documents via multiprocessing, %.2fm elapsed" % (utils.minelapsed(self.start)))
		self.pool = pool 
		return docs

	def run(self, docs, progress = True):
		return self.similarity_mat(docs = docs, progress = progress)
		if self.pool is not None:
			self.pool.close()
			self.pool.join()

if __name__=='__main__': # Test multi and serial processing speeds
	article_df = pd.read_pickle(os.path.join("data", "article_df_20180715"))
	n = 1000
	sample = [i for i in range(n)]
	print("Running comparisons with %d documents (%d comparisons)" % (n, (n * (n - 1))/2))

	comparer = MultiComparisons()
	comparer.setThresholds(thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .5)
	dd = comparisons.DuplicationDetection(thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .5)

	article_dict = comparer.dict_by_ids(article_df, sample, "###", "spacy")
	mat = comparer.run(article_dict)
	print("Jaccard sum:", np.sum(mat))
	print(np.round(mat, 2))
	dd.cluster_articles(mat)
	print("% Unique for 25, 75, 25_good, 75_good:")
	print(dd.prop_unique_clusters(thresh_same_doc = 0.25),
		dd.prop_unique_clusters(thresh_same_doc = 0.75),
		dd.prop_unique_clusters(thresh_same_doc = 0.25, subset = good_inds),
		dd.prop_unique_clusters(thresh_same_doc = 0.75, subset = good_inds))

	print("\n SERIAL PROCESSING \n")

	start = time.time()
	article_dict = dd.dict_by_ids(article_df, sample, "###", "spacy")
	print("Loaded documents via serial processing, %.2fm elapsed" % (utils.minelapsed(start)))
	serialmat = dd.similarity_mat(article_dict)
	print("Jaccard sum:", np.sum(serialmat))
	print(np.round(serialmat, 2))
	print("Finished document comparisons via serial processing, %.2fm elapsed\n" % (utils.minelapsed(start)))
	print("% Unique for 25, 75, 25_good, 75_good:")
	dd.cluster_articles(serialmat)
	print(dd.prop_unique_clusters(thresh_same_doc = 0.25),
		dd.prop_unique_clusters(thresh_same_doc = 0.75),
		dd.prop_unique_clusters(thresh_same_doc = 0.25, subset = good_inds),
		dd.prop_unique_clusters(thresh_same_doc = 0.75, subset = good_inds))

	### with 500 docs (125,000 comparisons)
	# mp: 40s to load processes and documents, 2.4m total 
	# serial: 40s to load documents, 5.5m total
	### with 1000 docs (500,000 comparisons)
	# mp: 1m to load, 9m total 
	# serial: 2m to load, 20m total 

	# Testing with article duplication python scripts:
	# 4551 articles: 172m (~3 hours)
	# 2033 articles: 35m
	# 1215 articles: 12m 