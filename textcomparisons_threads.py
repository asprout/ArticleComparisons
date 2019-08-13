import threading
import multiprocessing # Get number of cores 
import numpy as np
import time
import documents
import textcomparisons as tc 

class ParallelComparisons():
	def __init__(self, thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .25):
		# Article Comparisons objects 
		self.thresh_jaccard = thresh_jaccard
		self.thresh_same_sent = thresh_same_sent 
		self.thresh_same_doc = thresh_same_doc
		self.docs = None
		self.docids = None 
		self.score_mat = None
		self.clusters = None
		self.hclust = None 
		# Threading objects 
		self.tasks = [] # List of 2-element arrays of source and target ids 
		self.task_counter = 0
		self.running = False
		self.lock = threading.Lock()

	def progress(self):
		start = time.time()
		while self.running:
			print(f"{self.task_counter}/{len(self.tasks)} done, {round((time.time() - start)/60, 2)}m elapsed")
			time.sleep(5)
		print(f"{len(self.tasks)} completed in {round((time.time() - start)/60, 2)} minutes")

	def worker(self):
		comparer = tc.DocumentComparisons(self.thresh_jaccard, self.thresh_same_sent)
		while True:
			with self.lock:
				if self.task_counter >= len(self.tasks):
					break
				doc_indices = self.tasks[self.task_counter]
				self.task_counter += 1
			doc1 = self.docs[self.docids[doc_indices[0]]]
			doc2 = self.docs[self.docids[doc_indices[1]]]
			if tc.cosinesim(doc1.vec, doc2.vec) >= 0.9:
				self.score_mat[doc_indices[0], doc_indices[1]] = comparer.jaccard_score(doc1, doc2)

	def run(self, docs):
		self.docs = docs 
		self.docids = [i for i in docs.keys()]
		self.tasks = tc.flatten([[[i, j] for j in range(i + 1, len(docs))] for i in range(len(docs))])
		self.task_counter = 0 
		self.score_mat = np.zeros((len(docs), len(docs)))
		self.clusters = None
		self.hclust = None 

		self.running = True 
		progress = threading.Thread(target = self.progress)
		progress.start()
		threads = []
		for core in range(multiprocessing.cpu_count()):
			thread = threading.Thread(target = self.worker)
			thread.start()
			threads.append(thread)
		for thread in threads:
			thread.join()
		self.running = False

		self.score_mat = self.score_mat + self.score_mat.transpose()
		np.fill_diagonal(self.score_mat, 1.0)

		return self.score_mat 
