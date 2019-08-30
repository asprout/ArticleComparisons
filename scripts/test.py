import numpy as np
import pandas as pd 
import time
import os
# Article comparisons modules 
import utils # helper functions 
import documents # document loading classes
import comparisons # serial processing classes
import comparisonsmachine as machine

if __name__=='__main__': # Test multi and serial processing speeds
	article_df = pd.read_pickle(os.path.join("data", "article_df_20180715"))
	n = 1000
	sample = [i for i in range(n)]
	print("Running comparisons with %d documents (%d comparisons)" % (n, (n * (n - 1))/2))

	comparer = machine.MultiComparisons()
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