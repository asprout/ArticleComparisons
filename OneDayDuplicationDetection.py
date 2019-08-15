import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
import time

import utils
import documents
import comparisons
import comparisonsmachine

if __name__=='__main__':
    start = time.time()
    data_folder = "data"
    para_sep = "###"
    parser = "spacy"
    thresh_jaccard = .5
    thresh_same_sent = .9
    thresh_same_doc = .25
    date = "20180601"
    article_df = pd.read_pickle(os.path.join(data_folder, "article_df_" + date))
    
    events = [event for event in np.unique(article_df["event"]) if not np.isnan(event)]
    n = [len(article_df.loc[article_df["event"] == event]) for event in events]
    print("Event sizes: ", n)

    try:
        results_df = pd.read_csv("results_" + date + "_clusters_temp.csv")
    except:
        results_df = pd.DataFrame(list(zip(events, n)), columns = ["event", "n"])
        results_df["unique25"] = np.nan
        results_df["unique75"] = np.nan
        results_df["n_good"] = np.nan
        results_df["unique25_good"] = np.nan
        results_df["unique75_good"] = np.nan

    print("Setup time: %d s" % np.round(time.time() - start))

    comparer = comparisonsmachine.MultiComparisons()
    comparer.setThresholds(thresh_jaccard, thresh_same_sent, thresh_same_doc)
    dd = comparisons.DuplicationDetection(thresh_jaccard, thresh_same_sent, thresh_same_doc)

    i = len(events) - 1
    while i >= 0: # Loops over the events
        start = time.time()
        sample = np.array(article_df.loc[article_df["event"] == events[i], "id"])
        good_inds = [i for i in range(len(sample)) if article_df.loc[sample[i], "paywall"] == 0]

        sim_mat = comparer.run(article_df, sample, para_sep, parser)
        dd.cluster_articles(sim_mat)
        results_df.loc[i, "unique25"] = dd.prop_unique_clusters(thresh_same_doc = 0.25)
        results_df.loc[i, "unique25_good"] = dd.prop_unique_clusters(thresh_same_doc = 0.25, subset = good_inds)
        results_df.loc[i, "unique75"] = dd.prop_unique_clusters(thresh_same_doc = 0.75)
        results_df.loc[i, "unique75_good"] = dd.prop_unique_clusters(thresh_same_doc = 0.75, subset = good_inds)
        
        results_df.loc[i, "n_good"] = len(good_inds)
        try:
            results_df.to_csv("results_" + date + "_clusters_temp.csv", index = False)
        except:
            try:
                results_df.to_csv("results_" + date + "_clusters_temp2.csv", index = False)
            except:
                print("Can't write to csv")
        print(results_df.loc[i])
        print("Completed comparisons for Event %d of size %d, %.2fm elapsed" % (i, n[i], utils.minelapsed(start)))
        i = i - 1

    results_df.to_csv("results_" + date + "_clusters_temp.csv", index = False)
