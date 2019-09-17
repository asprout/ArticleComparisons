import os, sys 
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
    results_folder = os.path.join("..", "results")
    para_sep = "###"
    parser = "spacy"
    thresh_jaccard = .5
    thresh_same_sent = .9
    thresh_same_doc = .75

    date = "20180715"
    if len(sys.argv) > 1:
        date = sys.argv[1]

    article_df_path = os.path.join("..", data_folder, "article_df_" + date)
    try:
        article_df = pd.read_pickle(article_df_path)
    except:
        sys.exit(f"Could not read article dataframe from {article_df_path}. \nPlease check that your provided date ({date}) is formatted as YYYYmmdd.")
    
    events = [event for event in np.unique(article_df["event"]) if not np.isnan(event)]
    n = [len(article_df.loc[article_df["event"] == event]) for event in events]
    print("Event sizes: ", n)

    try:
        results_df = pd.read_csv(os.path.join(results_folder, "results_" + date + "_clusters_temp.csv"))
    except:
        results_df = pd.DataFrame(columns = ["event", "n", "n_valid"] + [f"unique{thresh}" for thresh in range(10, 100, 5)])
        results_df["event"] = events
        results_df["n"] = n

    print("Setup time: %d s" % np.round(time.time() - start))

    comparer = comparisonsmachine.MultiComparisons()
    comparer.setThresholds(thresh_jaccard, thresh_same_sent, thresh_same_doc)
    dd = comparisons.DuplicationDetection(thresh_jaccard, thresh_same_sent, thresh_same_doc)

    i = len(events) - 1
    while i >= 0: # Loops over the events
        if not np.isnan(results_df.loc[i, "unique25"]):
            print("Skipping completed event %d" % (i))
            i = i - 1
            continue 
        print("Starting comparisons for Event %d of size %d" % (i, n[i]))
        start = time.time()
        sample = np.array(article_df.loc[article_df["event"] == events[i], "id"])
        if n[i] > 10000:
            print(f"Sampling from event {i} due to large size")
            sample = random.sample(list(sample), 1000)

        article_dict = comparer.dict_by_ids(article_df, sample, para_sep, parser)

        article_df.loc[sample, "invalid"] = [article_df.loc[article_df["id"] == i, "doc"].iloc[0].invalid for i in sample]
        valid_inds = [i for i in sample if article_df.loc[i, "invalid"] == 0]
        article_dict_valid = {k: article_dict[k] for k in valid_inds}

        sim_mat = comparer.run(article_dict_valid)
        dd.cluster_articles(sim_mat)
        for thresh in range(10, 100, 5):
            results_df.loc[i, f"unique{thresh}"] = dd.prop_unique_clusters(thresh_same_doc = thresh/100)

        results_df.loc[i, "n_valid"] = len(valid_inds)

        try:
            results_df.to_csv(os.path.join(results_folder, "results_" + date + "_clusters_temp.csv"), index = False)
        except:
            try:
                results_df.to_csv(os.path.join(results_folder, "results_" + date + "_clusters_temp2.csv"), index = False)
            except:
                print("Can't write to csv")

        #print(results_df.loc[i])
        print("Completed comparisons for Event %d with %d valid articles, %.2fm elapsed" % (i, len(valid_inds), utils.minelapsed(start)))
        i = i - 1

    # Take random sample of non-event articles 
    n = 10000
    sample = random.sample(list(article_df.loc[np.isnan(article_df["event"]), "id"]), n)
    article_dict = comparer.dict_by_ids(article_df, sample, para_sep, parser)

    article_df.loc[sample, "invalid"] = [article_df.loc[article_df["id"] == i, "doc"].iloc[0].invalid for i in sample]
    valid_inds = [i for i in sample if article_df.loc[i, "invalid"] == 0]
    article_dict_valid = {k: article_dict[k] for k in valid_inds}

    sim_mat = comparer.run(article_dict_valid)
    dd.cluster_articles(sim_mat)
    results_df.loc[len(events), "event"] = np.sum(np.isnan(article_df["event"]))
    results_df.loc[len(events), "n"] = n
    results_df.loc[len(events), "n_valid"] = len(valid_inds)
    for thresh in range(10, 100, 5):
        results_df.loc[len(events), f"unique{thresh}"] = dd.prop_unique_clusters(thresh_same_doc = thresh/100)

    results_df.to_csv(results_folder, index = False)

# Takes 6.5m to parse 10,000 documents
# Of those, say about 5279 are "invalid"
# Takes to finish pairwise comparisons with all of those documents.