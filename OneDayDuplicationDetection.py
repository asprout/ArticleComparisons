import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import documents as docs
import textcomparisons as tc
import random
import time

start = time.time()

data_folder = "data"
article_files = ["articles2019-06-01_" + str(i) + "-" + str(i + 5000) + ".csv" for i in range(0, 100000, 5000)]
article_files = article_files + ["articles2019-06-01_100000-100755.csv"]

def readArticles(path):
    """ Reads df of articles from the given path, and adds a column
    to store the Document-processed article """
    article_df = pd.read_csv(path)
    article_df["doc"] = None
    return article_df

keywords = ["subscription", "subscribe", "full access", "digital access", "sign up", "unlimited access", "unlimited digital access", "log in", "sign up"]
def keywordsin(str):
    for word in keywords:
        if word in str:
            return True
    return False

def isPaywall(i, to_doc = True):
    text = article_df.loc[i, "text"]
    text = text.lower()
    if len(text) < 500:
        article_df.loc[i, "paywall"] += 0.5
    if len(text) < 1000 and keywordsin(text):
        article_df.loc[i, "paywall"] += 1
    if to_doc and article_df.loc[i, "doc"] is None:
        article_df.loc[i, "doc"] = docs.Document(text, clean = False)
    return article_df.loc[i, "paywall"] > 0

def dict_by_ids(df, ids):
    """ Given a dataframe of articles and a list of article ids, 
    returns a dictionary with ids as keys and Documents as items, 
    computing and storing the Documents back in the df as needed
    """
    doc_dict = {}
    for doc_id in ids:
        row = df["id"] == doc_id
        doc = df.loc[row, "doc"].iloc[0]
        if doc is None:
            doc = docs.Document(df.loc[row, "text"].iloc[0], clean = False)
            df.loc[row, "doc"] = doc
        doc_dict[doc_id] = doc
    return doc_dict

'''
article_df = [readArticles(os.path.join(data_folder, file)) for file in article_files]
article_df = pd.concat(article_df)
article_df = article_df.reset_index(drop = True)

article_df["paywall"] = 0
'''
article_df = pd.read_pickle(os.path.join(data_folder, "article_df_20190601"))

events = [event for event in np.unique(article_df["event"]) if not np.isnan(event)]
n = [len(article_df.loc[article_df["event"] == event]) for event in events]
print("Event sizes: ", n)

ac = tc.ArticleComparisons(thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .25)
try:
    results_df = pd.read_csv("results_20190601_clusters_temp.csv")
except:
    results_df = pd.DataFrame(list(zip(events, n)), columns = ["event", "n"])
    results_df["unique25"] = np.nan
    results_df["unique75"] = np.nan
    results_df["n_good"] = np.nan
    results_df["unique25_good"] = np.nan
    results_df["unique75_good"] = np.nan

print("Setup time: %d s" % np.round(time.time() - start))
i = len(events) - 1
while i >= 0: # Loops over the events
    if results_df.loc[i, "n"] > 500 or not np.isnan(results_df.loc[i, "unique25"]):
        print("Event", i, "of size", results_df.loc[i, "n"], "skipped")
        i -= 1
        continue
    start = time.time()
    sample = np.array(article_df.loc[article_df["event"] == events[i], "id"])
    article_dict = dict_by_ids(article_df, sample)
    clustering = ac.cluster_articles(article_dict, plot = False)
    good_inds = [i for i in range(len(sample)) if not isPaywall(sample[i])]
    results_df.loc[i, "n_good"] = len(good_inds)
    results_df.loc[i, "unique25"] = ac.prop_unique_clusters(thresh_same_doc = 0.25)
    results_df.loc[i, "unique25_good"] = ac.prop_unique_clusters(thresh_same_doc = 0.25, inds = good_inds)
    results_df.loc[i, "unique75"] = ac.prop_unique_clusters(thresh_same_doc = 0.75)
    results_df.loc[i, "unique75_good"] = ac.prop_unique_clusters(thresh_same_doc = 0.75, inds = good_inds)
    try:
        results_df.to_csv("results_20190601_clusters_temp.csv", index = False)
    except:
        results_df.to_csv("results_20190601_clusters_temp2.csv", index = False)
        print("Can't write to csv")
    print("Event", i, time.time() - start, "s elapsed")
    i = i - 1

results_df.to_csv("results_20190601_clusters.csv", index = False)
