import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import documents as docs
import textcomparisons as tc
import random
import time

def readArticles(path):
    """ Reads df of articles from the given path, and adds a column
    to store the Document-processed article """
    article_df = pd.read_csv(path)
    article_df["doc"] = None
    return article_df

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

data_folder = "data"
article_files = ["articles2019-05-31_0-7000.csv",
                 "articles2019-05-31_7000-14000.csv",
                 "articles2019-05-31_14000-16654.csv"]
article_df = [readArticles(os.path.join(data_folder, file)) for file in article_files]

article_df = pd.concat(article_df)
article_df = article_df.reset_index()

events = [event for event in np.unique(article_df["event"]) if not np.isnan(event)]
n = [len(article_df.loc[article_df["event"] == event]) for event in events]

results_df = pd.DataFrame(list(zip(events, n)), columns = ["event", "n"])
results_df["clusters"] = np.nan
results_df["unique25"] = np.nan
results_df["unique75"] = np.nan

ac = tc.ArticleComparisons(thresh_jaccard = .5, thresh_same_sent = .9, thresh_same_doc = .25)

i = len(events) - 1
while results_df.loc[i, "n"] < 1000 and i >= 0:
    sample = article_df.loc[article_df["event"] == events[i], "id"]
    article_dict = dict_by_ids(article_df, sample)
    results_df.loc[i, "clusters"] = {"clusters": ac.cluster_articles(article_dict, thresh_same_doc = 0.25)}
    results_df.loc[i, "unique25"] = ac.prop_unique_clusters()
    clusters = ac.cluster_articles(article_dict, thresh_same_doc = 0.75)
    results_df.loc[i, "unique75"] = ac.prop_unique_clusters()
    try:
        results_df.to_csv("results_20190531_clusters1.csv", index = False)
    except:
        print("Can't write to csv")
    i = i - 1

results_df.to_csv("results_20190531_clusters2.csv", index = False)
