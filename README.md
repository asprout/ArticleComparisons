# Provenance
## Tracing the origins and production of news. 

### Question 1: How much original news is being produced?
*Goal*: identify the percentage of "unique" articles that are published on any given day, and how this number changes based on the type of publishers and events sampled. 
We believe that the number of original news articles that exists in the information ecosystem is much lower than the number of published articles, particularly for smaller publishers. Based on previous work on this topic, there appears to be a high amount of news replication, of which there are two major types:
1. 'Churnalism': copy-paste press releases, sometimes with minor adjustments such as rewording and reorganization, and less often, the insertion or deletion of paragraphs. 
2. Direct reposting or paraphrasing of articles from major media sources 
   - In the U.S., there are recognized to be the Associated Press (AP) and Reuters
   
### Question 2: How do duplicated articles evolve from their original texts as they move through the chain of publishers?
*Goal*: investigate whether or not there are consistent differences in the changes made to article texts (e.g. the changing valence of statements based on topic, etc.) based on the known political leaning of the publishers that repost the articles. 

---
#### Algorithms
_NOTE: This section provides a general overview of the algorithms used and should not be referred to as a comprehensive guide. For specific implementation details and useful visualization/debugging/other functionalities, please refer to the code available in the [scripts](scripts) folder_
- [ ] _Article Similarity Detection_ (python [scripts/OneDayDuplicationDetection.py](scripts/OneDayDuplicationDetection.py) [YYYYmmdd])
     - [x] Parsing Documents [scripts/documents.py](scripts/documents.py)
    ```
    class Document(raw_text, para_sep = "###", parser = "spacy", text_encoding = "utf8", text_language = "english")
      - Identifies paragraphs in the raw text by separating by para_sep, filtering out paragraphs that contain < 4 words or ALL CAPS.
      - Creates lists of sentences and their corresponding bag-of-word (bow) dictionaries ({"token": count})
        - Keeps only the first instance of every sentence, and discards duplicates (repeated sentences OR bow's)
        - Tokens are lowercase lemmatized ("spacy") or stemmed ("nltk") words, identified by the repective parser, with at least one alphanumeric character.
      - If parser == "spacy" (default), also creates an average Document vector from [paragraph-specific word vectors learned by the "en_core_web_md" spacy model](https://spacy.io/usage/vectors-similarity) 
    ```
     - [x] Computing Article Similarity Scores [scripts/comparisons.py](scripts/comparisons.py)
    ```
    class DocumentComparisons(thresh_jaccard = .5, thresh_same_sent = .9)
      - thresh_jaccard: The minimum Jaccard index of a pair of sentences to be considered a possible match
      - thresh_same_sent: The minimum Jaccard index of a pair of sentences to be considered a definite match  
      - jaccard_index(bow_a, bow_b):
        - Computes the jaccard index of two bow dictionaries, defined as |intersection|/|union|
      - compute_jaccard_matrix(source, target):
        - Takes two Documents with s and t sentences and constructs a s by t matrix of pairwise sentence Jaccard indices. 
        - HEURISTIC: Simply computes the Jaccard index as 0 for any target sentence that does not fall in the required word length bounds to meet thresh_jaccard 
      - compute_match_matrix(jaccard_matrix):
        - Takes a jaccard matrix and constructs a match (weight) matrix of the same size. 
        - Initializes the match matrix elements as 1 if the corresponding jaccard matrix element meets thresh_jaccard, else 0.
        - For each non-zero row (/column) in the jaccard matrix
          - If the maximum element at [i, j] meets thresh_same_sent, considers sentences i and j a definite match by setting match_matrix[i, j] to 1 and all other elements in the same row (i) and column (j) to 0.
          - Otherwise, normalizes the corresponding row (/column) in the match matrix to sum to 1. 
      - jaccard_score(source, target):
        - Computes the similarity score between two documents as the weighted sum of the jaccard and match matrices.
    ```
  - [x] Multiprocessing [scripts/comparisonsmachine.py](scripts/comparisonsmachine.py)
  
- [ ] Article Origin Detection 


