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


#### Algorithms
_NOTE: This section provides a general overview of the algorithms used and should not be referred to as a comprehensive guide. For more specific implementation details, please refer to the [code](scripts)_
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
      - jaccard_index(bow_a, bow_b):
        - Computes the jaccard index of two bow dictionaries, defined as |intersection|/|union|
      - compute_jaccard_matrix(source, target):
        - Takes two articles and computes 
    ```
  - [x] Multiprocessing [scripts/comparisonsmachine.py](scripts/comparisonsmachine.py)
  
- [ ] Article Origin Detection 


