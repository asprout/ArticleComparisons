import numpy as np 
from collections import Counter # contains word counter function
import spacy
import nltk # tokenize words and sentences based on language
# Following line may be necessary to run nltk calls:
# nltk.download('punkt')
# For spacy:
NLP = spacy.load("en_core_web_md", disable = ['tagger'])
# May need to run: python -m spacy download en_core_web_md (or sm for small)

import utils

class Document:
    def __init__(self, text, para_sep = "###", parser = "spacy", clean = True, 
                       text_encoding = "utf8", text_language = "english"):
        # Document parsing parameters (should NOT be changed except for dev purposes!)
        self.clean = clean # removes 'bad' paragraphs and duplicate sentences 
        self.min_para = 4 # minimum number of words in paragraph
        self.stem = True # Should words be stemmed?
        # Document-specific Parameters 
        self.para_sep = para_sep
        self.text_encoding = text_encoding
        self.text_language = text_language # only handling english for now 
        # Parsed document objects 
        self.invalid = 0
        self.nchar = 0
        self.vec = None
        self.sentences = None # List of sentence strings 
        self.sent_para_map = None # List of corresponding paragraph numbers of each sentence       
        self.bow_sentences = None # List of bag-of-word dictionaries per sentence 
        self.sent_entities = None # List of entities per sentence 
        self.bow_sent_map = None # List of corresponding sentence indices for BOW dictionaries
        self.bow_sent_lens = None # List of lengths (number of words) of each bow sentence 
        if parser == "spacy":
            self.parse_spacy(text)
        else:
            self.parse_nltk(text)
        self.checkValid()

    def __str__(self):
        if self.sentences is None: 
            return None
        paras = []
        for para in np.unique(self.sent_para_map):
            para = (" ").join([self.sentences[i] for i in np.where(self.sent_para_map == para)[0]])
            paras.append(para)
        return ("\n").join(paras)

    def parse_paragraphs(self, text):
        ''' Returns list of paragraphs, excluding those below word threshold or CAPS-only '''
        self.nchar = len(text)
        paras = text.split(self.para_sep) if text is not None else text 
        if self.clean:
            paras = [p for p in paras if len(p.split()) >= self.min_para and not p.isupper()]
        return paras

    def parse_spacy(self, text):
        sentences = []
        sent_para_map = []
        bow_sentences = []
        sent_entities = []
        bow_sent_map = []
        bow_sent_lens = []
        vectors = []
        paras = self.parse_paragraphs(text)

        for i, para in enumerate(paras):
            para_span = NLP(para) # Spacy document object 
            vectors.append(para_span.vector)
            for s_span in para_span.sents: # for each Spacy sentence object 
                if self.clean and (len(s_span.text) == 0 or s_span.text in sentences):
                    continue # Not a valid 'clean' sentence 
                sentences.append(s_span.text)
                sent_para_map.append(i)
                # parse tokens 
                tokens = [t.lemma_ for t in s_span] if self.stem else [t.text for t in s_span]
                tokens = dict(Counter([t.lower() for t in tokens if utils.isalnum(t)]))
                if not self.clean or (len(tokens) > 0 and tokens not in bow_sentences):
                    bow_sentences.append(tokens)
                    sent_entities.append([ent.lemma_.lower() if (
                        ent.label_ in ['PERSON', 'GPE', 'ORG'] or 
                        ent.label_ == "DATE" and len(ent.text) >= 10) else None
                        for ent in s_span.ents])
                    bow_sent_map.append(len(sentences) - 1)
                    bow_sent_lens.append(len(tokens))

        self.sentences = np.array(sentences)
        self.sent_para_map = np.array(sent_para_map)
        self.bow_sentences = np.array(bow_sentences)
        self.sent_entities = np.array(sent_entities)
        self.bow_sent_map = np.array(bow_sent_map)
        self.bow_sent_lens = np.array(bow_sent_lens)
        self.vec = np.array(vectors).mean(axis = 0) if len(vectors) > 0 else None
        return self.bow_sentences

    def parse_nltk(self, text):
        ''' Calls several parsing functions on the text to extract
        instance variables (i.e. paragraph/sentence/bow lists)
        '''
        self.parse_sentences(text)
        self.parse_bow_sentences()

    def parse_sentences(self, text):
        '''Processes and stores a list of sentences and 
        corresponding paragraph mappings, excluding duplicates '''
        # List of paragraphs, excluding those below word threshold or CAPS-only
        paras = self.parse_paragraphs(text)
        # Extract sentences and the paragraphs they correspond to 
        sentences = []
        sent_para_map = []
        for i, para in enumerate(paras):
            append_sents = nltk.sent_tokenize(para, self.text_language)
            if self.clean:
                append_sents = [s for s in append_sents if s not in sentences]
            sentences.extend(append_sents)
            sent_para_map.extend([i] * len(append_sents))
        self.sentences = np.array(sentences)
        self.sent_para_map = np.array(sent_para_map)
        return self.sentences

    def parse_bow_sentences(self):
        ''' Returns a list of dictionaries for each sentence in self.sentences,
        where the keys are words and values are counts within the sentences, 
        again excluding duplicates
        '''
        bow_sentences = []
        bow_sent_map = []
        bow_sent_lens = []
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        ps = nltk.PorterStemmer()
        for i, sent in enumerate(self.sentences):
            tokens = tokenizer.tokenize(sent)
            if self.stem:
                tokens = [ps.stem(token) for token in tokens]
            # count of tokens in a sentence, excluding non-alphanumeric words 
            tokens = dict(Counter([token.lower() for token in tokens if utils.isalnum(token)]))
            if len(tokens) > 0 and (tokens not in bow_sentences or (not self.clean)):
                bow_sentences.append(tokens)
                bow_sent_map.append(i)
                bow_sent_lens.append(len(tokens))
        self.bow_sentences = np.array(bow_sentences)
        self.bow_sent_map = np.array(bow_sent_map)
        self.bow_sent_lens = np.array(bow_sent_lens)
        return self.bow_sentences

    def get_sentences(self, raw = True):
        if not raw:
            return [self.sentences[i] for i in self.bow_sent_map]
        return self.sentences

    def get_sentence(self, bow_ind):
        return self.sentences[self.bow_sent_map[bow_ind]]

    def get_bow_sentences(self):
        return self.bow_sentences
        
    def get_bow_sentence_lens(self):
        return self.bow_sent_lens 

    def checkValid(self):
        penalty = 0.5 * (self.nchar < 500)
        penalty += 1 * (len([1 for sent in self.sentences if utils.keywordsin(sent)]) > len(self.sentences)/2)
        entities = [ent for ent in utils.flatten(self.sent_entities) if ent is not None]
        penalty += 2 * (len(entities) < 1)
        self.invalid = penalty
