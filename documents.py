from collections import Counter # contains word counter function
import nltk # tokenize words and sentences based on language
# Following line may be necessary to run nltk calls:
# nltk.download('punkt')

class Document:
    def __init__(self, text, clean = False, para_sep = "###", stem = True,
                text_encoding = "utf8", text_language = "english"):
        self.text = text  # Raw text 
        self.clean = clean # True if text is already preprocessed
        self.stem = stem # Should words be stemmed?
        self.text_encoding = text_encoding
        self.text_language = text_language
        self.paragraphs = None # List of paragraphs, split by separator
        self.sent_para_map = None # Stores the corresponding paragraphs of indexed sentences        
        self.sentences = None # List of sentences 
        self.bow_sent_map = None # Stores the corresponding bow dictionary of indexed sentences 
        self.bow_sentences = None # List of bag-of-word dictionaries 
        self.parse_text(para_sep = para_sep)

    def __str__(self):
        if self.paragraphs is not None:
            return ("\n").join(self.paragraphs)
        return self.text

    def parse_text(self, para_sep = "###", min_para = 4, remove_dups = True):
        ''' Calls several parsing functions on the text to extract
        instance variables (i.e. paragraph/sentence/bow lists)
        '''
        self.parse_paragraphs(para_sep, min_para)
        self.parse_sentences(remove_dups)
        self.parse_bow_sentences(remove_dups)

    def parse_paragraphs(self, para_sep = "###", min_para = 4):
        ''' Stores list of paragraphs separated by para_sep; if text not clean,
        then excludes paragraphs with fewer than min_para words or only CAPS
        '''
        self.paragraphs = self.text.split(para_sep)
        if not self.clean:
            self.paragraphs = [p for p in self.paragraphs if 
                                len(p.split()) >= min_para and not p.isupper()]
        return self.paragraphs

    def parse_sentences(self, remove_dups = True):
        """ From self.paragraphs, processes and stores a list of sentences and 
        their paragraph mappings, excluding duplicates
        """
        self.sentences = []
        self.sent_para_map = []
        for i, para in enumerate(self.paragraphs):
            append_sents = nltk.sent_tokenize(para, self.text_language)
            if remove_dups:
                append_sents = [s for s in append_sents if s not in self.sentences]
            self.sentences.extend(append_sents)
            self.sent_para_map.extend([i] * len(append_sents))
        return self.sentences

    def parse_bow_sentences(self, remove_dups = True):
        ''' Returns a list of dictionaries for each sentence in self.sentences,
        where the keys are words and values are counts within the sentences, 
        again excluding duplicates
        '''
        self.bow_sentences = []
        self.bow_sent_map = []
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        ps = nltk.PorterStemmer()
        for i, sent in enumerate(self.sentences):
            tokens = tokenizer.tokenize(sent)
            if self.stem:
                tokens = [ps.stem(token) for token in tokens]
            tokens = dict(Counter([token for token in tokens if len(token) > 1]))
            if len(tokens) > 0 and (tokens not in self.bow_sentences or (not remove_dups)):
                self.bow_sentences.append(tokens)
                self.bow_sent_map.append(i)
        return self.bow_sentences

    def get_sentences(self, raw = False):
        ''' Returns sentences; if raw, then also returns duplicates
        '''
        if len(self.bow_sent_map) > 0 and not raw:
            return [self.sentences[i] for i in self.bow_sent_map]
        return self.sentences

    def get_bow_sentences(self):
        return self.bow_sentences
        