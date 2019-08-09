from collections import Counter # contains word counter function
import nltk # tokenize words and sentences based on language
# Following line may be necessary to run nltk calls:
# nltk.download('punkt')

class Document:
    def __init__(self, text, para_sep = "###", clean = True, min_para = 4, stem = True,
                text_encoding = "utf8", text_language = "english"):
        # Document Parameters 
        self.para_sep = para_sep 
        self.clean = clean # removes 'bad' paragraphs and duplicate sentences 
        self.min_para = min_para # cannot be None if self.clean is true!
        self.stem = stem # Should words be stemmed?
        self.text_encoding = text_encoding
        self.text_language = text_language
        # Parsed document objects 
        self.paragraphs = None 
        self.sentences = None # List of sentences 
        self.sent_para_map = None # List of corresponding paragraphs indices of sentences        
        self.bow_sentences = None # List of bag-of-word dictionaries 
        self.bow_sent_map = None # List of corresponding bow dictionary indices of sentences 
        self.bow_sent_lens = None # List of lengths of each bow sentence 
        self.parse_text(text)

    def __str__(self):
        if self.paragraphs is not None:
            return ("\n").join(self.paragraphs)

    def parse_text(self, text):
        ''' Calls several parsing functions on the text to extract
        instance variables (i.e. paragraph/sentence/bow lists)
        '''
        self.parse_paragraphs(text)
        self.parse_sentences()
        self.parse_bow_sentences()

    def parse_paragraphs(self, text):
        ''' Stores list of paragraphs separated by para_sep; 
        excluding paragraphs with fewer than min_para words or only CAPS
        '''
        self.paragraphs = text.split(self.para_sep) if text is not None else text 
        if self.clean:
            self.paragraphs = [p for p in paragraphs if
                               len(p.split()) >= self.min_para and not p.isupper()]
        return self.paragraphs

    def parse_sentences(self):
        """ From self.paragraphs, processes and stores a list of sentences and 
        their paragraph mappings, excluding duplicates
        """
        self.sentences = []
        self.sent_para_map = []
        for i, para in enumerate(self.paragraphs):
            append_sents = nltk.sent_tokenize(para, self.text_language)
            if self.clean:
                append_sents = [s for s in append_sents if s not in self.sentences]
            self.sentences.extend(append_sents)
            self.sent_para_map.extend([i] * len(append_sents))
        return self.sentences

    def parse_bow_sentences(self):
        ''' Returns a list of dictionaries for each sentence in self.sentences,
        where the keys are words and values are counts within the sentences, 
        again excluding duplicates
        '''
        self.bow_sentences = []
        self.bow_sent_map = []
        self.bow_sent_lens = []
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        ps = nltk.PorterStemmer()
        for i, sent in enumerate(self.sentences):
            tokens = tokenizer.tokenize(sent)
            if self.stem:
                tokens = [ps.stem(token) for token in tokens]
            tokens = dict(Counter([token for token in tokens if len(token) > 1]))
            if len(tokens) > 0 and (tokens not in self.bow_sentences or (not self.clean)):
                self.bow_sentences.append(tokens)
                self.bow_sent_map.append(i)
                self.bow_sent_lens.append(len(tokens))
        return self.bow_sentences

    def get_sentences(self):
        ''' Returns sentences; if raw, then also returns duplicates
        '''
        if len(self.bow_sent_map) > 0:
            return [self.sentences[i] for i in self.bow_sent_map]
        return self.sentences

    def get_bow_sentences(self):
        return self.bow_sentences
        
    def get_bow_sentence_lens(self):
        return self.bow_sent_lens 