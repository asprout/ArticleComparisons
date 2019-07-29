from collections import Counter # contains word counter function
import nltk # tokenize words and sentences based on language
# Following line may be necessary to run nltk calls:
# nltk.download('punkt')

class Document:
    def __init__(self, text, clean = False, stem = True,
                text_encoding = "utf8", text_language = "english"):
        self.text = text 
        self.clean = clean # Used to determine if text is already preprocessed
        self.stem = stem # Should words be stemmed?
        self.text_encoding = text_encoding
        self.text_language = text_language
        self.paragraphs = None
        self.sent_para_map = None # Stores the paragraphs of each sentence 
        self.sentences = None
        self.bow_sent_map = None # Stores the sentence of each bow
        self.bow_sentences = None
        self.parse_text()

    def __str__(self):
        if self.clean:
            return self.text
        return ("\n").join(self.paragraphs)

    def parse_text(self, para_sep = "\n", min_para = 4, remove_dups = True):
        self.parse_paragraphs(para_sep, min_para)
        self.parse_sentences(remove_dups)
        self.parse_bow_sentences(remove_dups)

    def parse_paragraphs(self, para_sep = "\n", min_para = 4):
        if self.clean:
            self.paragraphs = self.text.split(para_sep)
        else: # Clean up bad text from scraped article
            paras = self.text.split("###")
            paras = [p for p in paras if len(p.split()) >= min_para and not p.isupper()]
            self.paragraphs = paras 
        return self.paragraphs

    def parse_sentences(self, remove_dups = True):
        """ From self.paragraphs, processes and stores a list of sentences and 
        their paragraph mappings, ignoring duplicates
        """
        sentences = []
        sent_para_map = []
        for i, para in enumerate(self.paragraphs):
            append_sents = nltk.sent_tokenize(para, self.text_language)
            if remove_dups:
                append_sents = [sent for sent in append_sents if sent not in sentences]
            for sent in append_sents:
                sentences.append(sent)
                sent_para_map.append(i)
        self.sentences = sentences 
        self.sent_para_map = sent_para_map
        return self.sentences

    def parse_bow_sentences(self, remove_dups = True):
        ''' Returns a list of dictionaries for each sentence in self.sentences,
        where the keys are words and values are counts within the sentences
        '''
        bow_sentences = []
        bow_sent_map = []
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        ps = nltk.PorterStemmer()
        for i, sent in enumerate(self.sentences):
            tokens = tokenizer.tokenize(sent)
            if self.stem:
                tokens = [ps.stem(token) for token in tokens]
            if len(tokens) > 0 and (tokens not in bow_sentences or not remove_dups):
                bow_sentences.append(dict(Counter(tokens)))
                bow_sent_map.append(i)
        self.bow_sentences = bow_sentences
        self.bow_sent_map = bow_sent_map
        return self.bow_sentences

    def get_sentences(self):
        if len(self.bow_sent_map) > 0:
            return [self.sentences[i] for i in self.bow_sent_map]
        return self.sentences

    def get_bow_sentences(self):
        return self.bow_sentences