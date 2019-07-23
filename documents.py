
'''

'''

##########   IMPORT PACKAGES         #########

import Config # get parameter class
import string #remove punctuation
from collections import Counter #contains word counter function
import nltk #tokenize words and sentences based on language

###################################################
class Document:
    def __init__(self):
        config = Config()
        self.file_name = config.file_name
        self.text_encoding = config.text_encoding
        self.language = config.text_language
        with open(self.file_name, 'r+', encoding=self.text_encoding) as text_file:
            self.text = text_file.read()
        self.paragraphs = self.text.split("\n")  # A list of document paragraphs
        self.sentences = nltk.sent_tokenize(self.text, language=self.text_language) # A list of document sentences
        self.sentence_length = len(self.sentences)

    def __str__(self):
        return self.text

    def id_sentences(self):
        ''' Create a dictionary where each key (type: int) is an index for the sentence location
         in the document text and the value (type: string) is sentence text.
            Example: document text: "Hello! Today it is sunny. The sky is blue."
                    output: {0: 'Hello!', 1: 'Today it is sunny.', 2: 'The sky is blue.'} '''
        return {k:v for k,v in enumerate(self.sentences)}

    def id_wordbags(self):
        ''' Create a dictionary where each key (type: int) is an index for the sentence location
         in the document text and the value (type: dict) is a bag of words representation of the sentence.
            Example: document text: "Hello! Today it is sunny."
                    output: {0: {'hello':1}, 1: {'Today':1, 'it':1, 'is':1 'sunny':1}} '''

        new_dict = self.id_sentences()
        for k,v in new_dict.items():
            new_dict[k] = Sentence(v).wordbag()
        return new_dict


    def export_html(self):
        pass


class Paragraph:
    def __init__(self, text_string, text_encoding='utf8', text_language='english'):
        self.text = text_string
        self.encoding = text_encoding
        self.language = text_language
        self.sentences = nltk.sent_tokenize(self.text, language=text_language)

    def __str__(self):
        return self.text

    def export_html(self):
        pass


class Sentence:
    def __init__(self, text_string, text_encoding='utf8', text_language='english'):
        self.text = text_string
        self.encoding = text_encoding
        self.language = text_language
        self.words = nltk.word_tokenize(self.text.translate(str.maketrans('', '', string.punctuation)),
                                        language=text_language) # A list of sentence words without punctuation
        self.word_length = len(self.words)

    def __str__(self):
        return self.text

    def wordbag(self):
        ''' From the sentence text, create a dictionary where each key (type: str) is a unique word in the
         sentence text and each value (type: int) is the number of times the key word appears in the sentence.
            Example: for sentence: "The sky is blue, the sun is not blue."
                    output: {'the':2, 'sky':1, 'is':2, 'blue': 2, 'sun':1, 'not':1} '''
        sentence = self.text.lower() # make all words lowercase
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        return dict(Counter(nltk.word_tokenize(sentence, language=self.language)))

    def jaccard_index(self, sentence):
        ''' Input: sentence (type: Sentence class), the sentence to compare
            Output: The Jaccard similarity value (type: float) between the word bags of each sentence
                        https://en.wikipedia.org/wiki/Jaccard_index '''
        intsec = 0  # start intersection counter at 0
        wordbag1 = self.wordbag()
        wordbag2 = sentence.wordbag()
        for word in wordbag1.keys():
            if word in wordbag2.keys():
                intsec = intsec + min(wordbag1.get(word), wordbag2.get(word))
        wordbag1_size = sum(wordbag1.values())
        wordbag2_size = sum(wordbag2.values())
        union = wordbag1_size + wordbag2_size - intsec
        return float(intsec / union)


    def export_html(self):
        pass