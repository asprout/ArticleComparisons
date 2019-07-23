
''' This class contains all of the parameters, hardcoded numbers, and strings.
'''
class Config:
    def __init__(self):
        self.text_encoding='utf8'
        self.text_language = 'english'
        self.jaccard_threshold = .9
        self.min_sentence_length = 4
        self.min_consec_maybes = 3

        self.target_file_name = 'article3.txt'
        self.source_file_name = 'article3_pluspar.txt'


