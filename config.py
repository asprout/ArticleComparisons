
''' This class contains all of the parameters, hardcoded numbers, and strings.
'''
class Config:
    def __init__(self):

        self.text_encoding='utf8'

        self.text_language = 'english'

        # jaccard_threshold(type: float): a cut - off value for the jaccard index, below which the two sentences
        # will be considered definitely not amatch and assigned a match value of "No"
        self.jaccard_threshold = .9

        # min_sentence_length (type:int): the minimum sentence length  -- measured in words -- both sentence1
        #  and sentence2 must satisfy for the two sentences to be considered a "Yes" a match,
        #  GIVEN THAT the jaccard index is above jaccard_threshold.
        self.min_sentence_length = 4

        self.min_consec_maybes = 3

        self.target_file_name = 'article3.txt'
        self.source_file_name = 'article3.txt'


