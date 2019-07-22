import documents

class SentenceMatcher:
    ''' The SentenceMatcher

    '''
    def __init__(self, sentence1, sentence2, jaccard_threshold, min_sentence_length):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.jaccard_threshold = jaccard_threshold
        self.min_sentence_length = min_sentence_length
        self.wordbag1 = sentence1.wordbag()
        self.wordbag2 = sentence2.wordbag()
        self.sent1_length = sentence1.word_length
        self.sent2_length = sentence2.word_length

    def jaccard_index(self):
        '''  Output: The Jaccard similarity value (type: float) between the wordbag1 and wordbag2
                        https://en.wikipedia.org/wiki/Jaccard_index '''
        intsec = 0  # start intersection counter at 0
        for word in self.wordbag1.keys():
            if word in self.wordbag2.keys():
                intsec = intsec + min(wordbag1.get(word), wordbag2.get(word))
        wordbag1_size = sum(self.wordbag1.values())
        wordbag2_size = sum(self.wordbag2.values())
        union = wordbag1_size + wordbag2_size - intsec
        return float(intsec / union)

    def is_match(self):
        '''Output: "No" if Jaccard similarity threshold is not met
                   "Maybe" if Jaccard similarity threshold is met but both sentences do not exceed
                            min_sentence_length
                   "Yes" if Jaccard similarity threshold is met but both sentences exceed
                            min_sentence_length '''
        match = "No"
        if self.jaccard_index() >= self.jaccard_threshold:
            if (self.sent1_length and self.sent2_length) >= self.min_sentence_length:
                match = "Yes"
            else:
                match = "Maybe"
        return match



class History:
    pass

class DocumentMatcher:
    def __init__(self, min_consec_candidates):


