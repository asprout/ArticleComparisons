import documents

class SentenceMatcher:
    ''' The SentenceMatcher takes in two sentences (sentence1 and sentence2) which are each an
        instance of the Sentence class, a cut-off value for the jaccard index (jaccard_threshold)
        and the minimum sentence length (min_sentence_length) -- measured in words -- necessary
        for each sentence in order for the two sentences to be considered a match.
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

class MatchCandidate:
    ''' A MatchCandidate object provides information on a potential sentence match. It takes in a sentence id
            for each of the two sentences that make up the potential match (sent1_id, sent2_id),
            as well as a "match" value describing the potential sentence match as "yes," "no," or "maybe."
             For a description of the match values, see the is_match function in the SentenceMatcher class'''
    def __init__(self, sent1_id, sent2_id, match):
        self.sent1_id = sent1_id
        self.sent2_id = sent2_id
        self.match = match


class History:
    def __init__(self, ):

class DocumentMatcher:
    def __init__(self, document1, document2, min_consec_candidates):


