import documents

class SentenceMatcher:
    ''' The SentenceMatcher takes in:
         sentence1: an instance of the Sentence class
         sentence2: an instance of the Sentence class
         jaccard_threshold (type: float): a cut-off value for the jaccard index, below which the two sentences will be
                            considered definitely not a match and assigned a match value of "No"
        min_sentence_length (type:int): the minimum sentence length  -- measured in words -- both sentence1
                                and sentence2 must satisfy for the two sentences to be considered a "Yes" a match,
                                GIVEN THAT the jaccard index is above jaccard_threshold.
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
                intsec = intsec + min(self.wordbag1.get(word), self.wordbag2.get(word))
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
        if not(match in ["No", "Maybe", "Yes"]):
            raise TypeError('match must be type of is_match() output: "No", "Yes", or "Maybe".')
        self.sent1_id = sent1_id
        self.sent2_id = sent2_id
        self.match = match

class History:
    '''A History object is used to keep track of match candidates and help determine which candidates are
         true matches and which candidates can be ruled out as potential matches. A history starts with a single
         MatchCandidate object, but other candidates can be added to the history with the .add_candidate() method'''
    def __init__(self, match_candidate):
        self.candidates = [match_candidate]

    def length(self):
        ''' Output: An int, the number of match candidates in the list self.candidates '''
        return len(self.candidates)

    def is_next(self, candidate):
        '''Input: candidate: a MatchCandidate instance
           OUtput: a boolean: True if the sentence indices of candidate match with the history
                              False if the sentence indices do not match'''
        i = self.candidates[-1].sent1_id
        j = self.candidates[-1].sent2_id
        new_i = candidate.sent1_id
        new_j = candidate.sent2_id
        return (i + 1 == new_i and j + 1 == new_j)

    def flush_history(self, length):
        '''  '''
        if self.length() >= length:
            if self.length() >= min_consec_maybe:
                copied_sentences = history.candidates
            else:
                for candidate in self.candidates:
                    if candidate.match == "Yes":
                        copied_sentences = history_matches
                        break


    def add_candidate(self, candidate):
        ''' Add candidate '''
        self.candidates += [candidate]

    def rule_out_history(self, length):
        if self.length() < length:
            del self


class DocumentMatcher:
    def __init__(self, target_document, source_document, min_consec_maybe):
        self.target = document1.id_wordbags()
        self.source = document2.id_wordbags()
        self.min_consec_candidates = min_consec_maybe

    def get_match_candidates(self, jaccard_threshold, min_sentence_length):
        for sentence in self.target:
            for sentence in self.source:
                SentenceMatcher()






