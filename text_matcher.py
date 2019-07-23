import documents #get document classes
import config # get parameter class
config = Config()

class SentenceMatcher:
    ''' The SentenceMatcher takes in:
         sentence1: an instance of the Sentence class
         sentence2: an instance of the Sentence class
        min_sentence_length (type:int): the minimum sentence length  -- measured in words -- both sentence1
                                and sentence2 must satisfy for the two sentences to be considered a "Yes" a match,
                                GIVEN THAT the jaccard index is above jaccard_threshold.
    '''
    def __init__(self, sentence1, sentence2):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.jaccard_threshold = config.jaccard_threshold
        self.min_sentence_length = config.min_sentence_length
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
            raise TypeError('match must be type of is_match() output: "No" "Yes" or "Maybe".')
        self.sent1_id = sent1_id
        self.sent2_id = sent2_id
        self.match = match

    def __str__(self):
        return str((self.sent1_id, self.sent2_id, self.match))


class History:
    '''A History object is used to keep track of match candidates and help determine which candidates are
         true matches and which candidates can be ruled out as potential matches. A history starts with a single
         MatchCandidate object, but other candidates can be added to the history with the .add_candidate() method'''
    def __init__(self, match_candidate):
        self.candidates = [match_candidate]
        self.min_consec_maybes = config.min_consec_maybes

    def __str__(self):
        return str(self.candidates)

    def length(self):
        ''' Output: An int, the number of match candidates in the list self.candidates '''
        return len(self.candidates)

    def is_next(self, candidate):
        '''Input: candidate: a MatchCandidate instance
           Output: a boolean: True if the sentence indices of candidate match with the history
                              False if the sentence indices do not match'''
        i = self.candidates[-1].sent1_id
        j = self.candidates[-1].sent2_id
        new_i = candidate.sent1_id
        new_j = candidate.sent2_id
        return (i + 1 == new_i and j + 1 == new_j)

    def flush(self, length):
        ''' Output:   - If the history is long enough or contains "Yes" match candidates then
                    it is determine to be a true match and a list of the match candidates are returned.
                     - If the history does not meet the criterion then the match candidates are not a true
                    match and the history is deleted'''
        if self.length() >= length:
            if self.length() >= self.min_consec_maybes:
                matches = history.candidates
            else:
                for candidate in self.candidates:
                    if candidate.match == "Yes":
                        matches = history.candidates
                        break
            return matches
        else:
            del self

    def add_candidate(self, candidate):
        ''' Add candidate '''
        self.candidates += [candidate]


class DocumentMatcher:
    ''' The DocumentMatcher object is used to identify sentence matches between two documents
        target_document: a Document object
        source_document: a Document object
        min_consec_maybe (type: int): The minimum number of consecutive candidates with a match value of "Maybe"
                                        required to consider the sentences corresponding to the candidates as
                                        sentence matches (or copies)'''
    def __init__(self, target_document, source_document):
        self.target_doc = target_document
        self.source_doc = source_document
        self.jaccard_threshold = config.jaccard_threshold
        self.min_sentence_length = config.min_sentence_length
        self.min_consec_maybes = config.min_consec_maybes
        self.target_sentences = [Sentence(s) for s in target_document.sentences]
        self.source_sentences = [Sentence(s) for s in target_document.sentences]
          # a dict where each key is the target sentences index and each value is a list of MatchCandidate instances,
          # starts as an empty list
        self.candidates = {k:[] for k,sent in enumerate(self.target_sentences)}
        self.histories = [] # a list of History instances, starts as an empty list, updated during the find_matches method
        self.matches = [] # a list of all of the sentence matches, starts as an empty list, updated by the find_matches method

    def set_candidates(self):
        ''' Output: sets self.candidates attribute to be such that each value is a list of all of the sentence match
                candidates in the document
        '''
        for i, target_s in enumerate(self.target_sentences):
            for j,source_s in enumerate(self.source_sentences):
                match = SentenceMatcher(target_s, source_s)
                match_value = match.is_match()
                if (match_value in ["Yes", "Maybe"]):   #check if sentences are possibly a match
                    candidate = MatchCandidate(i, j, match_value)
                    self.candidates[i]+=[candidate]         # if possible a match, add to candidates

    def find_matches(self):
        for candidates in self.candidates.values():
            if self.histories == []:
                self.histories = [History(c) for c in candidates]
            else:
                # a list of History instances that have a candidate match
                matched_histories = []
                # a list of History instances that have no candidate match
                unmatched_histories = []
                for history in self.histories:
                    if any(list(filter(lambda x: history.is_next(x), candidates))):
                        matched_histories += [history]
                    else:
                        unmatched_histories += [history]
                # add candidates to matching histories
                for history in matched_histories:
                    for c in candidates:
                        if history.is_next(c):
                            history.add_candidate(c)
                            break
                        else:
                            pass
                self.histories = matched_histories
                # get max length of all histories (matching or unmatching)
                max_length = max([h.length() for h in matched_histories + unmatched_histories])
                # flush unmatching histories
                self.matches += list(map(lambda x: x.flush(max_length), unmatched_histories))

         #flush all remaining histories
        max_length = max([h.length() for h in self.histories])
        self.matches += list(map(lambda x: x.flush(max_length), self.histories))










