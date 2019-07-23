''' '''
class Main:


target_doc = Document(config.target_file_name)
source_doc = Document(config.source_file_name)
text_matcher = DocumentMatcher(target_doc, source_doc)
text_matcher.set_candidates()
self.candidates