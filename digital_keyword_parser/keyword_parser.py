from wikineural import Wikineural
from kbir import KBIR_kw_extraction
from normalization import Normalization


class KeywordParser:
    def __init__(self, synonyms_checker):
        self.wiki = Wikineural()
        self.kbir = KBIR_kw_extraction()
        self.norm = Normalization()
        self.synonyms = synonyms_checker

    def get_keywords(self, documents):
        result = []
        kw_kbir = self.kbir.get_keywords(documents)
        kw_wiki = self.wiki.get_keywords(documents)
        for ind in range(len(documents)):
            wiki_words_i = [j["word"] for j in kw_wiki[ind]["words"]]
            result.append(
                {
                    "text_system_id": documents[ind]["system_id"],
                    "kbir": {
                        "all_kw": kw_kbir[ind]["all_keywords"],
                        "choosen_kw": kw_kbir[ind]["choosen_keywords"],
                        "normalized": self.norm.normalize(
                            kw_kbir[ind]["choosen_keywords"]
                        ),
                    },
                    "wiki": {
                        "ner": kw_wiki[ind]["ner"],
                        "kw_with_entities": kw_wiki[ind]["words"],
                        "kw": wiki_words_i,
                        "normalized": self.norm.normalize(wiki_words_i),
                    },
                }
            )
            result[-1]["clustering"] = self.synonyms.get_synonyms([result[-1]])
        return result
