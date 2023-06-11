from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np
import argostranslate.package
import argostranslate.translate
from thefuzz import fuzz


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(
                model, low_cpu_mem_usage=True
            ),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])


class KBIR_kw_extraction:
    def __init__(self):
        self.documents = []
        self.model_name = "ml6team/keyphrase-extraction-kbir-inspec"
        self.kbir_extractor = KeyphraseExtractionPipeline(model=self.model_name)
        self.kbir_extractor.model.eval()
        self.translated_texts = []
        self.keywords = []
        self.blacklist = ["дисциплин", "билет"]

    def __download_and_install_argos(self, from_code, to_code):
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == from_code and x.to_code == to_code,
                available_packages,
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())

    def __translate_documents(self, from_code="ru", to_code="en"):
        self.__download_and_install_argos(from_code, to_code)
        for doc in self.documents:
            self.translated_texts.append(
                argostranslate.translate.translate(
                    doc["text"], from_code, to_code
                ).replace("\n", " ")
            )

    def __extract_kw_en(self):
        kw_en = []
        for text in self.translated_texts:
            print("KBIR extract words:\n", self.kbir_extractor(text))
            kw_en.append(self.kbir_extractor(text))
            print("end extract")
        return kw_en

    def __translate_kw_to_ru(self, kw_en):
        from_code = "en"
        to_code = "ru"
        self.__download_and_install_argos(from_code, to_code)
        for i in kw_en:
            arr = []
            for word in i:
                arr.append(argostranslate.translate.translate(word, from_code, to_code))
            self.keywords.append(arr)

    def __get_ratio(self):
        sentences = []
        ratio = []
        for i in range(len(self.keywords)):
            text = self.documents[i]["text"]
            arr = []
            s = []
            for word in self.keywords[i]:
                number = fuzz.partial_ratio(word, text)
                s1 = text
                for sentence in text.split("."):
                    if len(word) > len(sentence):
                        continue
                    if fuzz.partial_ratio(word, sentence) >= number:
                        number = fuzz.partial_ratio(word, sentence)
                        s1 = sentence

                arr.append(number)
                s.append(s1)

            sentences.append(s)
            ratio.append(arr)

        return sentences, ratio

    def __search_word_in_sentence(self, word, sentence):
        ratio = 0
        choosen_word = ""
        for w in sentence.split(" "):
            r = fuzz.ratio(w, word)
            if r > ratio:
                ratio = r
                choosen_word = "" + w
        return choosen_word

    def __check_blacklist_words(self, kw):
        new_res = []
        print("CHECK BL WORDS")
        for words in kw:
            arr = []
            for word in words:
                flag = 0
                for black in self.blacklist:
                    if black in word.lower():
                        print(
                            word,
                            "____________",
                            black,
                            fuzz.ratio(word.lower(), black.lower()),
                        )
                        if fuzz.ratio(word.lower(), black) > 80:
                            flag = 1
                if not flag:
                    arr.append(word)
            new_res.append(arr)
        return new_res

    def __choose_phrase(self, phrase, sentence):
        arr = []
        sent = sentence.split()
        words_in_sent = len(sent)
        for i in range(1, len(phrase.split()) + 2):
            j = 0
            while j + i <= words_in_sent:
                arr.append((" ").join(sent[j : j + i]))
                j += 1

        m = 0
        choosen_phrase = ""
        for ph in arr:
            r = fuzz.ratio(phrase, ph)
            if r > m:
                m = 0 + r
                choosen_phrase = "" + ph
        return choosen_phrase

    def __choose_phrases(self):
        punctuation = [";", ",", ".", ":"]
        sentences, ratio = self.__get_ratio()
        result = []
        for i in range(len(ratio)):
            arr = []
            for j in range(len(ratio[i])):
                # print(ratio[i][j], self.keywords[i][j])
                # print(sentences[i][j])
                if ratio[i][j] >= 80:
                    words = []
                    choosen_word = ""

                    # for word in self.keywords[i][j].split(" "):
                    #    choosen_word = self.__search_word_in_sentence(
                    #        word, sentences[i][j]
                    #    )
                    #    words.append(choosen_word)
                    # new_phrase = (" ").join(words)
                    new_phrase = self.__choose_phrase(
                        self.keywords[i][j], sentences[i][j]
                    )
                    if new_phrase[-1] in punctuation:
                        new_phrase = new_phrase[:-1]
                    arr.append(new_phrase)
                    # print(new_phrase)
                # print()

            result.append(list(set(arr)))
        result = self.__check_blacklist_words(result)
        return result

    def get_keywords(self, documents):
        self.documents = documents
        self.translated_texts = []
        self.keywords = []
        self.__translate_documents()
        kw_en = self.__extract_kw_en()
        self.__translate_kw_to_ru(kw_en)
        choosen_phrases = self.__choose_phrases()
        result = []
        for doc, kw, res in zip(self.documents, self.keywords, choosen_phrases):
            result.append(
                {
                    "text_system_id": doc["system_id"],
                    "all_keywords": kw,
                    "choosen_keywords": res,
                }
            )
        return result
