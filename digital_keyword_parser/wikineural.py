from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import numpy as np
from thefuzz import fuzz
import re


class Wikineural:
    def __init__(self):
        self.documents = []
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Babelscape/wikineural-multilingual-ner"
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            "Babelscape/wikineural-multilingual-ner", low_cpu_mem_usage=True
        )
        self.model.eval()
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        self.symbols = [" ", ".", ",", ":", ";", "(", ")", "!", "?", '"']
        self.words_blacklist = [
            "ЛЭТИ",
            "СПбГЭТУ",
            "науки и высшего образования Российской Федерации",
            "дисциплин",
            "билет",
            "Ленина",
        ]

    def __what_entity_is_this(self, ent, entities):
        # print('__what_entity_is_this')
        """A function for counting the number
        of each type of entity in a certain word

        Args:
            ent (string): entity of 1 piece of a certain word
            entities (list<integer>): array with all entities of a certain word
        """
        if ent == "I-PER" or ent == "B-PER":
            entities[0] += 1

        elif ent == "I-ORG" or ent == "B-ORG":
            entities[1] += 1

        elif ent == "I-LOC" or ent == "B-LOC":
            entities[2] += 1

        elif ent == "I-MISC" or ent == "B-MISC":
            entities[3] += 1

    def __connect_entities(self, arr, text):
        # print('__connect_entities')
        """Some words divide into parts. This function connect
        this parts in one word or phrase.

        Args:
            arr (list): array of entities. Consists of word, start, end and type of entity
            text (string): text from which the entities were extracted

        Returns:
            (list): array of entities
        """
        if len(arr) == 0:
            return []
        arr2 = []
        prev = arr[0]["word"]
        prev_end = arr[0]["end"]
        prev_start = arr[0]["start"]
        entities = [0, 0, 0, 0]  # per, org, loc, misc
        # make dictionary? ↑
        self.__what_entity_is_this(arr[0]["entity"], entities)

        for i in range(1, len(arr)):
            if prev_end == arr[i]["start"]:
                prev += arr[i]["word"]
                self.__what_entity_is_this(arr[i]["entity"], entities)

            elif prev_end == arr[i]["start"] - 1:
                prev += text[arr[i]["start"] - 1] + arr[i]["word"]
                self.__what_entity_is_this(arr[i]["entity"], entities)

            else:
                self.__what_entity_is_this(arr[i]["entity"], entities)
                phrase = prev.replace("#", "")
                arr2.append(
                    {
                        "word": phrase,
                        "start": prev_start,
                        "end": prev_end,
                        "entity": entities,
                    }
                )
                prev = arr[i]["word"]
                prev_start = arr[i]["start"]
                entities = [0, 0, 0, 0]

            prev_end = arr[i]["end"]

        phrase = prev.replace("#", "")
        self.__what_entity_is_this(arr[len(arr) - 1]["entity"], entities)
        arr2.append(
            {"word": phrase, "start": prev_start, "end": prev_end, "entity": entities}
        )
        return arr2

    def __looking_for_words(self, word_info, text):
        # print('__looking_for_words')
        # поиск слов
        # метод, критерий, теорема, система уравнений, цикл, расстояние, сети
        # диаграмма, триангуляция, операционная система, система, технология, модель, архитектура, модель
        start = word_info["start"]
        l = len(text)
        while start > 0 and text[start] != "." and text[start] != ";":
            start -= 1

        sent = text[start : word_info["start"]]

        sent_words = re.sub('[.,;:"]', "", sent).split()
        before = word_info["word"][:]

        words = [
            "метод",
            "критерий",
            "теорема",
            "цикл",
            "расстояние",
            "сети",
            "диаграмма",
            "классификация",
            "форма",
            "преобразование",
        ]

        for i in words:
            if fuzz.partial_ratio(i, sent) < 70:
                continue
            for j in sent_words[::-1][:2]:
                if fuzz.ratio(j, i) > 80:
                    index = sent.rindex(j)
                    word_info["word"] = sent[index:] + word_info["word"]
                    word_info["start"] -= len(sent) - index

        # if before != word_info["word"]:
        #    print("_____", before, "->", word_info["word"], "\n_____", sent, "\n")

        # else:
        #    print(word_info["word"], "\n", sent, "\n")

    def __choose_one_entity_type(self, entities):
        # print('__choose_one_entity_type')
        """Checks if the word contains only 1 entity type or several.
        In first case returns type, else returns array from args.
        """
        ent = list(entities)
        if ent[:3] == [0, 0, 0]:
            return "MISC"
        if ent[1:] == [0, 0, 0]:
            return "PER"
        if ent[:2] == [0, 0] and ent[3] == 0:
            return "LOC"
        if ent[2:4] == [0, 0] and ent[0] == 0:
            return "ORG"

        titles = ["PER", "ORG", "LOC", "MISC"]
        total_count = sum(entities)
        res_string = ""
        for i, j in zip(entities, titles):
            if i != 0:
                res_string += str(round(i / total_count, 3) * 100) + "% " + str(j) + " "
        return res_string

    def __make_full_word(self, arr):
        # print('__make_full_word')
        for i in arr:
            for j in i["words"]:
                while i["text"][j["start"] - 1] not in self.symbols:
                    j["start"] -= 1
                    if j["start"] < 0:
                        break
                    j["word"] = i["text"][j["start"]] + j["word"]
                txtlen = len(i["text"])
                while i["text"][j["end"]] not in self.symbols and txtlen - 1 > j["end"]:
                    j["word"] += i["text"][j["end"]]
                    j["end"] += 1

    def __remove_leti(self, arr):
        # print('__remove_leti')
        new_arr = []
        for i in range(len(arr)):
            new_arr.append(
                {
                    "text": arr[i]["text"],
                    "text_system_id": arr[i]["text_system_id"],
                    "ner": arr[i]["ner"],
                    "words": [],
                }
            )
            for j in arr[i]["words"]:
                if not any(
                    word.lower() in j["word"].lower() for word in self.words_blacklist
                ):
                    # "ЛЭТИ" not in j['word'] and "СПбГЭТУ" not in j['word'] and "науки и высшего образования Российской Федерации" not in j['word']:
                    new_arr[i]["words"].append(
                        {"word": j["word"], "entity": j["entity"]}
                    )
        return new_arr

    def __check_duplicates(self, words):
        # print('__check_duplicates')
        new_words_array = []
        checked_words = []
        for element in words:
            if element["word"] in checked_words:
                continue
            ent = np.array([0, 0, 0, 0])
            for el in words:
                if el["word"] == element["word"]:
                    ent = ent + np.array(el["entity"])
            new_words_array.append({"word": element["word"], "entity": ent})
            checked_words.append(element["word"])
        return new_words_array

    def get_keywords(self, documents):
        print("WIKI. gwt_kws")
        self.documents = documents
        result = []
        for document in self.documents:
            ner = self.nlp(document["text"])
            result.append(
                {
                    "text_system_id": document["system_id"],
                    "text": document["text"],
                    "ner": ner,
                    "words": self.__connect_entities(ner, document["text"]),
                }
            )

        for res in result:
            for w in res["words"]:
                self.__looking_for_words(w, res["text"])

        self.__make_full_word(result)
        result = self.__remove_leti(result)
        for el in result:
            el["words"] = self.__check_duplicates(el["words"])

        for i in result:
            for j in i["words"]:
                j["entity"] = self.__choose_one_entity_type(j["entity"])

        return result
