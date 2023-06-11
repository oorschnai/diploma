from transformers import GPT2LMHeadModel, GPT2Tokenizer
from thefuzz import fuzz
import numpy as np
from langdetect import detect

template = """магазинов приложений -> магазины приложений
человеко-машинного интерфейса -> человеко-машинный интерфейс
принципах взаимодействия нейронов -> принципы взаимодействия нейронов
отладками приложений -> отладка приложений
содержанием дисциплины -> содержание дисциплины
посвящен изобретениям -> посвящен изобретениям
it-проектом -> it-проект
методу Гаусса -> метод Гаусса
методом -> метод
манекен -> манекен
телефоном -> телефон
оценкой выше -> оценка выше
исследованиями посвященными использованию -> исследования посвященные использованию
методов обнаружения компьютерных атак -> методы обнаружения компьютерных атак
спорного -> спорный
сферах it -> сфера it
системных блоков -> системные блоки
старости -> старость
производственная практика -> производственная практика
шаблонных действий -> шаблонные действия
методами сбора эмпирических данных -> методы сбора эмпирических данных
потраченного времени -> потраченное время
российского -> российский
смелого -> смелый
рукой -> рука
оценками успеваемости -> оценки успеваемости
методами извлечения -> методы извлечения
старого Java -> старый Java
собранного хвороста -> собранный хворост
оценку -> оценка
запутанного дела -> запутанное дело
практических навыков разработки -> практические навыки разработки
"""


class Normalization:
    def __init__(self):
        self.kw_before = []
        self.kw_after = []
        self.model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name, low_cpu_mem_usage=True
        )
        self.model.eval()
        # self.generator = pipeline('text-generation', model=self.model_name)
        self.templ_lines_count = template.count("\n")

    def __set_kw(self, kw):
        self.kw_before = kw

    def __get_model_results(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        lang = ["en", "de", "so", "ro", "et", "sv", "tl", "id", "sw", "nl", "cy", "pl"]
        for word in self.kw_before:
            if detect(word) in lang:
                self.kw_after.append(word)
                continue
            text = template + word + """ -> """
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            out = self.model.generate(
                input_ids, max_new_tokens=2, pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text_old = list(map(self.tokenizer.decode, out))[0]
            flag = 1
            counter = 0
            while flag and counter < 5:
                counter += 1
                input_ids = self.tokenizer.encode(
                    generated_text_old, return_tensors="pt"
                )
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated_text = list(map(self.tokenizer.decode, out))[0]
                differences = generated_text.replace(generated_text_old, "")
                newl = generated_text.count("\n")
                if len(differences) < 2 or newl > self.templ_lines_count + 1:
                    flag = 0
                generated_text_old = "" + generated_text

            self.kw_after.append(generated_text.replace(text, ""))

    def __processing(self):
        symbols_blacklist = ["->", "-", ">", "`", "(", ")", ",", "."]
        words = []
        for ind in range(len(self.kw_after)):
            i = self.kw_after[ind]
            word = ""
            if i[0] == "\n":
                if "\n" in i[1:]:
                    word = i[1 : i[1:].index("\n") + 1]
                else:
                    word = i[1:]
            else:
                if "\n" in i:
                    word = i[: i.index("\n") + 1]
                else:
                    word = i

            w_before = self.kw_before[ind]
            for symb in symbols_blacklist:
                if symb not in w_before and symb in word:
                    word = word.replace(symb, "")

            words.append(word.replace("\n", ""))

        self.kw_after = words.copy()

    def __remove_spaces(self):
        for i in range(len(self.kw_before)):
            if not type(self.kw_after) is float:
                while len(self.kw_after[i]) > 0 and self.kw_after[i][0] == " ":
                    self.kw_after[i] = self.kw_after[i][1:]
                while len(self.kw_after[i]) > 0 and self.kw_after[i][::-1][0] == " ":
                    self.kw_after[i] = self.kw_after[i][0:-1]

            while len(self.kw_before[i]) > 0 and self.kw_before[i][0] == " ":
                self.kw_before[i] = self.kw_before[i][1:]
            while len(self.kw_before[i]) > 0 and self.kw_before[i][::-1][0] == " ":
                self.kw_before[i] = self.kw_before[i][0:-1]

    def __get_fuzz_ratio(self):
        self.__remove_spaces()
        fuzzed = []
        for i in range(len(self.kw_after)):
            if type(self.kw_after[i]) == float and np.isnan(self.kw_after[i]):
                fuzzed.append(0)
            else:
                fuzzed.append(fuzz.ratio(self.kw_before[i], self.kw_after[i]))
        return fuzzed

    def __select_words(self):
        ratio = self.__get_fuzz_ratio()
        for i in range(len(self.kw_after)):
            if ratio[i] <= 91:
                self.kw_after[i] = self.kw_before[i]

    def normalize(self, kw):
        print("start norm")
        if len(kw) == 0:
            return []
        self.kw_before = []
        self.kw_after = []
        self.__set_kw(kw)
        self.__get_model_results()
        self.__processing()
        self.__select_words()
        return self.kw_after
