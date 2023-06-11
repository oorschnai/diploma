import re
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
from langdetect import detect
import numpy as np
from thefuzz import fuzz


class SynonymsChecker:
    def __init__(self, model=None, train_data=None):
        self.keywords_wiki = []
        self.keywords_kbir = []
        self.keywords = []
        self.filename_model = model
        self.train_data = train_data

    def __load_model(self):
        w2v = None
        if self.filename_model != None:
            w2v = Word2Vec.load(self.filename_model)

        if self.train_data != None:
            w2v = self.__train_model()
        return w2v

    def __preprocessing_texts(self):
        text = ""
        for document in self.train_data:
            text += document["text"]
        sentences = re.split(r"\.+\s", text.lower())
        sentences_word_by_word = []
        for i in sentences:
            i = re.sub(r"\xa0", " ", i)
            i = re.sub(r"[^а-яёa-z -+.\/0-9]", "", i)
            i = re.sub(r"\s+", " ", i)
            sentences_word_by_word.append(i.split(" "))
        return sentences_word_by_word

    def __preprocessing_keywords(self):
        arr = []
        dict_wiki = {}
        for i in self.keywords_wiki:
            ii = re.sub("[^а-яёa-z -+.\/0-9]", "", i.lower())
            ii = re.sub(r"\xa0", " ", ii)
            ii = re.sub(r"\s+", " ", ii)
            if ii in dict_wiki.keys():
                dict_wiki[ii].append(i)
            else:
                dict_wiki[ii] = [i]
            arr.append(ii)

        dict_kbir = {}
        for i in self.keywords_kbir:
            ii = re.sub("[^а-яёa-z -+.\/0-9]", "", i.lower())
            ii = re.sub(r"\xa0", " ", ii)
            ii = re.sub(r"\s+", " ", ii)
            if ii in dict_kbir.keys():
                dict_kbir[ii].append(i)
            else:
                dict_kbir[ii] = [i]
            arr.append(ii)

        # kw = list(set([] + arr))
        self.keywords = list(set([] + arr))
        return dict_wiki, dict_kbir

    def __train_model(self):
        sentences_word_by_word = self.__preprocessing_texts()
        w2v_model = Word2Vec(
            min_count=1,
            window=5,
            negative=0,
            alpha=0.05,
            min_alpha=0.001,
            sample=0,
            sg=1,
            hs=1,
            epochs=10,
        )
        w2v_model.build_vocab(sentences_word_by_word)
        w2v_model.train(
            sentences_word_by_word,
            total_examples=w2v_model.corpus_count,
            epochs=10,
            report_delay=1,
        )
        w2v_model.init_sims(replace=True)
        return w2v_model

    def __compute_vectors_for_phrases(self, w2v_model):
        X = []
        dictionary = {}
        notInVocab = []

        for phrase in self.keywords:
            vec = 0
            flag = 0
            for word in phrase.split(" "):
                if not word in w2v_model.wv.key_to_index.keys():
                    flag = 1
                    notInVocab.append(word)
                    break
                vec += w2v_model.wv[word]
            vec /= len(phrase.split(" "))
            if flag:
                continue
            string = " ".join(str(v) for v in vec)
            if string in dictionary.keys():
                dictionary[string].append(phrase)
            else:
                dictionary[string] = [phrase]
                X.append(vec)
        return X, dictionary, notInVocab

    def __get_words_by_vectors(self, X, dictionary, clustering):
        synonyms_array = []
        vectors = []
        number_of_clusters = clustering.labels_.max()
        for i in range(number_of_clusters):
            cluster = np.where(clustering.labels_ == i)[0]
            synonyms = []
            vecs = []
            for vec in cluster:
                synonyms.extend(dictionary[" ".join([str(v) for v in X[vec]])])
                vecs.append(X[vec])
            vectors.append(vecs)
            synonyms_array.append(synonyms)
        return synonyms_array, vectors

    def __chose_one_word(self, synonyms_array, vectors, dict_wiki, dict_kbir):
        lang_en = [
            "en",
            "de",
            "so",
            "ro",
            "et",
            "sv",
            "tl",
            "id",
            "sw",
            "nl",
            "cy",
            "pl",
            "pt",
        ]
        clusters = []
        for i in range(len(synonyms_array)):
            clusters.append({"id": i, "cluster": synonyms_array[i]})
            if (len(synonyms_array[i])) > 2:
                dist = []
                for v in vectors[i]:
                    d = []
                    for v2 in vectors[i]:
                        if np.all(v == v2):
                            continue
                        d.append(sum([(v[j] - v2[j]) ** 2 for j in range(len(v))]))
                    dist.append(d)

                dist = [sum(d) ** 0.5 for d in dist]
                ind = 0
                minimum = 10000000
                for k in range(len(dist)):
                    if dist[k] < minimum:
                        minimum = dist[k]
                        ind = 0 + k

                clusters[i]["choosen"] = synonyms_array[i][ind]

            else:
                # 2 words in cluster
                # wiki > kbir
                if synonyms_array[i][0] in dict_wiki.keys():
                    clusters[i]["choosen"] = synonyms_array[i][0]
                    continue
                elif synonyms_array[i][1] in dict_wiki.keys():
                    clusters[i]["choosen"] = synonyms_array[i][1]
                    continue

                # en > ru
                lang0 = detect(synonyms_array[i][0])
                lang1 = detect(synonyms_array[i][1])
                if lang0 in lang_en and lang1 not in lang_en:
                    clusters[i]["choosen"] = synonyms_array[i][0]
                    continue
                elif lang0 not in lang_en and lang1 in lang_en:
                    clusters[i]["choosen"] = synonyms_array[i][1]
                    continue

                # len
                l0 = len(synonyms_array[i][0])
                l1 = len(synonyms_array[i][1])
                if l0 > l1:
                    clusters[i]["choosen"] = synonyms_array[i][0]
                    continue
                elif l1 > l0:
                    clusters[i]["choosen"] = synonyms_array[i][1]
                    continue
                # else
                clusters[i]["choosen"] = synonyms_array[i][0]

            return clusters

    def __choose_best(self, key, words):
        max_score = 0
        res = ""
        for word in words:
            ratio = fuzz.ratio(key, word)
            if ratio > max_score:
                max_score = ratio
                res = word
        return res

    def __form_list_of_words(self, dict_wiki, dict_kbir, clusters):
        wiki_words = set(dict_wiki.keys())
        kbir_words = set(dict_kbir.keys())
        kbir_words.difference_update(wiki_words)

        if clusters:
            hidden_words = []
            for c in clusters:
                hidden_words.extend(c["cluster"])
                hidden_words.remove(c["choosen"])

            hidden_words = set(hidden_words)
            wiki_words.difference_update(hidden_words)
            kbir_words.difference_update(hidden_words)

        kbir_words = list(kbir_words)
        wiki_words = list(wiki_words)

        kbir_words_res = []
        for word in kbir_words:
            if len(dict_kbir[word]) > 1:
                kbir_words_res.append(self.__choose_best(word, dict_kbir[word]))
            else:
                kbir_words_res.append(dict_kbir[word][0])

        wiki_words_res = []
        for word in wiki_words:
            if len(dict_wiki[word]) > 1:
                wiki_words_res.append(self.__choose_best(word, dict_wiki[word]))
            else:
                wiki_words_res.append(dict_wiki[word][0])

        return wiki_words_res, kbir_words_res

    def get_synonyms(self, keywords):
        print("Synonyms Checker start")
        self.keywords_kbir = []
        self.keywords_wiki = []
        if type(keywords[0]) == list:
            if "synonyms" in keywords[0][0].keys():
                for kw in keywords:
                    self.keywords_wiki.extend(kw["synonyms"]["wiki_words"])
                    self.keywords_kbir.extend(kw["synonyms"]["kbir_words"])
                eps = 0.1
        else:
            for kw in keywords:
                self.keywords_wiki.extend(kw["wiki"]["normalized"])
                self.keywords_kbir.extend(kw["kbir"]["normalized"])
            eps = 0.3
        w2v_model = self.__load_model()
        dict_wiki, dict_kbir = self.__preprocessing_keywords()
        X, dictionary, notInVocab = self.__compute_vectors_for_phrases(w2v_model)
        clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(X)
        synonyms_array, vectors = self.__get_words_by_vectors(X, dictionary, clustering)
        clusters = self.__chose_one_word(synonyms_array, vectors, dict_wiki, dict_kbir)

        wiki_words, kbir_words = self.__form_list_of_words(
            dict_wiki, dict_kbir, clusters
        )
        result = {
            "clusters": clusters,
            "wiki_words": wiki_words,
            "kbir_words": kbir_words,
        }

        return result
