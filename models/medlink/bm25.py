"""
Reference: https://github.com/dorianbrown/rank_bm25
"""

import math

import numpy as np


class BM25:
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.corpus_ids = list(corpus.keys())
        self.corpus_list = [corpus[id] for id in self.corpus_ids]
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        nd = self._initialize(self.corpus_list)
        self._calc_idf(nd)

    def _initialize(self, corpus_list):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus_list:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()


class BM25Okapi(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus)

    def _calc_idf(self, nd):
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query, random=False):
        if not random:
            score = np.zeros(self.corpus_size)
            doc_len = np.array(self.doc_len)
            query = query.split(' ')
            for q in query:
                q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
                score += (self.idf.get(q) or 0) * \
                         (q_freq * (self.k1 + 1) / (q_freq + self.k1 * (
                                 1 - self.b + self.b * doc_len / self.avgdl)))
        else:
            score = np.random.rand(self.corpus_size)
        score = score.tolist()
        score = {self.corpus_ids[idx]: s for idx, s in enumerate(score)}
        return score
