# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Vit Novotny <witiko@mail.muni.cz>, lhy<lhy_in_blcu@126.com>
@description:
Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

This module provides classes that deal with sentence similarities from mean term vector.
Adjust the gensim similarities Index to compute sentence similarities.
"""

import os
from typing import List, Union, Dict

import jieba
import jieba.analyse
import jieba.posseg
import numpy as np
from loguru import logger

from tqdm import tqdm

from utils.distance import string_hash, hamming_distance, longest_match_size
from utils.rank_bm25 import BM25Okapi
from utils.tfidf import TFIDF, load_stopwords, default_stopwords_file
from utils.util import cos_sim, semantic_search

pwd_path = os.path.abspath(os.path.dirname(__file__))

from typing import List, Union, Dict


class SimilarityABC:
    """
    Interface for similarity compute and search.

    In all instances, there is a corpus against which we want to perform the similarity search.
    For each similarity search, the input is a document or a corpus, and the output are the similarities
    to individual corpus documents.
    """

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute similarity between two texts.
        :param a: list of str or str
        :param b: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: Dict[str(query_id), str(query_text)] or List[str] or str
        :param topn: int
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def search(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: Dict[str(query_id), str(query_text)] or List[str] or str
        :param topn: int
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        return self.most_similar(queries, topn=topn)
class SimHashSimilarity(SimilarityABC):
    """
    Compute SimHash similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: Union[List[str], Dict[str, str]] = None):
        self.corpus = {}

        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SimHash"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str or dict of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_texts = list(corpus_new.values())
        corpus_embeddings = []
        for sentence in tqdm(corpus_texts, desc="Computing corpus SimHash"):
            corpus_embeddings.append(self.simhash(sentence))
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def simhash(self, sentence: str):
        """
        Compute SimHash for a given text.
        :param sentence: str
        :return: hash code
        """
        seg = jieba.cut(sentence)
        key_word = jieba.analyse.extract_tags('|'.join(seg), topK=None, withWeight=True, allowPOS=())
        # 先按照权重排序，再按照词排序
        key_list = []
        for feature, weight in key_word:
            weight = int(weight * 20)
            temp = []
            for f in string_hash(feature):
                if f == '1':
                    temp.append(weight)
                else:
                    temp.append(-weight)
            key_list.append(temp)
        content_list = np.sum(np.array(key_list), axis=0)
        # 编码读不出来
        if len(key_list) == 0:
            return '00'
        hash_code = ''
        for c in content_list:
            if c > 0:
                hash_code = hash_code + '1'
            else:
                hash_code = hash_code + '0'
        return hash_code

    def _sim_score(self, seq1, seq2):
        """Convert hamming distance to similarity score."""
        # 将距离转化为相似度
        score = 0.0
        if len(seq1) > 2 and len(seq2) > 2:
            score = 1 - hamming_distance(seq1, seq2, normalize=True)
        return score

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute hamming similarity between two sentences.

        Parameters
        ----------
        a : str or list of str
        b : str or list of str

        Returns
        -------
        list of float
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")
        seqs1 = [self.simhash(text) for text in a]
        seqs2 = [self.simhash(text) for text in b]
        scores = [self._sim_score(seq1, seq2) for seq1, seq2 in zip(seqs1, seqs2)]
        return scores

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute hamming distance between two sentences.

        Parameters
        ----------
        a : str or list of str
        b : str or list of str

        Returns
        -------
        list of float
        """
        sim_scores = self.similarity(a, b)
        return [1 - score for score in sim_scores]

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: list of str or str
        :param topn: int
        :return: list of list tuples (corpus_id, corpus_text, similarity_score)
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            query_emb = self.simhash(query)
            for (corpus_id, doc), doc_emb in zip(self.corpus.items(), self.corpus_embeddings):
                score = self._sim_score(query_emb, doc_emb)
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result


class TfidfSimilarity(SimilarityABC):
    """
    Compute TFIDF similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: Union[List[str], Dict[str, str]] = None):
        super().__init__()
        self.corpus = {}

        self.corpus_embeddings = []
        self.tfidf = TFIDF()
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: Tfidf"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_texts = list(corpus_new.values())
        corpus_embeddings = []
        for sentence in tqdm(corpus_texts, desc="Computing corpus TFIDF"):
            corpus_embeddings.append(self.tfidf.get_tfidf(sentence))
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute cosine similarity score between two sentences.
        :param a:
        :param b:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        features1 = [self.tfidf.get_tfidf(text) for text in a]
        features2 = [self.tfidf.get_tfidf(text) for text in b]
        return cos_sim(np.array(features1), np.array(features2))

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(a, b)

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())

        queries_embeddings = np.array([self.tfidf.get_tfidf(query) for query in queries_texts], dtype=np.float32)
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn)
        for idx, hits in enumerate(all_hits):
            for hit in hits[0:topn]:
                result[queries_ids_map[idx]][hit['corpus_id']] = hit['score']

        return result


class BM25Similarity(SimilarityABC):
    """
    Compute BM25OKapi similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: Union[List[str], Dict[str, str]] = None):
        super().__init__()
        self.corpus = {}

        self.bm25 = None
        self.default_stopwords = load_stopwords(default_stopwords_file)
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: BM25"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str or dict
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_texts = list(corpus_new.values())
        corpus_seg = [jieba.lcut(d) for d in corpus_texts]
        corpus_seg = [[w for w in doc if (w.strip().lower() not in self.default_stopwords) and
                       len(w.strip()) > 0] for doc in corpus_seg]
        self.bm25 = BM25Okapi(corpus_seg)
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn=10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: input query
        :param topn: int
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}}
        """
        if not self.bm25:
            raise ValueError("BM25 model is not initialized. Please add_corpus first, eg. `add_corpus(corpus)`")
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        for qid, query in queries.items():
            tokens = jieba.lcut(query)
            scores = self.bm25.get_scores(tokens)

            q_res = [{'corpus_id': corpus_id, 'score': score} for corpus_id, score in enumerate(scores)]
            q_res = sorted(q_res, key=lambda x: x['score'], reverse=True)[:topn]
            for res in q_res:
                corpus_id = res['corpus_id']
                result[qid][corpus_id] = res['score']

        return result

class SameCharsSimilarity(SimilarityABC):
    """
    Compute text chars similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    不考虑文本字符位置顺序，基于相同字符数占比计算相似度
    """

    def __init__(self, corpus: Union[List[str], Dict[str, str]] = None):
        super().__init__()
        self.corpus = {}

        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SameChars"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start add new docs: {len(corpus_new)}")
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute Chars similarity between two texts.
        :param a:
        :param b:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")

        def calc_pair_sim(sentence1, sentence2):
            if not sentence1 or not sentence2:
                return 0.0
            same = set(sentence1) & set(sentence2)
            similarity_score = max(len(same) / len(set(sentence1)), len(same) / len(set(sentence2)))
            return similarity_score

        return [calc_pair_sim(sentence1, sentence2) for sentence1, sentence2 in zip(a, b)]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return [1 - s for s in self.similarity(a, b)]

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            for corpus_id, doc in self.corpus.items():
                score = self.similarity(query, doc)[0]
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result


class SequenceMatcherSimilarity(SimilarityABC):
    """
    Compute text sequence matcher similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    考虑文本字符位置顺序，基于最长公共子串占比计算相似度
    """

    def __init__(self, corpus: Union[List[str], Dict[str, str]] = None):
        super().__init__()
        self.corpus = {}

        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SequenceMatcher"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start add new docs: {len(corpus_new)}")
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]],
                   min_same_len: int = 70, min_same_len_score: float = 0.9):
        """
        Compute Chars similarity between two texts.
        :param a:
        :param b:
        :param min_same_len:
        :param min_same_len_score:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")

        def calc_pair_sim(sentence1, sentence2):
            if not sentence1 or not sentence2:
                return 0.0
            same_size = longest_match_size(sentence1, sentence2)
            same_score = min_same_len_score if same_size > min_same_len else 0.0
            similarity_score = max(same_size / len(sentence1), same_size / len(sentence2), same_score)
            return similarity_score

        return [calc_pair_sim(sentence1, sentence2) for sentence1, sentence2 in zip(a, b)]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return [1 - s for s in self.similarity(a, b)]

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            for corpus_id, doc in self.corpus.items():
                score = self.similarity(query, doc)[0]
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result