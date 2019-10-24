# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.toktok import ToktokTokenizer
import random
import numpy as np
from sklearn.svm import LinearSVC
#import utils
import os
from utils import read_config, load_pickle, save_pickle


class SubSolver(object):
    """
    Классификатор между заданиями.
    Работает на Tfidf векторах и мультиклассовом SVM.

    Parameters
    ----------
    seed : int, optional (default=42)
        Random seed.
    ngram_range : tuple, optional uple (min_n, max_n) (default=(1, 3))
        Used forTfidfVectorizer.
        he lower and upper boundary of the range of n-values for different n-grams to be extracted.
        All values of n such that min_n <= n <= max_n will be used.
    num_tasks : int, optional (default=27)
        Count of all tasks.

    Examples
    --------
    >>> # Basic usage
    >>> from solvers import classifier
    >>> import json
    >>> from utils import read_config
    >>> clf = classifier.Solver()
    >>> tasks = []
    >>> dir_path = "data/"
    >>> for file_name in os.listdir(dir_path):
    >>>     if file_name.endswith(".json"):
    >>>         data = read_config(os.path.join(dir_path, file_name))
    >>>         tasks.append(data)
    >>> clf = solver.fit(tasks)
    >>> # Predict for last file in dir
    >>> numbers_of_tasks = clf.predict(read_config(os.path.join(dir_path, file_name)))
    >>> numbers_of_tasks
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 12, 13, 14, 15, 16, 17,
       18, 19, 17, 21, 22, 23, 24, 25, 26, 24])
    >>> # Save classifier
    >>> clf.save("clf.pickle")
    >>> # Load classifier
    >>> clf.load("clf.pickle")
    """

    def __init__(self, t='text', seed=42, ngram_range=(1, 3)):
        self.seed = seed
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        self.clf = LinearSVC(multi_class='ovr')
        self.init_seed()
        self.word_tokenizer = ToktokTokenizer()
        self.type = t

    def init_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def fit(self, tasks):
        texts = []
        classes = []
        for data in tasks:
            for task in data:
                if task['question']['type'] == self.type:
                    idx = int(task["id"])
                    text = " ".join(self.word_tokenizer.tokenize(task['text']))
                    texts.append(text)
                    classes.append(idx)
        classes = np.array(classes)
        self.classes = np.unique(classes)
        if len(self.classes) > 1:
            vectors = self.vectorizer.fit_transform(texts)
            self.clf.fit(vectors, classes)
        return self

    def predict_one(self, task):
        if len(self.classes) == 1:
            return self.classes[0]
        text = " ".join(self.word_tokenizer.tokenize(task['text']))
        return self.clf.predict(self.vectorizer.transform([text]))[0]

    def fit_from_dir(self, dir_path):
        tasks = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".json"):
                data = read_config(os.path.join(dir_path, file_name))
                tasks.append(data)
        return self.fit(tasks)

    @classmethod
    def load(cls, path):
        return load_pickle(path)

    def save(self, path):
        save_pickle(self, path)


class Solver:
    def __init__(self, seed=42, ngram_range=(1, 3)):
        self.seed = seed
        self.ngram_range = ngram_range
        self.clfs = {t: SubSolver(t, seed, ngram_range) for t in ('text', 'choice', 'multiple_choice', 'matching')}
    
    def fit(self, tasks):
        for el in self.clfs.values():
            el.fit(tasks)
    
    def predict(self, tasks):
        res = []
        for task in tasks:
            res.append(self.clfs[task['question']['type']].predict_one(task))
        return res