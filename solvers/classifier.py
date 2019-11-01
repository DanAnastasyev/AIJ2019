# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from solvers.utils import ToktokTokenizer
import random
import numpy as np
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier
import catboost
from sklearn.model_selection import train_test_split
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
        self.vectorizer2 = TfidfVectorizer(ngram_range=ngram_range)
        self.clf = LinearSVC(multi_class="ovr")
        self.init_seed()
        self.word_tokenizer = ToktokTokenizer()
        self.type = t

    def init_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def convert_to_text(self, task):
        #return task['text']
        text = self.word_tokenizer.tokenize(task['text'])
        if self.type in ["choice", "multiple_choice"]:
            choice_type = [t for t in task['question']['choices'][0].keys() if t != 'id'][0]
            text.append(choice_type)
            for el in task['question']['choices']:
                text += self.word_tokenizer.tokenize(el[choice_type])
        text = ' '.join(text)
        return text
    
    def fit(self, tasks):
        texts = []
        classes = []
        for data in tasks:
            for task in data:
                if task['question']['type'] == self.type:
                    idx = int(task["id"])
                    if idx in range(17, 21):
                        idx = 17
                    texts.append(self.convert_to_text(task))
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
        text = self.convert_to_text(task)
        return int(
            self.clf.predict(self.vectorizer.transform([text])).ravel()[0])

    def fit_from_dir(self, dir_path):
        tasks = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".json"):
                data = read_config(os.path.join(dir_path, file_name))
                tasks.append(data)
        tasks = [task for task in tasks if 'hint' not in task]
        return self.fit(tasks)

    def load(self, d):
        self.vectorizer = d['vec']
        self.clf = d['clf']
        self.classes = d['classes']

    def save(self):
        return {
            "vec": self.vectorizer,
            "clf": self.clf,
            "classes": self.classes
        }


class Solver:
    def __init__(self, seed=42, ngram_range=(1, 3)):
        self.seed = seed
        self.ngram_range = ngram_range
        self.clfs = {
            t: SubSolver(t, seed, ngram_range)
            for t in ('text', 'choice', 'multiple_choice', 'matching')
        }

    def fit(self, tasks):
        for k, el in self.clfs.items():
            print('Fitting classifier for ' + k)
            el.fit(tasks)

    def predict(self, tasks):
        res = []
        for task in tasks:
            res.append(self.clfs[task['question']['type']].predict_one(task))
        return res

    def load(self, path):
        d = load_pickle(path)
        for k, subsolver in self.clfs.items():
            subsolver.load(d[k])

    def save(self, path):
        save_pickle(
            {k: subsolver.save()
             for k, subsolver in self.clfs.items()}, path)
