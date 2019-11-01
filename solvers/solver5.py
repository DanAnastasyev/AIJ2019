from ufal.udpipe import Model, Pipeline
from difflib import get_close_matches
from string import punctuation
import pickle
import pymorphy2
import re
import random
from solvers.utils import ToktokTokenizer, BertEmbedder
from sklearn.metrics.pairwise import cosine_similarity

class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.morph = pymorphy2.MorphAnalyzer()
        self.seed = seed
        self.init_seed()
        self.toktok = ToktokTokenizer()
        self.paronyms = self.get_paronyms()

    def init_seed(self):
        return random.seed(self.seed)

    def get_paronyms(self):
        paronyms = []
        with open('data/paronyms.csv', 'r', encoding='utf-8') as in_file:
            for line in in_file.readlines():
                pair = line.strip(punctuation).strip().split('\t')
                paronyms.append(pair)
        return paronyms

    def lemmatize(self, token):
        token_all = self.morph.parse(token.lower().rstrip('.,/;!:?'))[0]
        lemma = token_all.normal_form
        return lemma

    def find_closest_paronym(self, par):
        paronyms = set()
        for par1, par2 in self.paronyms:
            paronyms.add(par1)
            paronyms.add(par2)
        try:
            closest = get_close_matches(par, list(paronyms))[0]
        except IndexError:
            closest = None
        return closest

    def check_pair(self, token_norm):
        paronym = None
        for p1, p2 in self.paronyms:
            if token_norm == p1:
                paronym = p2
                break
            if token_norm == p2:
                paronym = p1
                break
        return paronym

    def find_paronyms(self, token):
        token_all = self.morph.parse(token.lower().rstrip('.,/;!:?'))[0]
        token_norm = token_all.normal_form
        paronym = self.check_pair(token_norm)

        if paronym is None:
            paronym_close = self.find_closest_paronym(token_norm)
            paronym = self.check_pair(paronym_close)

        if paronym is not None:
            paronym_parse = self.morph.parse(paronym)[0]
            try:
                str_grammar = str(token_all.tag).split()[1]
            except IndexError:
                str_grammar = str(token_all.tag)

            gr = set(str_grammar.replace("Qual ", "").replace(' ',',').split(','))
            try:
                final_paronym = paronym_parse.inflect(gr).word
            except AttributeError:
                final_paronym = paronym
        else:
            final_paronym = ''
        return final_paronym

    def predict(self, task):
        return self.predict_from_model(task)

    def fit(self, tasks):
        pass

    def load(self, path="data/models/solver5.pkl"):
        pass

    def save(self, path="data/models/solver5.pkl"):
        pass

    def get_score(self, a, b, paronym):
        return self.fill_mask(a, b, paronym.lower())
        return cosine_similarity(self.sentence_embedding([sent])[0].reshape(1, -1),
                                 self.sentence_embedding([paronym.lower()])[0].reshape(1, -1))[0][0]

    def predict_from_model(self, task):
        description = task["text"].replace('НЕВЕРНО ', "неверно ")
        sents = []
        for line in self.toktok.sentenize(description):
            line_tok = self.toktok.tokenize(line)
            for idx, token in enumerate(line_tok):
                if token.isupper() and len(token) > 2: # get CAPS paronyms
                    second_pair = self.find_paronyms(token)
                    line_before = ' '.join(line_tok[:idx])
                    line_after = ' '.join(line_tok[idx + 1:])
                    if second_pair != '':
                        score = self.get_score(line_before, line_after, token) - self.get_score(line_before, line_after, second_pair)
                        sents.append((score, token, second_pair))
        sents.sort()
        return sents[0][2].strip(punctuation+'\n')