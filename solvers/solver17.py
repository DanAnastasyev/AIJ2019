import re
import pymorphy2
import stanfordnlp
import joblib

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from solvers.utils import AbstractSolver


def _iterate_subsets(elements):
    if not elements:
        yield []
        return
    for subset in _iterate_subsets(elements[1:]):
        element = elements[0]
        yield [] + subset
        yield [element] + subset


class Solver(AbstractSolver):
    def __init__(self, seed=42, train_size=0.85, models_dir='data/stanfordnlp_resources/'):
        self.has_model = False
        self.is_train_task = False
        self.morph = pymorphy2.MorphAnalyzer()
        self.pos2n = {None: 0}
        self.n2pos = [None, ]
        self.train_size = train_size
        self.seed = seed
        self._pipeline = stanfordnlp.Pipeline(lang='ru', models_dir=models_dir)

        self.model = CatBoostClassifier(loss_function="Logloss",
                                   eval_metric='Accuracy',
                                   use_best_model=True, random_seed=self.seed)
        super().__init__(seed)

    @staticmethod
    def _convert_sentence(sentence):
        result_sent, positions = '', []
        prev_match_end = 0
        for match in re.finditer('\(\d+\)', sentence):
            result_sent += sentence[prev_match_end: match.start()].strip() + ' '
            positions.append(len(result_sent) - 1)
            prev_match_end = match.end()

        result_sent += sentence[prev_match_end:].strip()
        return result_sent, positions

    @staticmethod
    def _get_sentence(task):
        task_text = ''
        for sent in task['text'].split('.'):
            sent = sent.strip() + '. '
            if re.search('\(\d+\)', sent):
                task_text += sent
        print(task_text.strip())
        return task_text.strip()

    def _find_best_positions(self, sentence, positions):
        best_positions, best_score = ["1"], -100
        for subset in _iterate_subsets(positions):
            sent_with_punct = ''
            prev_position = 0
            for position in subset:
                sent_with_punct += sentence[prev_position: position] + ','
                prev_position = position
            sent_with_punct += sentence[prev_position:]

            doc = self._pipeline(sent_with_punct)
            score = doc.conll_file.sents[0][0][-1]

            if score > best_score:
                best_positions = subset
                best_score = score

        return [str(positions.index(position) + 1) for position in best_positions]

    def get_placeholder(self, token):
        if len(token) < 5 and token[0] == '(' and token[-1] == ')':
            return token[1:-1]
        return ''

    def save(self, path="data/models/solver17.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="data/models/solver17.pkl"):
        self.model = joblib.load(path)

    def get_target(self, task):
        if 'solution' not in task:
            return []
        y_true = task['solution']['correct_variants'] if 'correct_variants' in task['solution'] \
            else [task['solution']['correct']]
        return list(y_true[0])

    def clear_token(self, token):
        for char in '?.!/;:':
            token = token.replace(char, '')
        return token

    def get_feat(self, token):
        if self.get_placeholder(token):
            return 'PHDR'
        else:
            p = self.morph.parse(self.clear_token(token))[0]
            return str(p.tag.POS)

    def encode_feats(self, feats):
        res = []
        for feat in feats:
            if feat not in self.pos2n and self.is_train_task:
                self.pos2n[feat] = len(self.pos2n)
                self.n2pos.append(feat)
            elif feat not in self.pos2n:
                feat = None
            res.append(self.pos2n[feat])
        return res

    def correct_spaces(self, text):
        text = re.sub(r'(\(\d\))', r' \1 ', text)
        text = re.sub(r'  *', r' ', text)
        return text

    def parse_task(self, task):
        feat_ids = [-3, -2, -1, 1, 2, 3]
        tokens = self.correct_spaces(task['text']).split()
        targets = self.get_target(task)
        X, y = [], []
        for i, token in enumerate(tokens):
            placeholder = self.get_placeholder(token)
            if not placeholder:
                continue
            if placeholder in targets:
                y.append(1)
            else:
                y.append(0)
            feats = []
            for feat_idx in feat_ids:
                if i + feat_idx < 0 or i + feat_idx >= len(tokens):
                    feats.append('PAD')
                else:
                    feats.append(self.get_feat(tokens[i + feat_idx]))
            X.append(self.encode_feats(feats))
        return X, y

    def fit(self, tasks):
        self.is_train_task = True
        X, y = [], []
        for task in tasks:
            task_x, task_y = self.parse_task(task)
            X += task_x
            y += task_y
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, y, shuffle=True,
                                                          train_size=self.train_size, random_state=self.seed)
        cat_features = [0, 1, 2, 3, 4, 5]
        self.model = self.model.fit(X_train, Y_train, cat_features, eval_set=(X_dev, Y_dev))
        self.has_model = True

    def _predict_with_parser(self, task):
        sentence, positions = self._convert_sentence(self._get_sentence(task))

        if len(positions) > 5:
            return

        # assert task['question']['type'] == 'multiple_choice'
        # assert len(task['question']['choices']) == len(positions)
        # assert [int(choice['id']) == i + 1 for i, choice in enumerate(task['question']['choices'])]

        return self._find_best_positions(sentence, positions)

    def predict_from_model(self, task):
        pred = self._predict_with_parser(task)
        if pred:
            return pred

        self.is_train_task = False
        X, y = self.parse_task(task)
        pred = self.model.predict(X)
        pred = [str(i+1) for i, p in enumerate(pred) if p >= 0.5]
        return pred if pred else ["1"]
