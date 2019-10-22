import re
import os
import random
from string import punctuation


class Solver(object):

    def __init__(self, seed=42, data_path='data/'):
        self.is_train_task = False
        self.seed = seed
        self.init_seed()
        with open(os.path.join(data_path, 'agi_stress.txt'), 'r', encoding='utf8') as f:
            self.stress = set()
            for line in f:
                line = line.strip()
                self.stress.add(line)
                if len(line.split()) > 1:
                    for token in line.split():
                        if token.lower() != token:
                            self.stress.add(token)
        with open(os.path.join(data_path, 'zaliznyak_preprocessed.txt'), 'r', encoding='utf8') as f:
            for line in f:
                self.stress.add(line.strip())

    def init_seed(self):
        random.seed(self.seed)

    def predict(self, task):
        return self.predict_from_model(task)

    def is_in_dictionary(self, text):
        if text in self.stress:
            return True
        for token in text.split():
            if token.lower() != token and token in self.stress:
                return True
        return False

    def compare_text_with_variants(self, variants, task_type='incorrect'):
        dict_variants = []
        if task_type == 'incorrect':
            for variant in variants:
                if not self.is_in_dictionary(variant):
                    dict_variants.append(variant)
        else:
            for variant in variants:
                if self.is_in_dictionary(variant):
                    dict_variants.append(variant)
        if not variants:
            return ''
        print(dict_variants)
        if dict_variants:
            result = random.choice(dict_variants)
        else:
            result = random.choice(variants)

        for token in result.split():
            if token.lower() != token:
                return token.lower()
        return result.lower().strip(punctuation)

    def process_task(self, task):
        task_text = re.split(r'\n', task['text'])
        if 'Выпишите' in task_text[-1]:
            task = task_text[0] + task_text[-1]
            variants = [text.strip() for text in task_text[1:-1] if text.strip() and text.lower() != text]
        else:
            task = task_text[0]
            variants = [text.strip() for text in task_text[1:] if text.strip() and text.lower() != text]
        if 'неверно' in task.lower():
            task_type = 'incorrect'
        else:
            task_type = 'correct'
        return task_type, task, variants

    def fit(self, tasks):
        pass

    def load(self, path="data/models/solver4.pkl"):
        pass

    def save(self, path="data/models/solver4.pkl"):
        pass

    def predict_from_model(self, task):
        print(task)
        task_type, task, variants = self.process_task(task)
        result = self.compare_text_with_variants(variants, task_type)
        return result.strip()


def main():
    solver = Solver()
    task = {'id': '4', 'text': 'В каком слове допущена ошибка в постановке ударения: НЕВЕРНО выделена буква, обозначающая ударный гласный звук? Выпишите это слово. \n созЫв\n Отзыв (посла)\n добелА\n оптОвый\n тубдиспансЕр', 'attachments': [],      'question': {'type': 'text', 'max_length': 30, 'recommended_length': 20, 'restriction': 'word'}, 'solution': {'correct': 'отзыв'}, 'score': 1}
    print(solver.predict_from_model(task))

    task = {'id': '4', 'text': 'В каком слове допущена ошибка в постановке ударения: НЕВЕРНО выделена буква, обозначающая ударный гласный звук? Выпишите это слово. \n озлОбить\n отозвалАсь\n нАчавшись\n донЕльзя\n принЯвшись', 'attachments': []     , 'question': {'type': 'text', 'max_length': 30, 'recommended_length': 20, 'restriction': 'word'}, 'solution': {'correct': 'начавшись'}, 'score': 1}
    print(solver.predict_from_model(task))


if __name__ == "__main__":
    main()
