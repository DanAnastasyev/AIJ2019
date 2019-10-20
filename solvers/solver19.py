# -*- coding: utf-8 -*-

import re
import stanfordnlp

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
    def __init__(self, models_dir='/home/dan-anastasev/Documents/stanfordnlp_resources/'):
        self._pipeline = stanfordnlp.Pipeline(lang='ru', models_dir=models_dir)

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

    def predict_from_model(self, task):
        sentence, positions = self._convert_sentence(self._get_sentence(task))

        assert task['question']['type'] == 'multiple_choice'
        assert len(task['question']['choices']) == len(positions)
        assert [int(choice['id']) == i + 1 for i, choice in enumerate(task['question']['choices'])]

        return self._find_best_positions(sentence, positions)

    @classmethod
    def load(cls, path):
        return cls()

    def save(self, path):
        pass


if __name__ == "__main__":
    text = "Расставьте знаки препинания: укажите цифру(-ы), на месте которой(-ых) в предложении должна(-ы) стоять запятая(-ые). Все с участием спрашивали об её здоровье и слышали в ответ (1) что она занята необходимыми делами (2) по окончании (3) которых (4) непременно явится."

    solver = Solver()
    print(solver._get_sentence({'text': text}))
