# coding: utf-8

import re
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

from string import punctuation
from razdel import sentenize


_SENTENCE_START_PATTERN = re.compile('\(\d\d?\)')
_AUTHOR_PATTERN = re.compile(
    '\(\s*(По|по)?\s*(По|по)?\s*([A-ZА-ЯЁ\u0301\u0306]\.\s*и?\s*(?:[A-ZА-ЯЁ\u0301\u0306]\.)?'
    '\s*[A-ZА-ЯЁ\u0301\u0306][-A-ZА-ЯЁa-zа-яё\u0301\u0306]+)[^)]*\)'
)
_AUTHOR_FALLBACK_PATTERN = re.compile('((?:[A-ZА-ЯЁ\u0301\u0306][-A-ZА-ЯЁa-zа-яё\u0301\u0306]*\s*){1,3})')
_REPEATING_SPACE_PATTERN = re.compile('\s+')


class Solver(object):
    def __init__(self):
        self._init_use('data/use-multilingual-large')
        self._init_problems()
        self._init_arguments()

    def _init_use(self, path):
        g = tf.Graph()
        with g.as_default():
            self._text_input = tf.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module(path)
            self._embedded_text = embed(self._text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

        # Initialize session.
        self._session = tf.Session(graph=g)
        self._session.run(init_op)

    def _init_problems(self):
        with open('data/good_essays.json') as f:
            essays = json.load(f)

        self._predefined_problems, problem_embedded_texts = [], []
        for essay in essays:
            for hint in essay['hints']:
                hint = {
                    'theme': hint['theme'].strip(),
                    'comment': _REPEATING_SPACE_PATTERN.sub(' ', hint['comment']).strip(),
                    'problem': _REPEATING_SPACE_PATTERN.sub(' ', hint['problem']).strip(),
                    'author_position': _REPEATING_SPACE_PATTERN.sub(' ', hint['author_position']).strip(),
                }
                problem_embedded_texts.append(self._get_hint_text(hint))
                self._predefined_problems.append({
                    key: value[0].lower() + value[1:] if value else ''
                    for key, value in hint.items()
                })

        self._problem_embeddings = self._session.run(
            self._embedded_text, feed_dict={self._text_input: problem_embedded_texts}
        )
        assert len(self._predefined_problems) == len(problem_embedded_texts) == len(self._problem_embeddings)

    def _init_arguments(self):
        with open('data/arguments.json') as f:
            arguments = json.load(f)

        self._predefined_arguments, argument_embedded_texts = [], []
        for argument in arguments:
            for example in argument['examples']:
                example = _REPEATING_SPACE_PATTERN.sub(' ', example).strip(punctuation + ' ') + '.'

                argument_embedded_text = argument['name'].strip(punctuation + ' ') + '.'
                argument_embedded_text = argument_embedded_text[0].upper() + argument_embedded_text[1:]
                argument_embedded_text += ' ' + example

                self._predefined_arguments.append(example)
                argument_embedded_texts.append(argument_embedded_text)

        self._argument_embeddings = self._session.run(
            self._embedded_text, feed_dict={self._text_input: argument_embedded_texts}
        )

        assert len(self._predefined_arguments) == len(argument_embedded_texts) == len(self._argument_embeddings)

    @staticmethod
    def _get_hint_text(hint):
        hint_text_parts = [hint['problem'].strip(punctuation + ' ') + '.']
        if hint['comment'].strip(punctuation):
            hint_text_parts.append(hint['comment'].strip(punctuation + ' ') + '.')
        hint_text_parts.append(hint['author_position'].strip(punctuation + ' ') + '.')
        return ' '.join(hint_text_parts)

    @staticmethod
    def _extract_author(last_sentence):
        match = _AUTHOR_PATTERN.search(last_sentence)
        if match:
            last_sentence = last_sentence[:match.start()].strip()
            author = match.group(3).strip()
            preposition = match.group(1)
        else:
            splitted_last_sentence = list(sentenize(last_sentence))
            last_sentence = splitted_last_sentence[0].text.strip()
            appended_info = ' '.join(part.text for part in splitted_last_sentence[1:])
            author_match = _AUTHOR_FALLBACK_PATTERN.search(appended_info)
            if author_match:
                author = author_match.group(1).strip()
            else:
                author = None
            preposition = None

        if author and (len(author) < 4 or '.' not in author):
            author = None

        return last_sentence, preposition, author

    @classmethod
    def _parse_text(cls, text):
        text = text.replace('(З)', '(3)').replace('(ЗЗ)', '(33)').replace('(ЗО)', '(30)')
        if not _SENTENCE_START_PATTERN.search(text):
            sentences = [sent.text for sent in sentenize(text)]
        else:
            sentences = []
            prev_match = None
            for match in _SENTENCE_START_PATTERN.finditer(text):
                if prev_match:
                    sentences.append(text[prev_match.end(): match.start()].strip())
                prev_match = match
            sentences.append(text[prev_match.end():].strip())

        sentences[-1], preposition, author = cls._extract_author(sentences[-1])

        return sentences, preposition, author

    def _get_best_match(self, sentences, part_embeddings):
        is_best_theme = np.zeros(len(part_embeddings))

        for i, sent in enumerate(sentences):
            cur_text = (([sentences[i - 1]] if i > 0 else [])
                + [sent]
                + ([sentences[i + 1]] if i + 1 < len(sentences) else []))

            use_inputs = [' '.join(cur_text)]
            cur_text_embedding = self._session.run(
                self._embedded_text, feed_dict={self._text_input: use_inputs}
            )

            similarities = np.dot(cur_text_embedding, part_embeddings.T)[0]
            is_best_theme[similarities.argmax()] += 1

        return is_best_theme.argmax()

    def _extract_problem_sentences(self, sentences, problem_embedding):
        similarities = []
        for sentence in sentences:
            sent_embedding = self._session.run(self._embedded_text, feed_dict={self._text_input: [sentence]})
            similarities.append(np.dot(sent_embedding, problem_embedding)[0])

        return np.argpartition(similarities, kth=-2)[-2:]

    def generate(self, text):
        sentences, preposition, author = self._parse_text(text)

        problem_index = self._get_best_match(sentences, self._problem_embeddings)
        problem = self._predefined_problems[problem_index]

        problem_sentences = self._extract_problem_sentences(sentences, self._problem_embeddings[problem_index])

        argument_index = np.dot(self._argument_embeddings, self._problem_embeddings[problem_index]).argmax()
        argument = self._predefined_arguments[argument_index]

        text = ['Есть темы мимолетные, а есть вечные. Не счесть, сколько уже раз писатели'
                ' касались темы {}, сколько строк посвятили ей.'.format(problem['theme'])]

        if preposition:
            text.append('Автору данного произведения, {}, приходится столкнуться с проблемой {}.'.format(
                        author, problem['problem']))
            author_name = author
        else:
            text.append('Автор данного произведения, {}, задумывается над проблемой {}.'.format(
                        author, problem['problem']))
            author_name = 'автору'

        if problem['comment']:
            text.append('Он задается вопросом: {}.'.format(problem['comment']))
        text.append('\n')

        text.extend([
            'Рассуждая над проблемой, автор приводит два взаимодополняющих примера.',
            'Он пишет: "{}".'.format(sentences[problem_sentences[0]]),
            'Его позицию дополнительно подчеркивают такие слова, как: "{}".'.format(sentences[problem_sentences[1]]),
            'В итоге авторская позиция ясна: {}\n'.format(problem['author_position']),
            'Откровенно говоря, я не могу уверенно сказать, что сам когда-либо задумывался над данной проблемой. '
            'Но теперь, прочитав данное произведение автора, я могу сказать, что я целиком и полностью согласен с ним. '
            'Больше того, мне уже даже кажется, что где-то в глубине души я всегда рассуждал именно так. '
            'И я даже легко вспоминаю другое произведение, в котором поднимается схожий вопрос.',
            argument + '\n',
            'Подводя итоги, хочу сказать, что мне действительно запало в душу данное произведение и, '
            'подобно герою известного романа Сэлинджера, мне хотелось бы позвонить {} и обсудить это.'.format(author_name)
        ])

        text = ' '.join(text)
        text = [part.strip() for part in text.split('\n')]
        return '\n'.join(text)

    def fit(self, tasks):
        pass

    def load(cls, path):
        pass

    def save(self, path):
        pass

    def predict_from_model(self, task):
        return self.generate(task['text'])


def main():
    solver = Solver()

    import os

    with open('predicted_essays.txt', 'w') as f_out:
        for path in os.listdir('public_set/test/'):
            with open('public_set/test/' + path) as f:
                for task in json.load(f):
                    if int(task['id']) == 27:
                        f_out.write(task['text'] + '\n')
                        f_out.write(solver.predict_from_model(task) + '\n' + ('-' * 90) + '\n')


if __name__ == "__main__":
    main()
