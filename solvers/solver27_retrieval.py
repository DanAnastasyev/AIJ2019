# coding: utf-8

import re
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
import random

from string import punctuation
from razdel import sentenize
from solvers.utils import fix_spaces


_SENTENCE_START_PATTERN = re.compile('\(\d\d?\)')
_AUTHOR_PATTERN = re.compile(
    '\(\s*(По|по)?\s*(По|по)?\s*([A-ZА-ЯЁ\u0301\u0306]\.\s*и?\s*(?:[A-ZА-ЯЁ\u0301\u0306]\.)?'
    '\s*[A-ZА-ЯЁ\u0301\u0306][-A-ZА-ЯЁa-zа-яё\u0301\u0306]+)[^)]*\)'
)
_AUTHOR_FALLBACK_PATTERN = re.compile('((?:[A-ZА-ЯЁ\u0301\u0306][-A-ZА-ЯЁa-zа-яё\u0301\u0306]*\s*){1,3})')


class Solver(object):
    def __init__(self, problems_path='data/good_essays.json', arguments_path='data/arguments.json'):
        self._init_use('data/use-multilingual-large')
        self._init_problems(problems_path)
        self._init_arguments(arguments_path)

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

    def _init_problems(self, problems_path):
        with open(problems_path) as f:
            essays = json.load(f)

        self._predefined_problems, problem_embedded_texts = [], []
        for essay in essays:
            for hint in essay['hints']:
                problem = fix_spaces(hint['problem']).strip(punctuation + ' ')
                problem = problem.split(' ', 1)[1]
                comment = fix_spaces(hint['comment']).strip(punctuation + ' ')
                if len(comment) < 5:
                    comment = ''
                hint = {
                    'theme': hint['theme'].strip(),
                    'comment': comment,
                    'problem': problem,
                    'author_position': fix_spaces(hint['author_position']).strip(punctuation + ' ') + '.',
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

    def _init_arguments(self, arguments_path):
        with open(arguments_path) as f:
            arguments = json.load(f)

        self._predefined_arguments, argument_embedded_texts = [], []
        for argument in arguments:
            for example in argument['examples']:
                example = fix_spaces(example).strip(punctuation + ' ') + '.'

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
        if hint['comment'].strip(punctuation + ' '):
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

    @staticmethod
    def _get_long_phrases(sentences):
        long_phrases = []
        for i, sent in enumerate(sentences):
            phrase = (([sentences[i - 1]] if i > 0 else [])
                      + [sent]
                      + ([sentences[i + 1]] if i + 1 < len(sentences) else []))
            long_phrases.append(' '.join(phrase))
        return long_phrases

    def _get_best_match(self, sentences, part_embeddings):
        is_best_theme = np.zeros(len(part_embeddings))

        # TODO: Batchify me
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
        sent_embeddings = self._session.run(self._embedded_text, feed_dict={self._text_input: sentences})
        similarities = np.dot(sent_embeddings, problem_embedding)

        best_sentence_indices = np.argsort(similarities)[-5:]
        problem_sentences = []
        for index in best_sentence_indices:
            if len(sentences[index].split()) > 6:
                problem_sentences.append(sentences[index])

        if len(problem_sentences) < 2:
            long_phrases = self._get_long_phrases(sentences)

            sent_embeddings = self._session.run(self._embedded_text, feed_dict={self._text_input: long_phrases})
            similarities = np.dot(sent_embeddings, problem_embedding)

            best_indices = np.argpartition(similarities, kth=-2)[-2:]
            while len(problem_sentences) < 2:
                problem_sentences.append(long_phrases[best_indices])

        return problem_sentences

    def generate(self, text):
        sentences, preposition, author = self._parse_text(text)

        problem_index = self._get_best_match(sentences, self._problem_embeddings)
        problem = self._predefined_problems[problem_index]

        problem_sentences = self._extract_problem_sentences(sentences, self._problem_embeddings[problem_index])

        argument_index = np.dot(self._argument_embeddings, self._problem_embeddings[problem_index]).argmax()
        argument = self._predefined_arguments[argument_index]

        text = []

        intro_sentences = [
            'Есть темы мимолетные, а есть вечные. Не счесть, сколько уже раз писатели касались темы {}, сколько строк посвятили ей.',
            'Не раз и не два писатели посвящали себя теме {}.',
            'Можно ли прожить жизнь и не быть затронутым темой {}?'
        ]
        text.append(random.choice(intro_sentences).format(problem['theme']))

        if preposition:
            text.append('Автору данного произведения, {}, приходится столкнуться с проблемой {}.'.format(
                        author, problem['problem']))
            author_name = author
        else:
            text.append('Автор данного произведения, {}, задумывается над проблемой {}.'.format(
                        author, problem['problem']))
            author_name = 'автору'

        if problem['comment']:
            text.append('Это наводит на вопрос: {}.'.format(problem['comment']))
        text.append('\n')

        comment_start = [
            'Рассуждая об этом, автор приводит несколько аргументов.',
            'В тексте легко найти фразы, поясняющие позицию автора.',
            'Автор последовательно излагает свою точку зрения.'
        ]

        my_position = [
            'Откровенно говоря, я не могу уверенно сказать, что сам когда-либо задумывался над данной проблемой. '
            'Но теперь, прочитав данный отрывок, я могу сказать, что я целиком и полностью согласен с ним. '
            'Больше того, мне уже даже кажется, что где-то в глубине души я всегда рассуждал именно так. '
            'И я даже легко вспоминаю другое произведение, в котором поднимается схожий вопрос. {}\n',

            'Не могу не согласиться с такой позицией. Конечно же, автор прав в своих суждениях. '
            'Чтобы подчеркнуть эту правоту, я хотел бы упомянуть другое произведение, в котором идут рассуждения в схожем ключе. {}\n',

            'В глубине души я бы хотел отвергнуть авторскую позицию. Опровергнуть эти выводы из одного лишь духа противоречия! '
            'Но нет... Нужно признать: автор прав. Не только лишь аргументы из данного текста - вся мировая литература против меня. '
            'Как не вспомнить тут ещё одно произведение. {}\nС такими авторитетами не поспоришь (да и не хочется).'
        ]

        conclusion = [
            'Подводя итоги, хочу сказать, что мне действительно запало в душу данное произведение и, '
            'подобно герою известного романа Сэлинджера, мне хотелось бы позвонить {} и обсудить это.',

            'Подводя итоги, хочу написать, что данная позиция делает честь {}, '
            'а прекрасный язык усиливает эффект от прочитанного.'
        ]

        dots = ['.' if sent[-1] == '.' else '' for sent in problem_sentences]
        problem_sentences = [sent.strip('.') for sent in problem_sentences]

        text.extend([
            random.choice(comment_start),
            'Автор пишет: «{}»{}'.format(problem_sentences[0], dots[0]),
            'Еще яснее раскрывается мысль в следующих словах: «{}»{}'.format(problem_sentences[1], dots[1]),
            'В итоге авторская позиция ясна: {}\n'.format(problem['author_position']),
            random.choice(my_position).format(argument),
            random.choice(conclusion).format(author_name)
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
    solver = Solver(problems_path='data/good_essays.json')

    import os

    with open('predicted_essays.txt', 'w') as f_out:
        for path in os.listdir('dataset/test/')[2:]:
            with open('dataset/test/' + path) as f:
                for task in json.load(f):
                    if int(task['id']) == 27:
                        f_out.write(task['text'] + '\n')
                        f_out.write(solver.predict_from_model(task) + '\n' + ('-' * 90) + '\n')


if __name__ == "__main__":
    main()
