# coding: utf-8

import attr
import json
import numpy as np
import os
import pymorphy2
import random
import re
import torch
import torch.nn as nn

from string import punctuation
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, WarmupLinearSchedule

from solvers.torch_utils import (
    ModelTrainer, Field, BatchCollector, F1ScoreCounter, AccuracyCounter,
    BertMulticlassClassifier, init_bert, load_bert, load_bert_tokenizer
)

from solvers.utils import fix_spaces


_ERRORS = [
    'деепричастный оборот',
    'косвенный речь',
    'несогласованный приложение',
    'однородный член',
    'причастный оборот',
    'связь подлежащее сказуемое',
    'сложноподчинённый',
    'сложный',
    'соотнесённость глагольный форма',
    'форма существительное',
    'числительное',
    'None'
]


@attr.s(frozen=True)
class Example(object):
    text = attr.ib()
    token_ids = attr.ib()
    segment_ids = attr.ib()
    mask = attr.ib()
    error_type = attr.ib()
    error_type_id = attr.ib()

    def __len__(self):
        return len(self.token_ids)


@attr.s(frozen=True)
class TrainConfig(object):
    learning_rate = attr.ib(default=1e-5)
    train_batch_size = attr.ib(default=32)
    test_batch_size = attr.ib(default=32)
    epoch_count = attr.ib(default=10)
    warm_up = attr.ib(default=0.1)


def _get_optimizer(model, train_size, config):
    num_total_steps = int(train_size / config.train_batch_size * config.epoch_count)
    num_warmup_steps = int(num_total_steps * config.warm_up)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    return optimizer, scheduler


def _get_batch_collector(device, is_train=True):
    if is_train:
        vector_fields = [Field('error_type_id', torch.long)]
    else:
        vector_fields = []

    return BatchCollector(
        matrix_fields=[
            Field('token_ids', np.int64),
            Field('segment_ids', np.int64),
            Field('mask', np.int64),
        ],
        vector_fields=vector_fields,
        device=device
    )


class ClassifierTrainer(ModelTrainer):
    def on_epoch_begin(self, *args, **kwargs):
        super(ClassifierTrainer, self).on_epoch_begin(*args, **kwargs)

        self._f1_scores = [
            F1ScoreCounter(str(i), compact=True) for i in range(len(_ERRORS) - 1)
        ]
        self._accuracies = [
            AccuracyCounter(),
            AccuracyCounter(masked_values=[len(_ERRORS) - 1])
        ]

    def on_epoch_end(self):
        info = super(ClassifierTrainer, self).on_epoch_end()

        return '{}, {}'.format(info, ', '.join(str(score) for score in self._f1_scores + self._accuracies))

    def forward_pass(self, batch):
        outputs = self._model(batch)

        predictions = outputs['error_type_logits'].argmax(-1)
        assert predictions.shape == batch['error_type_id'].shape

        for i in range(len(_ERRORS) - 1):
            self._f1_scores[i].update(predictions == i, batch['error_type_id'] == i)

        self._accuracies[0].update(predictions, batch['error_type_id'])
        self._accuracies[1].update(predictions, batch['error_type_id'])

        info = ', '.join(str(score) for score in self._f1_scores + self._accuracies)
        return outputs['loss'], info


class Solver(object):
    def __init__(self, data_path='data'):
        self._model = BertMulticlassClassifier(
            init_bert(data_path), class_count=len(_ERRORS), output_name='error_type'
        )
        self._tokenizer = load_bert_tokenizer(data_path)
        self.morph = pymorphy2.MorphAnalyzer()

        use_cuda = torch.cuda.is_available()
        self._device = torch.device('cuda:0' if use_cuda else 'cpu')
        self._batch_collector = _get_batch_collector(self._device, is_train=False)

    def normalize_category(self, cond):
        """ {'id': 'A', 'text': 'ошибка в построении сложного предложения'} """
        condition = cond["text"].lower().strip(punctuation)
        condition = re.sub("[a-дabв]\)\s", "", condition).replace('членами.', "член")
        norm_cat = ""
        for token in condition.split():
            lemma = self.morph.parse(token)[0].normal_form
            if lemma not in [
                    "неправильный", "построение", "предложение", "с", "ошибка", "имя",
                    "видовременной", "видо-временной", "предложно-падежный", "падежный",
                    "неверный", "выбор", "между", "нарушение", "в", "и", "употребление",
                    "предлог", "видовременный", "временной"
                ]:
                norm_cat += lemma + ' '
        return norm_cat

    def parse_task(self, task):
        assert task["question"]["type"] == "matching"

        conditions = task["question"]["left"]
        normalized_conditions = [self.normalize_category(cond).rstrip() for cond in conditions]

        choices = []
        for choice in task["question"]["choices"]:
            choice['text'] = re.sub("[0-9]\\s?\)", "", choice["text"])
            choices.append(choice)

        return choices, normalized_conditions

    def _get_examples_from_task(self, task):
        choices, conditions = self.parse_task(task)
        if 'correct_variants' in task['solution']:
            answers = task['solution']['correct_variants'][0]
        else:
            answers = task['solution']['correct']

        choice_index_to_error_type = {
            int(answers[option]) - 1: conditions[option_index]
            for option_index, option in enumerate(sorted(answers))
        }
        choice_index_to_error_type = {
            choice_index: choice_index_to_error_type.get(choice_index, _ERRORS[-1])
            for choice_index, choice in enumerate(choices)
        }

        assert len(answers) == sum(1 for error_type in choice_index_to_error_type.values() if error_type != _ERRORS[-1])

        for choice_index, choice in enumerate(choices):
            error_type = choice_index_to_error_type[choice_index]

            text = fix_spaces(choices[choice_index]['text'])

            tokenization = self._tokenizer.encode_plus(text, add_special_tokens=True)
            assert len(tokenization['input_ids']) == len(tokenization['token_type_ids'])

            yield Example(
                text=text,
                token_ids=tokenization['input_ids'],
                segment_ids=tokenization['token_type_ids'],
                mask=[1] * len(tokenization['input_ids']),
                error_type=error_type,
                error_type_id=_ERRORS.index(error_type)
            )

    def _prepare_examples(self, tasks):
        examples = [
            example for task in tasks
            for example in self._get_examples_from_task(task)
        ]

        print('Examples:')
        for example in examples[0:100:10]:
            print(example)

        print('Examples count:', len(examples))
        for idx, error in enumerate(_ERRORS):
            print('Examples for error {}: {}'.format(error, sum(1 for ex in examples if ex.error_type_id == idx)))

        return examples

    def predict_random(self, task):
        """ Test a random choice model """
        conditions = task["question"]["left"]
        choices = task["question"]["choices"]
        pred = {}
        for cond in conditions:
            pred[cond["id"]] = random.choice(choices)["id"]
        return pred

    def predict(self, task):
        return self.predict_from_model(task)

    def fit(self, tasks):
        examples = self._prepare_examples(tasks)

        config = TrainConfig()
        self._model = BertMulticlassClassifier(
            load_bert('data/'), class_count=len(_ERRORS), output_name='error_type'
        ).to(self._device)

        batch_collector = _get_batch_collector(self._device, is_train=True)

        train_loader = DataLoader(
            examples, batch_size=config.train_batch_size,
            shuffle=True, collate_fn=batch_collector, pin_memory=False
        )
        print(next(iter(train_loader)))

        optimizer, scheduler = _get_optimizer(self._model, len(examples), config)

        trainer = ClassifierTrainer(
            self._model, optimizer, scheduler, use_tqdm=True
        )
        trainer.fit(
            train_iter=train_loader,
            train_batches_per_epoch=int(len(examples) / config.train_batch_size),
            epochs_count=config.epoch_count,
        )

    def load(self, path="data/models/solver8.pkl"):
        model_checkpoint = torch.load(path, map_location=self._device)
        self._model.load_state_dict(model_checkpoint)
        self._model.to(self._device)
        self._model.eval()

    def save(self, path="data/models/solver8.pkl"):
        torch.save(self._model.state_dict(), path)

    def predict_from_model(self, task):
        choices, conditions = self.parse_task(task)
        examples = []
        for choice in choices:
            text = fix_spaces(choice['text'])
            tokenization = self._tokenizer.encode_plus(text, add_special_tokens=True)

            examples.append(Example(
                text=text,
                token_ids=tokenization['input_ids'],
                segment_ids=tokenization['token_type_ids'],
                mask=[1] * len(tokenization['input_ids']),
                error_type=None,
                error_type_id=None
            ))

        model_inputs = self._batch_collector(examples)

        target_condition_indices = [_ERRORS.index(condition) for condition in conditions]

        pred_dict = {}
        with torch.no_grad():
            model_prediction = self._model(model_inputs)
            model_prediction = model_prediction['error_type_logits'][:, target_condition_indices]
            model_prediction = model_prediction.argmax(0)

            for i, condition in enumerate(task['question']['left']):
                pred_dict[condition['id']] = choices[model_prediction[i]]['id']

        return pred_dict


def _train():
    from utils import load_tasks

    solver = Solver()
    solver.fit(load_tasks('dataset/train/', task_num=8))
    solver.save()


def _test_apply():
    solver = Solver()
    solver.load()

    print(solver.predict_from_model({
        "id": "8",
        "text": "Установите соответствие между грамматическими ошибками и предложениями, в которых они допущены: к каждой позиции первого столбца подберите соответствующую позицию из второго столбца. ГРАММАТИЧЕСКИЕ ОШИБКИ   ПРЕДЛОЖЕНИЯА) нарушение в построении предложения с причастным оборотомБ) неправильное употребление падежной формы существительного с предлогомВ) ошибка в построении предложения с деепричастным оборотомГ) нарушение в построении предложения с однородными членамиД) нарушение связи между подлежащим и сказуемым в предложении       1) Автор рассказал об изменениях в книге, готовящейся им к переизданию.2) Пони из местного цирка по вечерам катало детей.3) Мальчишка, катавшийся на велосипеде и который с него упал, сидел рядом с мамой, прикрывая разбитое колено.4) На небе не было ни одного облачка, но в воздухе чувствовался избыток влаги.5) Родители требовали, чтобы я по приезду отправил им подробный отчёт и рассказал всё в мельчайших подробностях.6) В народных представлениях власть повелевать ветрами приписывается различным божествам и мифологическим персонажам.7) Я имею поручение как от судьи, так равно и от всех наших знакомых примирить вас с приятелем вашим.8) Заглянув на урок, директору представилась интересная картина.9) Беловежская пуща — наиболее крупный остаток реликтового первобытного равнинного леса, который в доисторические времена произрастал на территории Европы. Запишите в ответ цифры, расположив их в порядке, соответствующем буквам: AБВГД     ",
        "attachments": [],
        "question": {
            "type": "matching",
            "left": [
                {
                    "id": "A",
                    "text": "А) нарушение в построении предложения с причастным оборотом"
                },
                {
                    "id": "B",
                    "text": "Б) неправильное употребление падежной формы существительного с предлогом"
                },
                {
                    "id": "C",
                    "text": "В) ошибка в построении предложения с деепричастным оборотом"
                },
                {
                    "id": "D",
                    "text": "Г) нарушение в построении предложения с однородными членами"
                },
                {
                    "id": "E",
                    "text": "Д) нарушение связи между подлежащим и сказуемым в предложении       "
                }
            ],
            "choices": [
                {
                    "id": "1",
                    "text": "1) Автор рассказал об изменениях в книге, готовящейся им к переизданию"
                },
                {
                    "id": "2",
                    "text": "2) Пони из местного цирка по вечерам катало детей"
                },
                {
                    "id": "3",
                    "text": "3) Мальчишка, катавшийся на велосипеде и который с него упал, сидел рядом с мамой, прикрывая разбитое колено"
                },
                {
                    "id": "4",
                    "text": "4) На небе не было ни одного облачка, но в воздухе чувствовался избыток влаги"
                },
                {
                    "id": "5",
                    "text": "5) Родители требовали, чтобы я по приезду отправил им подробный отчёт и рассказал всё в мельчайших подробностях"
                },
                {
                    "id": "6",
                    "text": "6) В народных представлениях власть повелевать ветрами приписывается различным божествам и мифологическим персонажам"
                },
                {
                    "id": "7",
                    "text": "7) Я имею поручение как от судьи, так равно и от всех наших знакомых примирить вас с приятелем вашим"
                },
                {
                    "id": "8",
                    "text": "8) Заглянув на урок, директору представилась интересная картина"
                },
                {
                    "id": "9",
                    "text": "9) Беловежская пуща — наиболее крупный остаток реликтового первобытного равнинного леса, который в доисторические времена произрастал на территории Европы. "
                }
            ]
        },
        "solution": {
            "correct": {
                "A": "1",
                "B": "5",
                "C": "8",
                "D": "3",
                "E": "2"
            }
        },
        "score": 5
    }))


if __name__ == "__main__":
    _train()
    _test_apply()
