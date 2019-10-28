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
import torch.nn.functional as F

from string import punctuation
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.optimization import AdamW, WarmupLinearSchedule

from solvers.torch_utils import ModelTrainer, Field, BatchCollector, F1ScoreCounter, AccuracyCounter


_SPACES_FIX_PATTERN = re.compile('\s+')

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


def init_bert(bert_path):
    bert_config = BertConfig.from_json_file(os.path.join(bert_path, 'bert_config.json'))
    return BertModel(bert_config)


def load_bert(bert_path):
    bert_model = init_bert(bert_path)

    state_dict = torch.load(os.path.join(bert_path, 'pytorch_model.bin'))
    new_state_dict = OrderedDict()
    for key, tensor in state_dict.items():
        if key.startswith('bert'):
            new_state_dict[key[5:]] = tensor
        else:
            new_state_dict[key] = tensor
    missing_keys, unexpected_keys = bert_model.load_state_dict(new_state_dict, strict=False)

    for key in missing_keys:
        print('Key {} is missing in the bert checkpoint!'.format(key))
    for key in unexpected_keys:
        print('Key {} is unexpected in the bert checkpoint!'.format(key))

    bert_model.eval()
    return bert_model


def load_bert_tokenizer(bert_path):
    return BertTokenizer.from_pretrained(os.path.join(bert_path, 'vocab.txt'), do_lower_case=False)


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


def get_examples_from_task(task, solver, tokenizer):
    choices, conditions = solver.parse_task(task)
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

        text = _SPACES_FIX_PATTERN.sub(' ', choices[choice_index]['text'])

        tokenization = tokenizer.encode_plus(text, add_special_tokens=True)
        assert len(tokenization['input_ids']) == len(tokenization['token_type_ids'])

        yield Example(
            text=text,
            token_ids=tokenization['input_ids'],
            segment_ids=tokenization['token_type_ids'],
            mask=[1] * len(tokenization['input_ids']),
            error_type=error_type,
            error_type_id=_ERRORS.index(error_type)
        )


def _read_examples(data_path, solver, tokenizer):
    examples = []
    for path in os.listdir(data_path):
        with open(os.path.join(data_path, path)) as f:
            for task in json.load(f):
                if int(task['id']) == 8:
                    examples.extend(get_examples_from_task(task, solver, tokenizer))

    return examples


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


class Classifier(nn.Module):
    def __init__(self, bert, bert_output_dim=768):
        super(Classifier, self).__init__()

        self._bert = bert
        self._dropout = nn.Dropout(0.3)
        self._predictor = nn.Linear(bert_output_dim, len(_ERRORS))

    def forward(self, batch):
        outputs, pooled_outputs = self._bert(
            input_ids=batch['token_ids'],
            attention_mask=batch['mask'],
            token_type_ids=batch['segment_ids'],
        )

        status_logits = self._predictor(self._dropout(pooled_outputs))

        loss = 0.
        if 'error_type_id' in batch:
            loss = F.cross_entropy(status_logits, batch['error_type_id'])

        return {
            'error_type_logits': status_logits,
            'loss': loss
        }


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


def _pack_model(model_path):
    checkpoint = torch.load(model_path)
    torch.save({'model': checkpoint['model']}, model_path)


def _fit_classifier(train_examples, test_examples):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    batch_collector = _get_batch_collector(device)

    train_batch_size, test_batch_size = 32, 32
    epoch_count = 10

    bert = load_bert('data/')
    model = Classifier(bert, bert_output_dim=768).to(device)

    train_loader = DataLoader(
        train_examples, batch_size=train_batch_size,
        shuffle=True, collate_fn=batch_collector
    )
    test_loader = DataLoader(
        test_examples, batch_size=test_batch_size,
        shuffle=False, collate_fn=batch_collector
    )

    print(next(iter(train_loader)))

    num_total_steps = int(len(train_examples) / train_batch_size * epoch_count)
    num_warmup_steps = int(num_total_steps * 0.1)

    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    model_path = 'data/models/solver8.pt'
    trainer = ClassifierTrainer(
        model, optimizer, scheduler, use_tqdm=True, model_path=model_path
    )
    trainer.fit(
        train_iter=train_loader, train_batches_per_epoch=int(len(train_examples) / train_batch_size),
        epochs_count=epoch_count, val_iter=test_loader,
        val_batches_per_epoch=int(len(test_examples) / test_batch_size)
    )
    _pack_model(model_path)


class Solver():
    def __init__(self):
        self._model = None
        self._tokenizer = None
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
        pass

    def load(self, path="data/models/solver8.pt", bert_path='data'):
        self._tokenizer = load_bert_tokenizer(bert_path)

        model_checkpoint = torch.load('data/models/solver8.pt', map_location=self._device)
        self._model = Classifier(init_bert(bert_path))
        self._model.load_state_dict(model_checkpoint['model'])
        self._model.to(self._device)
        self._model.eval()

    def save(self, path="data/models/solver8.pkl"):
        pass

    def predict_from_model(self, task):
        choices, conditions = self.parse_task(task)
        examples = []
        for choice in choices:
            text = _SPACES_FIX_PATTERN.sub(' ', choice['text'])
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


def train():
    bert_tokenizer = load_bert_tokenizer('data/')

    solver = Solver()
    train_examples = _read_examples('public_set/train', solver, bert_tokenizer)
    test_examples = _read_examples('public_set/test', solver, bert_tokenizer)

    train_examples = [example for example in train_examples if example not in test_examples]

    print('Examples:')
    for example in train_examples[0:100:10]:
        print(example)

    print('Examples count:', len(train_examples))
    for idx, error in enumerate(_ERRORS):
        print('Examples for error {}: {}'.format(error, sum(1 for ex in train_examples if ex.error_type_id == idx)))

    _fit_classifier(train_examples, test_examples)


if __name__ == "__main__":
    train()
