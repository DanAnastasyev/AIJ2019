import operator
import random
import re
import attr
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity
from solvers.utils import BertEmbedder
from solvers.solver27_retrieval import Solver as EssayGenerator

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, WarmupLinearSchedule

from solvers.torch_utils import (
    ModelTrainer, BertExample, Field, BatchCollector, AccuracyCounter,
    BertBinaryClassifier, init_bert, load_bert, load_bert_tokenizer
)


class Encoder(object):
    def __init__(self):
        g = tf.Graph()
        with g.as_default():
            self._text_input = tf.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module('data/use-multilingual-large')
            self._embedded_text = embed(self._text_input)

            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

        # Initialize session.
        self._session = tf.Session(graph=g)
        self._session.run(init_op)

    def embed(self, sentences):
        return self._session.run(self._embedded_text, feed_dict={self._text_input: sentences})


@attr.s
class Example(BertExample):
    label_id = attr.ib(default=None)

    @classmethod
    def build(cls, text, tokenizer, text_pair, label_id=None):
        example = super(Example, cls).build(text, tokenizer, text_pair)
        example.label_id = label_id

        return example


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
        vector_fields = [Field('label_id', torch.float32)]
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
        self._accuracies = [AccuracyCounter()]

    def on_epoch_end(self):
        info = super(ClassifierTrainer, self).on_epoch_end()
        return '{}, {}'.format(info, ', '.join(str(score) for score in self._accuracies))

    def forward_pass(self, batch):
        outputs = self._model(batch)

        predictions = (outputs['label_logits'] > 0.).float()
        self._accuracies[0].update(predictions, batch['label_id'])

        info = ', '.join(str(score) for score in self._accuracies)
        return outputs['loss'], info


class Solver(object):

    def __init__(self, seed=42):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._batch_collector = _get_batch_collector(self._device, is_train=False)
        self._tokenizer = load_bert_tokenizer('data/')
        self._model = BertBinaryClassifier(init_bert('data/'), output_name='label').to(self._device)
        self._encoder = Encoder()

    def fit(self, tasks):
        train_examples = [
            example
            for task in tasks
            for example in self._prepare_examples(task)
        ]

        print('Example count:', len(train_examples))
        for example in train_examples[0:100:10]:
            print(example)

        config = TrainConfig()
        self._model = BertBinaryClassifier(load_bert('data/'), output_name='label').to(self._device)

        batch_collector = _get_batch_collector(self._device, is_train=True)

        train_loader = DataLoader(
            train_examples, batch_size=config.train_batch_size,
            shuffle=True, collate_fn=batch_collector, pin_memory=False
        )
        print(next(iter(train_loader)))

        optimizer, scheduler = _get_optimizer(self._model, len(train_examples), config)

        trainer = ClassifierTrainer(
            self._model, optimizer, scheduler, use_tqdm=True
        )
        trainer.fit(
            train_iter=train_loader,
            train_batches_per_epoch=int(len(train_examples) / config.train_batch_size),
            epochs_count=config.epoch_count,
        )

    def load(self, path='data/models/solver22.pkl'):
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._model.eval()

    def save(self, path='data/models/solver22.pkl'):
        torch.save(self._model.state_dict(), path)

    def parse_task(self, task):
        assert task["question"]["type"] == "multiple_choice"

        choices = task["question"]["choices"]
        description = task["text"]

        label = "pro" # against
        if "противоречат" in description:
            label = "against"
        elif "не соответствуют" in description:
            label = "against"

        bad_strings = ["Какие из высказываний соответствуют содержанию текста?",
                       "Какие из высказываний не соответствуют содержанию текста?",
                       "Укажите номера ответов.", "Какие из высказываний противоречат содержанию текста?"
                       "Источник текста не определён."]
        if "(1)" in description:
            text = "(1)" + " ".join(description.split("(1)")[1:])
        else:
            text = "(1)" + " ".join(description.split("1)")[1:])
        for bad in bad_strings:
            text = re.sub(bad, "", text)

        return text, choices, label

    def _prepare_examples(self, task):
        text, choices, label = self.parse_task(task)

        sentences = EssayGenerator._parse_text(text)[0]
        sentences = EssayGenerator._get_long_phrases(sentences)

        sentence_embeddings = self._encoder.embed(sentences)
        choice_embeddings = self._encoder.embed([choice['text'] for choice in choices])

        most_similar_sent_indices = np.argmax(np.dot(choice_embeddings, sentence_embeddings.T), -1)
        most_similar_sentences = [sentences[ind] for ind in most_similar_sent_indices]

        if 'correct' in task['solution']:
            correct_indices = task['solution']['correct']
        else:
            correct_indices = sorted(task['solution']['correct_variants'][0])

        for sent, choice in zip(most_similar_sentences, choices):
            choice_text = re.sub('\s*\(?\d+(?:.|\))?\s*', '', choice['text'])
            yield Example.build(
                text=sent,
                tokenizer=self._tokenizer,
                text_pair=choice['text'],
                label_id=int(choice['id'] in correct_indices)
            )

    def predict(self, task):
        prediction = self.predict_from_model(task)
        if not prediction:
            prediction = self.predict_random(task)
        return prediction

    def predict_from_model(self, task):
        examples = list(self._prepare_examples(task))
        if not examples:
            return

        model_inputs = self._batch_collector(examples)
        with torch.no_grad():
            model_prediction = self._model(model_inputs)['label_logits'].cpu().numpy()

        choices = task["question"]["choices"]
        prediction = []
        for index in np.argsort(model_prediction)[::-1]:
            if len(prediction) < 2:
                prediction.append(choices[index]['id'])
            elif len(prediction) == 2 and model_prediction[index] > 0:
                prediction.append(choices[index]['id'])
        return sorted(prediction)

    def predict_random(self, task):
        choices = task["question"]["choices"]
        pred = []
        for _ in range(random.choice([2, 3])):
            choice = random.choice(choices)
            pred.append(choice["id"])
            choices.remove(choice)
        return pred


def main():
    from utils import load_tasks

    solver = Solver()
    solver.fit(load_tasks('dataset/train', 22))
    solver.save()

    for task in load_tasks('dataset/test', 22):
        print(solver.predict(task))


if __name__ == "__main__":
    main()
