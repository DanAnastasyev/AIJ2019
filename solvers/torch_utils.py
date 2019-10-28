# coding: utf-8

import attr
import logging
import numpy as np
import os
import shutil
import torch
import torch.nn as nn

from operator import attrgetter

logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class Field(object):
    name = attr.ib()
    dtype = attr.ib()


class BatchCollector(object):
    def __init__(self, matrix_fields, vector_fields, device):
        self._matrix_fields = matrix_fields
        self._vector_fields = vector_fields
        self._device = device

    def __call__(self, samples):
        def batchify_matrix(get_field, dtype):
            tensor = np.zeros((len(samples), max_length), dtype=dtype)

            for sample_id, sample in enumerate(samples):
                data = get_field(sample)
                tensor[sample_id, :len(data)] = data

            return torch.from_numpy(tensor)

        def batchify_vector(get_field, dtype):
            return torch.as_tensor([get_field(sample) for sample in samples], dtype=dtype)

        max_length = max(len(sample) for sample in samples)

        batch = {
            field.name: batchify_matrix(attrgetter(field.name), field.dtype).to(self._device)
            for field in self._matrix_fields
        }
        for field in self._vector_fields:
            batch[field.name] = batchify_vector(attrgetter(field.name), field.dtype).to(self._device)

        return batch


def save_model(model, optimizer, scheduler, model_path):
    model_dir = os.path.dirname(model_path)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, model_path + '_tmp')
    shutil.copyfile(model_path + '_tmp', model_path)
    os.remove(model_path + '_tmp')


def _load_model(model, optimizer, scheduler, model_path, strict, device):
    try:
        if os.path.isfile(model_path):
            logger.info('Loading model from %s, strict = %s', model_path, strict)
            state = torch.load(model_path, map_location=device)
            if 'model' in state:
                model.load_state_dict(state['model'], strict=strict)
                if optimizer:
                    optimizer.load_state_dict(state['optimizer'], strict=strict)
                if scheduler:
                    scheduler.load_state_dict(state['scheduler'], strict=strict)
            else:
                model.load_state_dict(state, strict=strict)
            return True
    except Exception as error:
        logger.warning('Things went wrong in the model loading:\n[%s] %s', error.__class__, error)
    return False


def load_model(model, optimizer, scheduler, model_path, device, strict=True):
    return _load_model(model, optimizer, scheduler, model_path + '_tmp', strict=strict, device=device) or \
        _load_model(model, optimizer, scheduler, model_path, strict=strict, device=device)


class F1ScoreCounter(object):
    def __init__(self, name=None, compact=False):
        self.name = name
        self.true_positives_count = 0.
        self.false_positives_count = 0.
        self.false_negatives_count = 0.
        self._compact = compact

    def update(self, predictions, labels):
        self.true_positives_count += ((predictions == 1).float() * (labels == 1).float()).sum().item()
        self.false_positives_count += ((predictions == 1).float() * (labels == 0).float()).sum().item()
        self.false_negatives_count += ((predictions == 0).float() * (labels == 1).float()).sum().item()

    @property
    def value(self):
        if self.true_positives_count + self.false_positives_count != 0:
            precision = self.true_positives_count / (self.true_positives_count + self.false_positives_count)
        else:
            precision = 0.

        if self.true_positives_count + self.false_negatives_count != 0.:
            recall = self.true_positives_count / (self.true_positives_count + self.false_negatives_count)
        else:
            recall = 0.

        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0. else 0.

        return precision, recall, f1

    def __str__(self):
        prefix = '{}: '.format(self.name) if self.name else 'F1: '
        if self._compact:
            return prefix + '{:.2%}'.format(self.value[-1])
        return prefix + '({:.2%} / {:.2%} / {:.2%})'.format(*self.value)


class AccuracyCounter(object):
    def __init__(self, name=None, masked_values=None):
        self.name = name
        self.correct_count = 0.
        self.total_count = 0.
        self._masked_values = masked_values or []

    def update(self, predictions, labels):
        mask = torch.ones_like(labels, dtype=torch.bool)
        for masked_value in self._masked_values:
            mask &= labels != masked_value

        self.correct_count += ((predictions == labels).float() * mask.float()).sum()
        self.total_count += mask.sum()

    @property
    def value(self):
        return self.correct_count / self.total_count

    def __str__(self):
        prefix = '{}: '.format(self.name) if self.name else 'Acc: '
        return prefix + '{:.2%}'.format(self.value)


class ConsoleProgressBar(object):
    def __init__(self, total):
        self._step = 0
        self._total_steps = total

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def update(self):
        self._step += 1

    def set_description(self, text):
        if self._step % 100 == 0:
            logger.info(text + ' | [{}/{}]'.format(self._step, self._total_steps))

    def refresh(self):
        pass


class ModelTrainer(object):
    def __init__(self, model, optimizer, scheduler=None, use_tqdm=False, model_path=None, clip_norm=5.):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._use_tqdm = use_tqdm
        self._model_path = model_path
        self._clip_norm = clip_norm
        self._global_step = 0

    def on_epoch_begin(self, is_train, name, batches_count):
        """
        Initializes metrics
        """
        self._epoch_loss = 0.
        self._is_train = is_train
        self._name = name
        self._batches_count = batches_count

        self._model.train(is_train)

    def on_epoch_end(self):
        """
        Outputs final metrics
        """
        return '{:>5s} Loss = {:.5f}'.format(
            self._name, self._epoch_loss / self._batches_count
        )

    def on_batch(self, batch):
        """
        Performs forward and (if is_train) backward pass with optimization, updates metrics
        """

        loss, metrics_info = self.forward_pass(batch)
        self._epoch_loss += loss.item()

        if self._is_train:
            self.backward_pass(loss)
            self._global_step += 1

            if self._global_step % 1000 == 0 and self._model_path:
                save_model(self._model, self._optimizer, self._scheduler, self._model_path)

        return '{:>5s} Loss = {:.5f}, '.format(self._name, loss.item()) + metrics_info

    def forward_pass(self, batch):
        raise NotImplementedError

    def backward_pass(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_norm)
        self._optimizer.step()
        if self._scheduler:
            self._scheduler.step()

    def fit(self, train_iter, train_batches_per_epoch=None, epochs_count=1,
            val_iter=None, val_batches_per_epoch=None):

        train_batches_per_epoch = train_batches_per_epoch or len(train_iter)
        if val_iter is not None:
            val_batches_per_epoch = val_batches_per_epoch or len(val_iter)

        for epoch in range(epochs_count):
            name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
            self._do_epoch(iter(train_iter), is_train=True,
                           batches_count=train_batches_per_epoch,
                           name=name_prefix + 'Train:')

            if val_iter is not None:
                self._do_epoch(iter(val_iter), is_train=False,
                               batches_count=val_batches_per_epoch,
                               name=name_prefix + '  Val:')

        if self._model_path:
            save_model(self._model, self._optimizer, self._scheduler, self._model_path)

    def _do_epoch(self, data_iter, is_train, batches_count, name=None):
        self.on_epoch_begin(is_train, name, batches_count=batches_count)

        progress_bar_class = ConsoleProgressBar
        if self._use_tqdm:
            try:
                from tqdm import tqdm
                tqdm.get_lock().locks = []

                progress_bar_class = tqdm
            except:
                pass

        with torch.autograd.set_grad_enabled(is_train):
            with progress_bar_class(total=batches_count) as progress_bar:
                try:
                    for _ in range(batches_count):
                        batch = next(data_iter)
                        batch_progress = self.on_batch(batch)

                        progress_bar.update()
                        progress_bar.set_description(batch_progress)
                except StopIteration:
                    pass
                epoch_progress = self.on_epoch_end()
                progress_bar.set_description(epoch_progress)
                progress_bar.refresh()
