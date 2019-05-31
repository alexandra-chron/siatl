import numpy
import time

import torch
from torch.nn.utils import clip_grad_norm_

from modules.trainer import Trainer
from utils.logging import epoch_progress
from utils.training import save_checkpoint


class SentClfNoAuxTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_f1 = None
        self.best_acc = None

    def process_batch(self, inputs, labels, lengths):

        cls_logits = self.model(inputs, lengths)

        cls_loss = self.criterion(cls_logits, labels)

        return [cls_loss], labels, cls_logits

    def train_epoch(self):
        """
        Train the network for one epoch and return the average loss.
        * This will be a pessimistic approximation of the true loss
        of the network, as the loss of the first batches will be higher
        than the true.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.train()
        losses = []

        self.epoch += 1
        epoch_start = time.time()

        if isinstance(self.train_loader, (tuple, list)):
            iterator = zip(*self.train_loader)
        else:
            iterator = self.train_loader

        for i_batch, batch in enumerate(iterator, 1):

            self.step += 1

            if self.config["gu"]:
                if self.epoch < self.unfreeze_rnn:
                    self.optimizers[2].zero_grad()
                elif self.epoch < self.unfreeze_embed:
                    self.optimizers[2].zero_grad()
                    self.optimizers[1].zero_grad()
                else:
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()
            else:
                # zero gradients
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

            if isinstance(self.train_loader, (tuple, list)):
                batch = list(map(lambda x:
                                 list(map(lambda y: y.to(self.device), x)),
                                 batch))
            else:
                batch = list(map(lambda x: x.to(self.device), batch))

            batch_loss, _, _ = self.process_batch(*batch)

            # aggregate the losses into a single loss value
            loss_sum, loss_list = self.aggregate_losses(batch_loss)
            losses.append(loss_list)

            # back-propagate
            loss_sum.backward()

            if self.config["gu"]:
                # we have more than 1 optimizers (3 to be exact)
                if self.clip is not None:

                    if self.epoch < self.unfreeze_rnn:
                        clip_grad_norm_((p for group in
                                         self.optimizers[2].param_groups
                                         for p in group['params']),
                                        self.clip)
                    elif self.epoch < self.unfreeze_embed:
                        clip_grad_norm_((p for group in
                                         self.optimizers[2].param_groups
                                         for p in group['params']),
                                        self.clip)
                        clip_grad_norm_((p for group in
                                         self.optimizers[1].param_groups
                                         for p in group['params']),
                                        self.clip)
                    else:
                        for optimizer in self.optimizers:
                            clip_grad_norm_(
                                (p for group in optimizer.param_groups
                                 for p in group['params']), self.clip)
                # update weights
                if self.epoch < self.unfreeze_rnn:
                    self.optimizers[2].step()
                elif self.epoch < self.unfreeze_embed:
                    self.optimizers[2].step()
                    self.optimizers[1].step()
                else:
                    for optimizer in self.optimizers:
                        optimizer.step()

            else:
                if self.clip is not None:
                    for optimizer in self.optimizers:
                        clip_grad_norm_((p for group in optimizer.param_groups
                                         for p in group['params']), self.clip)

                # update weights
                for optimizer in self.optimizers:
                    optimizer.step()

            if self.step % self.log_interval == 0:
                self.progress_log = epoch_progress(self.epoch, i_batch,
                                                   self.batch_size,
                                                   self.train_set_size,
                                                   epoch_start)

            for c in self.batch_end_callbacks:
                if callable(c):
                    c(i_batch, batch_loss)

        return numpy.array(losses).mean(axis=0)

    def eval_epoch(self, train_set=False, val_set=False):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()

        if val_set:
            if isinstance(self.valid_loader, (tuple, list)):
                iterator = zip(*self.valid_loader)
            else:
                iterator = self.valid_loader
        elif train_set:
            if isinstance(self.valid_loader_train_set, (tuple, list)):
                iterator = zip(*self.valid_loader_train_set)
            else:
                iterator = self.valid_loader_train_set

        labels = []
        posteriors = []
        losses = []
        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):

                # move all tensors in batch to the selected device
                if isinstance(self.valid_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                batch_losses, label, cls_logits = self.process_batch(*batch)
                labels.append(label)
                posteriors.append(cls_logits)

                # aggregate the losses into a single loss value
                loss, _losses = self.aggregate_losses(batch_losses)
                losses.append(_losses)

        posteriors = torch.cat(posteriors, dim=0)
        predicted = numpy.argmax(posteriors.cpu(), 1)
        # predicted = predicted.numpy()
        labels_array = numpy.array((torch.cat(labels, dim=0)).cpu())

        return numpy.array(losses).mean(axis=0), labels_array, predicted

    def get_state(self):
        _vocab = self.train_loader.dataset.vocab

        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "vocab": _vocab,
            "f1:": self.best_f1,
            "acc": self.best_acc
        }

        return state

    def checkpoint(self, name=None, timestamp=False, tags=None, verbose=False):

        if name is None:
            name = self.config["name"]

        return save_checkpoint(self.get_state(),
                               name=name, tag=tags, timestamp=timestamp,
                               verbose=verbose)
