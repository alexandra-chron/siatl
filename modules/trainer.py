import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from utils.logging import epoch_progress
from utils.training import save_checkpoint


class Trainer:
    """
    An abstract class representing a Trainer.
    A Trainer object is responsible for handling the training process and
    provides various helper methods.

    All other trainers should subclass it.
    All subclasses should override process_batch, which handles the way
    you feed the input data to the model and performs a forward pass.
    """
    def __init__(self, model, train_loader, valid_loader,
                 criterion,
                 optimizers, config, device,
                 valid_loader_train_set=None,
                 batch_end_callbacks=None,
                 unfreeze_embed=None,
                 unfreeze_rnn=None):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_loader_train_set = valid_loader_train_set
        self.criterion = criterion
        self.optimizers = optimizers
        self.device = device
        self.unfreeze_embed = unfreeze_embed
        self.unfreeze_rnn = unfreeze_rnn
        self.config = config
        self.log_interval = self.config["log_interval"]
        self.batch_size = self.config["batch_size"]
        self.checkpoint_interval = self.config["checkpoint_interval"]
        self.clip = self.config["model"]["clip"]

        if batch_end_callbacks is None:
            self.batch_end_callbacks = []
        else:
            self.batch_end_callbacks = [c for c in batch_end_callbacks
                                        if callable(c)]
        if not isinstance(self.optimizers, (tuple, list)):
            self.optimizers = [self.optimizers]
        self.epoch = 0
        self.step = 0
        self.progress_log = None
        if self.train_loader:
            if isinstance(self.train_loader, (tuple, list)):
                self.train_set_size = len(self.train_loader[0].dataset)
            else:
                self.train_set_size = len(self.train_loader.dataset)

        if isinstance(self.valid_loader, (tuple, list)):
            self.val_set_size = len(self.valid_loader[0].dataset)
        else:
            self.val_set_size = len(self.valid_loader.dataset)

    def _roll_seq(self, x, dim=1, shift=1):
        length = x.size(dim) - shift

        seq = torch.cat([x.narrow(dim, shift, length),
                         torch.zeros_like(x[:, :1])], dim)

        return seq

    def _anneal(self, param):
        x = np.linspace(param[0], param[1], num=15)
        return np.exp(x).tolist()

    def _seq_loss(self, logits, labels):
        loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)),
                              labels.contiguous().view(-1))
        return loss

    def process_batch(self, *args):
        raise NotImplementedError

    def aggregate_losses(self, batch_losses, weights=None):
        """
        This function computes a weighted sum of the models losses
        Args:
            batch_losses(torch.Tensor, tuple):

        Returns:
            loss_sum (int): the aggregation of the constituent losses
            loss_list (list, int): the constituent losses

        """
        if isinstance(batch_losses, (tuple, list)):
            loss_sum = sum(batch_losses)
            loss_list = [x.item() for x in batch_losses]
        else:
            loss_sum = batch_losses
            loss_list = batch_losses.item()
        return loss_sum, loss_list

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

            # zero gradients
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            if isinstance(self.train_loader, (tuple, list)):
                batch = list(map(lambda x:
                                 list(map(lambda y: y.to(self.device), x)),
                                 batch))
            else:
                batch = list(map(lambda x: x.to(self.device), batch))

            batch_losses = self.process_batch(*batch)

            # aggregate the losses into a single loss value
            loss_sum, loss_list = self.aggregate_losses(batch_losses)
            losses.append(loss_list)

            # back-propagate
            loss_sum.backward()

            if self.clip is not None:
                # clip_grad_norm_(self.model.parameters(), self.clip)
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
                    c(i_batch, loss_list)

        return np.array(losses).mean(axis=0)

    def eval_epoch(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []

        if isinstance(self.valid_loader, (tuple, list)):
            iterator = zip(*self.valid_loader)
        else:
            iterator = self.valid_loader

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):

                # move all tensors in batch to the selected device
                if isinstance(self.valid_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                batch_losses = self.process_batch(*batch)

                # aggregate the losses into a single loss value
                loss, _losses = self.aggregate_losses(batch_losses)
                losses.append(_losses)

        return np.array(losses).mean(axis=0)

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
        }

        return state

    def checkpoint(self, name=None, timestamp=False, tags=None, verbose=False):

        if name is None:
            name = self.config["name"]

        return save_checkpoint(self.get_state(),
                               name=name, tag=tags, timestamp=timestamp,
                               verbose=verbose)
