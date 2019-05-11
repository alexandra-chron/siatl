import sys
import os

from modules.trainer import Trainer

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))


class LMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, inputs, labels, lengths):
        logits, outputs, hidden = self.model(inputs, None, lengths)

        loss = self._seq_loss(logits, self._roll_seq(inputs))

        return loss
