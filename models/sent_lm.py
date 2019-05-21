import math
import os
import sys
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))

from utils.opts import train_options
from models.sent_lm_trainer import LMTrainer
from modules.modules import LangModel
from sys_config import EXP_DIR
from utils.datasets import LMDataset, LMCollate, BucketBatchSampler, \
    SortedSampler

####################################################################
# SETTINGS
####################################################################
opts, config = train_options("lm_20m_word.yaml")

from logger.experiment import Experiment
####################################################################
# Data Loading and Preprocessing
####################################################################

vocab = None
opts.name = config["name"]
preprocessor = None


print("Building training dataset...")
train_set = LMDataset(config["data"]["train_path"], preprocess=preprocessor,
                      vocab=vocab, vocab_size=config["vocab"]["size"],
                      seq_len=config["data"]["seq_len"])

print("Building validation dataset...")
val_set = LMDataset(config["data"]["val_path"], preprocess=preprocessor,
                    seq_len=train_set.seq_len, vocab=train_set.vocab)

src_lengths = [len(x) for x in train_set.data]
val_lengths = [len(x) for x in val_set.data]
train_sampler = BucketBatchSampler(src_lengths, config["batch_size"],
                                   True)
val_sampler = SortedSampler(val_lengths)

train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                          num_workers=opts.cores, collate_fn=LMCollate())
val_loader = DataLoader(val_set, sampler=val_sampler,
                        batch_size=config["batch_size"],
                        num_workers=opts.cores, collate_fn=LMCollate())

####################################################################
# Model
####################################################################
ntokens = len(train_set.vocab)
model = LangModel(ntokens, **config["model"]).to(opts.device)

loss_function = nn.CrossEntropyLoss(ignore_index=0)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=0.0001)

####################################################################
# Training Pipeline
####################################################################


def batch_callback(batch, losses):
    if trainer.step % config["log_interval"] == 0:
        exp.update_metric("loss", losses)
        exp.update_metric("ppl", math.exp(losses))

        losses_log = exp.log_metrics(["loss", "ppl"])
        exp.update_value("progress", trainer.progress_log + "\n" +
                         losses_log)

        # clean lines and move cursor back up N lines
        print("\n\033[K" + losses_log)
        print("\033[F" * (len(losses_log.split("\n")) + 2))


# Trainer: responsible for managing the training process
trainer = LMTrainer(model, train_loader, val_loader, loss_function,
                    [optimizer], config, opts.device,
                    batch_end_callbacks=[batch_callback])

####################################################################
# Experiment: logging and visualizing the training process
####################################################################
exp = Experiment(opts.name, config, src_dirs=opts.source, output_dir=EXP_DIR)
exp.add_metric("loss", "line")
exp.add_metric("ppl", "line", "perplexity")
exp.add_metric("ep_loss", "line", "epoch loss", ["TRAIN", "VAL"])
exp.add_metric("ep_ppl", "line", "epoch perplexity", ["TRAIN", "VAL"])
exp.add_value("progress", title="training progress")
exp.add_value("epoch", title="epoch summary")

####################################################################
# Training Loop
####################################################################
best_loss = None
for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch()
    val_loss = trainer.eval_epoch()
    exp.update_metric("ep_loss", train_loss, "TRAIN")
    exp.update_metric("ep_loss", val_loss, "VAL")
    exp.update_metric("ep_ppl", math.exp(train_loss), "TRAIN")
    exp.update_metric("ep_ppl", math.exp(val_loss), "VAL")
    print()
    epoch_log = exp.log_metrics(["ep_loss", "ep_ppl"])
    print(epoch_log)
    exp.update_value("epoch", epoch_log)

    # Save the model if the val loss is the best we've seen so far.
    if not best_loss or val_loss < best_loss:
        best_loss = val_loss
        trainer.checkpoint(name=opts.name)

    print("\n" * 2)

