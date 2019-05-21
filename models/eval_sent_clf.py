import argparse

from models.run_clf import load_dataset

from utils.opts import train_options

from torch import nn
from torch.utils.data import DataLoader
from models.clf_trainer import ClfTrainer
from modules.modules import Classifier
from utils.datasets import SortedSampler, ClfDataset, ClfCollate
from utils.nlp import twitter_preprocessor
from utils.training import load_checkpoint, f1_macro
from utils.transfer import load_state_dict_subset

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", required=True, help="checkpoint"
                                                              "of model to test")
parser.add_argument("-i", "--input", required=False,
                    default='SCV2_aux_ft_gu.yaml',
                    help="config file of input data")

args = parser.parse_args()
checkpoint = args.checkpoint
input_config = args.input

opts, config = train_options(input_config)
if config["preprocessor"] == "twitter":
    preprocessor = twitter_preprocessor()
else:
    preprocessor = None

checkpoint = load_checkpoint(checkpoint)
vocab = checkpoint["vocab"]


dataset = load_dataset(config, test=True)
X_test, y_test = dataset


test_set = ClfDataset(X_test, y_test,
                       vocab=checkpoint.vocab, preprocess=preprocessor,
                       vocab_size=config["vocab"]["size"],
                       seq_len=config["data"]["seq_len"])

src_lengths = [len(x) for x in test_set.data]

ntokens = len(test_set.vocab)
val_sampler = SortedSampler(src_lengths)
val_loader = DataLoader(test_set, sampler=val_sampler,
                        batch_size=config["batch_size"],
                        num_workers=opts.cores, collate_fn=ClfCollate())
model = Classifier(ntokens, len(set(test_set.labels)), **config["model"])
model.to(opts.device)

clf_criterion = nn.CrossEntropyLoss()
lm_criterion = nn.CrossEntropyLoss(ignore_index=0)

trainer = ClfTrainer(model, None, val_loader,
                     (lm_criterion, clf_criterion),
                     config, opts.device,
                     unfreeze_embed=config["unfreeze_embed"],
                     unfreeze_rnn=config["unfreeze_rnn"])
load_state_dict_subset(model, checkpoint["model"])
print(model)
val_loss, y, y_pred = trainer.eval_epoch(val_set=True)
print("\n")
print("test cls loss is {}".format(val_loss[1]))
print("\n")
print("test lm loss is {}".format(val_loss[0]))
print("\n")
print("f1 test is {}".format(f1_macro(y, y_pred)))