import argparse

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))

from models.sent_clf import sent_clf
from models.sent_clf_no_aux import sent_clf_no_aux
from utils.data_parsing import load_dataset
from utils.opts import train_options


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=False,
                    default='SCV2_aux_ft_gu.yaml',
                    help="config file of input data")
parser.add_argument("-t", "--transfer", action='store_true',
                    help="transfer from pretrained language model or train"
                         "a randomly initialized model")
parser.add_argument("-a", "--aux_loss", action='store_true',
                    help="add an auxiliary LM loss to the transferred model"
                         "or simply transfer a LM to a classifier"
                         " and fine-tune")
args = parser.parse_args()
input_config = args.input
transfer = args.transfer
aux_loss = args.aux_loss

opts, config = train_options(input_config, parser)

dataset = load_dataset(config)

if aux_loss:
    loss, accuracy, f1 = sent_clf(dataset=dataset, config=config,
                                  opts=opts, transfer=transfer)
else:
    loss, accuracy, f1 = sent_clf_no_aux(dataset=dataset, config=config,
                                         opts=opts, transfer=transfer)
