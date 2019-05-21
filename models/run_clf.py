import argparse
from models.sent_clf import sent_clf
from utils.data_parsing import load_dataset
from utils.opts import train_options


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=False,
                    default='SCV2_aux_ft_gu.yaml',
                    help="config file of input data")
parser.add_argument("-t", "--transfer", required=False, default=True,
                    help="transfer from pretrained language model or train"
                         "a randomly initialized model")
parser.add_argument("-a", "--aux_loss", required=False, default=True,
                    help="add an auxiliary LM loss to the transferred model"
                         "or simply transfer a LM to a classifier and fine-tune")
args = parser.parse_args()
input_config = args.input
transfer = args.transfer
aux_loss = args.aux_loss
opts, config = train_options(input_config)
dataset = load_dataset(config)

if aux_loss:
    loss, accuracy, f1 = sent_clf(dataset=dataset, config=config,
                                  opts=opts, transfer=transfer)
# else:
#     loss, accuracy, f1 = clf_naive(yaml=input_config, transfer=transfer)
