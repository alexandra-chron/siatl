import argparse
import os
import signal
import sys

import torch

from sys_config import MODEL_CNF_DIR, BASE_DIR
from utils.config import load_config

signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))


def train_options(def_config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=def_config)
    parser.add_argument('--name')
    parser.add_argument('--desc')
    parser.add_argument('--resume')

    parser.add_argument('--device', default="auto")
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--source', nargs='*',
                        default=["models", "modules", "utils"])

    args = parser.parse_args()
    config = load_config(os.path.join(MODEL_CNF_DIR, args.config))

    if args.name is None:
        args.name = os.path.splitext(args.config)[0]

    # config["name"] = args.name
    config["desc"] = args.desc

    if args.device == "auto":
        args.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

    if args.source is None:
        args.source = []

    args.source = [os.path.join(BASE_DIR, dir) for dir in args.source]

    for arg in vars(args):
        print("{}:{}".format(arg, getattr(args, arg)))
    print()

    return args, config
