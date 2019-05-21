from itertools import chain

from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.clf_trainer import ClfTrainer
from modules.modules import Classifier
from sys_config import EXP_DIR
from utils.datasets import BucketBatchSampler, SortedSampler, ClfDataset, \
    ClfCollate
from utils.early_stopping import EarlyStopping
from utils.nlp import twitter_preprocessor
from utils.training import load_checkpoint, f1_macro, acc
from utils.transfer import dict_pattern_rename, load_state_dict_subset

####################################################################
# SETTINGS
####################################################################


def sent_clf(dataset, config, opts, transfer=False):
    from logger.experiment import Experiment

    opts.name = config["name"]
    X_train, y_train, X_val, y_val = dataset
    vocab = None
    if transfer:
        opts.transfer = config["pretrained_lm"]
        checkpoint = load_checkpoint(opts.transfer)
        config["vocab"].update(checkpoint["config"]["vocab"])
        dict_pattern_rename(checkpoint["config"]["model"],
                            {"rnn_": "bottom_rnn_"})
        config["model"].update(checkpoint["config"]["model"])
        vocab = checkpoint["vocab"]

    ####################################################################
    # Load Preprocessed Datasets
    ####################################################################
    if config["preprocessor"] == "twitter":
        preprocessor = twitter_preprocessor()
    else: preprocessor = None

    print("Building training dataset...")
    train_set = ClfDataset(X_train, y_train,
                           vocab=vocab, preprocess=preprocessor,
                           vocab_size=config["vocab"]["size"],
                           seq_len=config["data"]["seq_len"])

    print("Building validation dataset...")
    val_set = ClfDataset(X_val, y_val,
                         seq_len=train_set.seq_len, preprocess=preprocessor,
                         vocab=train_set.vocab)

    src_lengths = [len(x) for x in train_set.data]
    val_lengths = [len(x) for x in val_set.data]

    # select sampler & dataloader
    train_sampler = BucketBatchSampler(src_lengths, config["batch_size"], True)
    val_sampler = SortedSampler(val_lengths)
    val_sampler_train = SortedSampler(src_lengths)

    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              num_workers=opts.cores, collate_fn=ClfCollate())
    val_loader = DataLoader(val_set, sampler=val_sampler,
                            batch_size=config["batch_size"],
                            num_workers=opts.cores, collate_fn=ClfCollate())
    val_loader_train_dataset = DataLoader(train_set,
                                          sampler=val_sampler_train,
                                          batch_size=config["batch_size"],
                                          num_workers=opts.cores,
                                          collate_fn=ClfCollate())
    ####################################################################
    # Model
    ####################################################################
    ntokens = len(train_set.vocab)
    model = Classifier(ntokens, len(set(train_set.labels)), **config["model"])
    model.to(opts.device)

    clf_criterion = nn.CrossEntropyLoss()
    lm_criterion = nn.CrossEntropyLoss(ignore_index=0)

    embed_parameters = filter(lambda p: p.requires_grad,
                              model.embed.parameters())
    bottom_parameters = filter(lambda p: p.requires_grad,
                               chain(model.bottom_rnn.parameters(),
                                     model.vocab.parameters()))
    if config["model"]["has_att"]:
        top_parameters = filter(lambda p: p.requires_grad,
                                chain(model.top_rnn.parameters(),
                                      model.attention.parameters(),
                                      model.classes.parameters()))
    else:
        top_parameters = filter(lambda p: p.requires_grad,
                                chain(model.top_rnn.parameters(),
                                      model.classes.parameters()))

    embed_optimizer = optim.ASGD(embed_parameters, lr=0.0001)
    rnn_optimizer = optim.ASGD(bottom_parameters)
    top_optimizer = Adam(top_parameters, lr=config["top_lr"])
    ####################################################################
    # Training Pipeline
    ####################################################################

    # Trainer: responsible for managing the training process
    trainer = ClfTrainer(model, train_loader, val_loader,
                         (lm_criterion, clf_criterion),
                         [embed_optimizer,
                          rnn_optimizer,
                          top_optimizer],
                         config, opts.device,
                         valid_loader_train_set=val_loader_train_dataset,
                         unfreeze_embed=config["unfreeze_embed"],
                         unfreeze_rnn=config["unfreeze_rnn"])

    ####################################################################
    # Experiment: logging and visualizing the training process
    ####################################################################

    exp = Experiment(opts.name, config, src_dirs=opts.source,
                     output_dir=EXP_DIR)
    exp.add_metric("ep_loss_lm", "line", "epoch loss lm",
                   ["TRAIN", "VAL"])
    exp.add_metric("ep_loss_cls", "line", "epoch loss class",
                   ["TRAIN", "VAL"])
    exp.add_metric("ep_f1", "line", "epoch f1", ["TRAIN", "VAL"])
    exp.add_metric("ep_acc", "line", "epoch accuracy", ["TRAIN", "VAL"])

    exp.add_value("epoch", title="epoch summary")
    exp.add_value("progress", title="training progress")

    ####################################################################
    # Resume Training from a previous checkpoint
    ####################################################################
    if transfer:
        print("Transferring Encoder weights ...")
        dict_pattern_rename(checkpoint["model"],
                            {"encoder": "bottom_rnn", "decoder": "vocab"})
        load_state_dict_subset(model, checkpoint["model"])
    print(model)

    ####################################################################
    # Training Loop
    ####################################################################
    best_loss = None
    early_stopping = EarlyStopping("min", config["patience"])

    for epoch in range(0, config["epochs"]):

        train_loss = trainer.train_epoch()
        val_loss, y, y_pred = trainer.eval_epoch(val_set=True)
        _, y_train, y_pred_train = trainer.eval_epoch(train_set=True)
        exp.update_metric("ep_loss_lm", train_loss[0], "TRAIN")
        exp.update_metric("ep_loss_lm", val_loss[0], "VAL")

        exp.update_metric("ep_loss_cls", train_loss[1], "TRAIN")
        exp.update_metric("ep_loss_cls", val_loss[1], "VAL")

        exp.update_metric("ep_f1", f1_macro(y_train, y_pred_train),
                          "TRAIN")
        exp.update_metric("ep_f1", f1_macro(y, y_pred), "VAL")

        exp.update_metric("ep_acc", acc(y_train, y_pred_train), "TRAIN")
        exp.update_metric("ep_acc", acc(y, y_pred), "VAL")

        print()
        epoch_log = exp.log_metrics(["ep_loss_lm", "ep_loss_cls",
                                     "ep_f1", "ep_acc"])
        print(epoch_log)
        exp.update_value("epoch", epoch_log)

        # Save the model if the val loss is the best we've seen so far.
        if not best_loss or val_loss[1] < best_loss:
            best_loss = val_loss[1]
            trainer.best_acc = acc(y, y_pred)
            trainer.best_f1 = f1_macro(y, y_pred)
            trainer.checkpoint(name=opts.name)

        if early_stopping.stop(val_loss[1]):
            print("Early Stopping (according to classification loss)....")
            break

        print("\n" * 2)
    
    return best_loss, trainer.best_acc, trainer.best_f1

