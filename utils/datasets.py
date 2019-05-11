import collections
import math
import os
from collections import Counter

import numpy
import torch
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


def vectorize(tokens, vocab):
    """
    Covert array of tokens, to array of ids
    Args:
        tokens (list): list of tokens
        vocab (Vocab):
    Returns:  list of ids
    """
    ids = []
    for token in tokens:
        if token in vocab.tok2id:
            ids.append(vocab.tok2id[token])
        else:
            ids.append(vocab.tok2id[vocab.UNK])
    return ids


def iterate_data(data):
    if isinstance(data, str):
        assert os.path.exists(data)
        with open(data, "r") as f:
            for line in f:
                if len(line.strip()) > 1:
                    yield line

    elif isinstance(data, collections.Iterable):
        for x in data:
            yield x


def read_corpus(file, tokenize, vocab=None, vocab_size=None):
    if vocab is not None:
        _vocab = vocab
    else:
        _vocab = Vocab()

    _data = []
    for line in iterate_data(file):
        tokens = tokenize(line)
        _vocab.read_tokens(tokens)
        if len(tokens) <= 1:
            continue
        else:
            _data.append(tokens)

    if vocab is None:
        _vocab.build(vocab_size)

    return _vocab, _data


class LMCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, sort=False, batch_first=True, has_attributes=False):
        self.sort = sort
        self.batch_first = batch_first
        self.has_attributes = has_attributes

    def pad_collate(self, batch):
        inputs = pad_sequence([torch.LongTensor(x[0]) for x in batch],
                              self.batch_first)
        targets = pad_sequence([torch.LongTensor(x[1]) for x in batch],
                               self.batch_first)

        if self.has_attributes:
            attributes = torch.FloatTensor([x[2] for x in batch])
            lengths = torch.LongTensor([x[3] for x in batch])
            return inputs, targets, attributes, lengths
        else:
            lengths = torch.LongTensor([x[2] for x in batch])
            return inputs, targets, lengths

    def __call__(self, batch):
        return self.pad_collate(batch)


class ClfCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, sort=False, batch_first=True):
        self.sort = sort
        self.batch_first = batch_first

    def pad_collate(self, batch):
        # inputs = pad_sequence([torch.FloatTensor(x[0]) for x in batch],
        #                       self.batch_first)
        inputs = pad_sequence([torch.LongTensor(x[0]) for x in batch],
                              self.batch_first)
        labels = torch.LongTensor([x[1] for x in batch])
        lengths = torch.LongTensor([x[2] for x in batch])
        return inputs, labels, lengths

    def __call__(self, batch):
        return self.pad_collate(batch)


class Vocab(object):
    """
    The Vocab Class, holds the vocabulary of a corpus and
    mappings from tokens to indices and vice versa.
    """

    def __init__(self, pad="<pad>", sos="<sos>", eos="<eos>", unk="<unk>"):
        self.PAD = pad
        self.SOS = sos
        self.EOS = eos
        self.UNK = unk

        self.vocab = Counter()

        self.tok2id = dict()
        self.id2tok = dict()

        self.size = 0

    def read_tokens(self, tokens):
        self.vocab.update(tokens)

    def trim(self, size):
        self.tok2id = dict()
        self.id2tok = dict()
        self.build(size)

    def __add_token(self, token):
        index = len(self.tok2id)
        self.tok2id[token] = index
        self.id2tok[index] = token

    def from_file(self, file, skip=0):
        self.__add_token(self.PAD)
        self.__add_token(self.SOS)
        self.__add_token(self.EOS)
        self.__add_token(self.UNK)

        lines = open(file + ".vocab").readlines()[skip:]
        for line in lines:
            token = line.split()[0]
            self.__add_token(token)

    def to_file(self, file):
        raise NotImplementedError

    def build(self, size=None):
        self.__add_token(self.PAD)
        self.__add_token(self.SOS)
        self.__add_token(self.EOS)
        self.__add_token(self.UNK)

        for w, k in self.vocab.most_common(size):
            self.__add_token(w)

        self.size = len(self)

    def __len__(self):
        return len(self.tok2id)


class SortedSampler(Sampler):
    """
    Defines a strategy for drawing samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, lengths, descending=False):
        self.lengths = lengths
        self.desc = descending

    def __iter__(self):

        if self.desc:
            return iter(numpy.flip(numpy.array(self.lengths).argsort(), 0))
        else:
            return iter(numpy.array(self.lengths).argsort())

    def __len__(self):
        return len(self.lengths)


class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=False, drop_last=True):
        sorted_indices = numpy.array(lengths).argsort()
        num_sections = math.ceil(len(lengths) / batch_size)
        self.batches = numpy.array_split(sorted_indices, num_sections)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(self.batches[i]
                        for i in torch.randperm(len(self.batches)))
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class LMDataset(Dataset):
    def __init__(self, input,
                 attributes=None,
                 seq_len=0,
                 preprocess=None,
                 vocab=None, vocab_size=None,
                 verbose=True):
        """
        Dataset for single corpus. Used for tasks like language modeling.

        Args:
            preprocess (callable): preprocessing callable, which takes as input
                a string and returns a list of tokens
            input (str, list): the path to the data file, or a list of samples.
            attributes (numpy.ndarray): list of attributes
                shape: samples x n_attributes
            seq_len (int): sequence length -
                if stateful==True: refers to size of backpropagation through time.
                the dataset will be split to small sequences of bptt size.
            vocab (Vocab): a vocab instance. If None, then build a new one
                from the Datasets data.
            vocab_size(int): if given, then trim the vocab to the given number.
            verbose(bool): print useful statistics about the dataset.
        """

        self.input = input
        self.attributes = attributes

        if preprocess is not None:
            self.preprocess = preprocess

        # tokenize the dataset
        self.vocab, self.data = read_corpus(
            input, self.preprocess, vocab, vocab_size)

        if seq_len == 0:
            self.seq_len = max([len(x) for x in self.data])
        else:
            self.seq_len = seq_len

        if verbose:
            print(self)
            print()

    def __str__(self):

        props = []
        if isinstance(self.input, str):
            props.append(("source", os.path.basename(self.input)))

        props.append(("size", len(self)))
        props.append(("vocab size", len(self.vocab)))
        props.append(("unique tokens", len(self.vocab.vocab)))
        props.append(("max seq length", self.seq_len))

        if self.attributes is not None:
            props.append(("attributes", len(self.attributes[0])))

        return tabulate([[x[1] for x in props]], headers=[x[0] for x in props])

    def truncate(self, n):
        self.data = self.data[:n]

    @staticmethod
    def preprocess(text, lower=True):
        if lower:
            text = text.lower()
        return text.split()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.data[index][:self.seq_len]
        targets = inputs[1:][:self.seq_len]

        length = len(inputs)
        inputs_vec = vectorize(inputs, self.vocab)
        targets_vec = vectorize(targets, self.vocab)

        if self.attributes is not None:
            attributes = numpy.array(self.attributes[index])
            return inputs_vec, targets_vec, attributes, length
        else:
            return inputs_vec, targets_vec, length


class ClfDataset(Dataset):
    def __init__(self, input,
                 labels=None,
                 seq_len=0,
                 preprocess=None,
                 vocab=None, vocab_size=None,
                 verbose=True, dataset=None):
        """
        Args:
            preprocess (callable): preprocessing callable, which takes as input
                a string and returns a list of tokens
            input (str, list): the path to the data file, or a list of samples.
            labels (numpy.ndarray): list of labels
            seq_len (int): sequence length
            vocab (Vocab): a vocab instance. If None, then build a new one
                from the Datasets data.
            vocab_size(int): if given, then trim the vocab to the given number.
            verbose(bool): print useful statistics about the dataset.
        """
        self.input = input
        self.labels = labels
        self.dataset = dataset
        self.vocab = vocab
        if preprocess is not None:
            self.preprocess = preprocess

        self.vocab, self.data = read_corpus(input, self.preprocess, vocab,
                                            vocab_size)

        if seq_len == 0:
            self.seq_len = max([len(x) for x in self.data])
        else:
            self.seq_len = seq_len

        if verbose:
            print(self)
            print()

        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

    def __str__(self):
        props = []
        if isinstance(self.input, str):
            props.append(("source", os.path.basename(self.input)))
        props.append(("size", len(self)))
        props.append(("vocab size", len(self.vocab)))
        props.append(("unique tokens", len(self.vocab.vocab)))
        props.append(("max seq length", self.seq_len))
        return tabulate([[x[1] for x in props]], headers=[x[0] for x in props])

    def truncate(self, n):
        self.data = self.data[:n]

    @staticmethod
    def preprocess(text, lower=True):
        if lower:
            text = text.lower()
        return text.split()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        input = self.data[index][:self.seq_len]
        label = self.labels[index]
        length = len(input)
        inputs_vec = vectorize(input, self.vocab)

        return inputs_vec, label, length
