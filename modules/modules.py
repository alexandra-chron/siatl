import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.layers import Embed, SelfAttention
from modules.embed_regularize import embedded_dropout
from modules.locked_dropout import LockedDropout


class RecurrentHelper:
    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

    def last_timestep(self, outputs, lengths, bi=False):
        if bi:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    def pad_outputs(self, out_packed, max_length):

        out_unpacked, _lengths = pad_packed_sequence(out_packed,
                                                     batch_first=True)

        # pad to initial max length
        pad_length = max_length - out_unpacked.size(1)
        out_unpacked = F.pad(out_unpacked, (0, 0, 0, pad_length))
        return out_unpacked

    @staticmethod
    def hidden2vocab(output, projection):
        # output_unpacked.size() = batch_size, max_length, hidden_units
        # flat_outputs = (batch_size*max_length, hidden_units),
        # which means that it is a sequence of *all* the outputs (flattened)
        flat_output = output.contiguous().view(output.size(0) * output.size(1),
                                               output.size(2))

        # the sequence of all the output projections
        decoded_flat = projection(flat_output)

        # reshaped the flat sequence of decoded words,
        # in the original (reshaped) form (3D tensor)
        decoded = decoded_flat.view(output.size(0), output.size(1),
                                    decoded_flat.size(1))
        return decoded

    @staticmethod
    def sort_by(lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        if lengths.data.is_cuda:
            reverse_idx = reverse_idx.cuda()

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):

            if iterable is None:
                return None

            if len(iterable.shape) > 1:
                return iterable[sorted_idx][reverse_idx]
            else:
                return iterable

        def unsort(iterable):

            if iterable is None:
                return None

            if len(iterable.shape) > 1:
                return iterable[reverse_idx][original_idx][reverse_idx]
            else:
                return iterable

        return sorted_lengths, sort, unsort


def transfer_weights(target, source):
    target_params = target.named_parameters()
    source_params = source.named_parameters()

    dict_target_params = dict(target_params)

    for name, param in source_params:
        if name in dict_target_params:
            dict_target_params[name].data.copy_(param.data)


def tie_weights(target, source):
    target_params = target.named_parameters()
    source_params = source.named_parameters()

    dict_target_params = dict(target_params)

    for name, param in source_params:
        if name in dict_target_params:
            setattr(target, name, getattr(source, name))


class RNNModule(nn.Module, RecurrentHelper):
    def __init__(self, input_size,
                 rnn_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.,
                 pack=True, last=False):
        """
        A simple RNN Encoder, which produces a fixed vector representation
        for a variable length sequence of feature vectors, using the output
        at the last timestep of the RNN.
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        """
        super(RNNModule, self).__init__()

        self.pack = pack
        self.last = last

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=rnn_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # the dropout "layer" for the output of the RNN

        # define output feature size
        self.feature_size = rnn_size

        # double if bidirectional
        if bidirectional:
            self.feature_size *= 2

    def reorder_hidden(self, hidden, order):
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, order, :], hidden[1][:, order, :]
        else:
            hidden = hidden[:, order, :]

        return hidden

    def forward(self, x, hidden=None, lengths=None):

        batch, max_length, feat_size = x.size()

        if lengths is not None and self.pack:

            ###############################################
            # sorting
            ###############################################
            lenghts_sorted, sorted_i = lengths.sort(descending=True)
            _, reverse_i = sorted_i.sort()

            x = x[sorted_i]

            if hidden is not None:
                hidden = self.reorder_hidden(hidden, sorted_i)

            ###############################################
            # forward
            ###############################################
            packed = pack_padded_sequence(x, lenghts_sorted, batch_first=True)

            self.rnn.flatten_parameters()
            out_packed, hidden = self.rnn(packed, hidden)

            out_unpacked, _lengths = pad_packed_sequence(out_packed,
                                                         batch_first=True)

            ###############################################
            # un-sorting
            ###############################################
            outputs = out_unpacked[reverse_i]
            hidden = self.reorder_hidden(hidden, reverse_i)

        else:
            self.rnn.flatten_parameters()
            outputs, hidden = self.rnn(x, hidden)

        if self.last:
            return outputs, hidden, self.last_timestep(outputs, lengths,
                                                       self.rnn.bidirectional)

        return outputs, hidden


class LangModel(nn.Module, RecurrentHelper):
    def __init__(self, ntokens, **kwargs):
        super(LangModel, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens
        self.emb_size = kwargs.get("emb_size", 100)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.embed_dropout = kwargs.get("embed_dropout", .0)
        self.rnn_size = kwargs.get("rnn_size", 100)
        self.rnn_layers = kwargs.get("rnn_layers", 1)
        self.rnn_dropout = kwargs.get("rnn_dropout", .0)
        self.decode = kwargs.get("decode", False)
        self.tie_weights = kwargs.get("tie_weights", False)
        self.pack = kwargs.get("pack", True)

        ############################################
        # Layers
        ############################################
        self.embed = Embed(ntokens, self.emb_size,
                           noise=self.embed_noise,
                           dropout=self.embed_dropout)

        self.encoder = RNNModule(input_size=self.emb_size,
                                 rnn_size=self.rnn_size,
                                 num_layers=self.rnn_layers,
                                 bidirectional=False,
                                 pack=self.pack)

        self.decoder = nn.Linear(self.rnn_size, ntokens)
        if self.tie_weights:
            self.decoder.weight = self.embed.embedding.weight
            if self.rnn_size != self.emb_size:
                self.down = nn.Linear(self.rnn_size, self.emb_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        if self.encoder.rnn.mode == 'LSTM':
            return (weight.new_zeros(self.rnn_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.rnn_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.rnn_layers, bsz, self.rnn_size)

    def forward(self, src, hidden=None, lengths=None):
        embeds = self.embed(src)

        outputs, hidden = self.encoder(embeds, hidden, lengths)

        if self.tie_weights and self.rnn_size != self.emb_size:
            outputs = self.down(outputs)

        logits = self.hidden2vocab(outputs, self.decoder)
        return logits, outputs, hidden


class Classifier(nn.Module, RecurrentHelper):
    def __init__(self, ntokens, nclasses, **kwargs):
        super(Classifier, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens
        self.emb_size = kwargs.get("emb_size", 100)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.embed_dropout = kwargs.get("embed_dropout", .0)
        self.bottom_rnn_size = kwargs.get("bottom_rnn_size", 100)
        self.bottom_rnn_layers = kwargs.get("bottom_rnn_layers", 1)
        self.bottom_rnn_dropout = kwargs.get("bottom_rnn_dropout", .0)
        self.top_rnn_size = kwargs.get("top_rnn_size", 100)
        self.top_rnn_layers = kwargs.get("top_rnn_layers", 1)
        self.top_rnn_dropout = kwargs.get("top_rnn_dropout", .0)
        self.tie_weights = kwargs.get("tie_weights", False)
        self.pack = kwargs.get("pack", True)
        self.attention_dropout = kwargs.get("attention_dropout", .0)
        self.attention_layers = kwargs.get("attention_layers", 1)
        self.dropout = kwargs.get("dropout", 0.1)
        self.dropouti = kwargs.get("dropouti", 0.1)
        self.dropouth = kwargs.get("dropouth", 0.1)
        self.dropoute = kwargs.get("dropoute", 0.1)
        self.wdrop = kwargs.get("wdrop", 0.0)
        self.att = kwargs.get("has_att", False)
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(self.dropouti)
        self.hdrop = nn.Dropout(self.dropouth)
        self.drop = nn.Dropout(self.dropout)
        self.top_bidir = kwargs.get("top_rnn_bidir", False)
        self.new_lm = kwargs.get("new_lm", False)
        ############################################
        # Layers
        ############################################
        self.embed = Embed(ntokens, self.emb_size,
               noise=self.embed_noise, dropout=self.embed_dropout)
        if self.att:
            last = False
        else:
            last = True

        self.bottom_rnn = RNNModule(input_size=self.emb_size,
                                    rnn_size=self.bottom_rnn_size,
                                    num_layers=self.bottom_rnn_layers,
                                    bidirectional=False,
                                    dropout=self.bottom_rnn_dropout,
                                    pack=self.pack)
        if self.tie_weights:
            input_top_size = self.emb_size
        else:
            input_top_size = self.bottom_rnn_size

        self.top_rnn = RNNModule(input_size=input_top_size,
                                 rnn_size=self.top_rnn_size,
                                 num_layers=self.top_rnn_layers,
                                 bidirectional=self.top_bidir,
                                 dropout=self.top_rnn_dropout,
                                 pack=self.pack,
                                 last=last)
        if self.att:
            self.attention = SelfAttention(attention_size=
                                           self.top_rnn.feature_size,
                                       dropout=self.attention_dropout,
                                       layers=self.attention_layers)

        self.vocab = nn.Linear(self.bottom_rnn_size, ntokens)
        self.classes = nn.Linear(self.top_rnn.feature_size, nclasses)

        if self.tie_weights:
            self.vocab.weight = self.embed.embedding.weight
            if self.bottom_rnn_size != self.emb_size:
                self.down = nn.Linear(self.bottom_rnn_size,
                                      self.emb_size)

    def forward(self, src, lengths=None):

        # step 1: embed the sentences
        embeds = embedded_dropout(self.embed.embedding, src,
                     dropout=self.dropoute if self.training else 0)

        embeds = self.lockdrop(embeds, self.dropouti)

        # step 2: encode the sentences
        bottom_outs, _ = self.bottom_rnn(embeds, lengths=lengths)

        if self.tie_weights and self.bottom_rnn_size != self.emb_size:
            bottom_outs = self.down(bottom_outs)

        bottom_outs = self.lockdrop(bottom_outs, self.dropout)

        if self.att:
            outputs, hidden = self.top_rnn(bottom_outs, lengths=lengths)
            representations, attentions = self.attention(outputs, lengths)
            cls_logits = self.classes(representations)
        else:
            outputs, hidden, last_hidden = self.top_rnn(bottom_outs,
                                                lengths=lengths)
            cls_logits = self.classes(last_hidden)
            attentions = []
        # step 3: output layers
        lm_logits = self.hidden2vocab(bottom_outs, self.vocab)

        return lm_logits, cls_logits, attentions


class NaiveClassifier(nn.Module, RecurrentHelper):
    def __init__(self, ntokens, nclasses, attention=False, **kwargs):
        super(NaiveClassifier, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens
        self.emb_size = kwargs.get("emb_size", 100)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.embed_dropout = kwargs.get("embed_dropout", .0)
        self.bottom_rnn_size = kwargs.get("bottom_rnn_size", 100)
        self.attention_dropout = kwargs.get("attention_dropout", .0)
        self.bottom_rnn_layers = kwargs.get("bottom_rnn_layers", 1)
        self.bottom_rnn_dropout = kwargs.get("bottom_rnn_dropout", .0)
        self.tie_weights = kwargs.get("tie_weights", False)
        self.pack = kwargs.get("pack", True)
        self.att = attention
        self.attention_layers = kwargs.get("attention_layers", 1)

        ############################################
        # Layers
        ############################################
        self.embed = Embed(ntokens, self.emb_size,
                           noise=self.embed_noise, dropout=
                           self.embed_dropout)
        if self.att:
            last = False
        else:
            last = True

        self.bottom_rnn = RNNModule(input_size=self.emb_size,
                                    rnn_size=self.bottom_rnn_size,
                                    num_layers=self.bottom_rnn_layers,
                                    bidirectional=False,
                                    dropout=self.bottom_rnn_dropout,
                                    pack=self.pack,
                                    last=last)
        if self.att:
            self.attention = SelfAttention(attention_size=
                                           self.bottom_rnn_size,
                                           dropout=
                                           self.attention_dropout)

        self.classes = nn.Linear(self.bottom_rnn_size, nclasses)

    def forward(self, src, lengths=None):

        # step 1: embed the sentences
        embeds = self.embed(src)

        # step 2: encode the sentences
        if self.att:
            outputs, hidden = self.bottom_rnn(embeds, lengths=lengths)
            representations, attentions = self.attention(outputs,  lengths)
            cls_logits = self.classes(representations)
        else:
            outputs, hidden, last_hidden = self.bottom_rnn(embeds, lengths=lengths)
            cls_logits = self.classes(last_hidden)

        return cls_logits

